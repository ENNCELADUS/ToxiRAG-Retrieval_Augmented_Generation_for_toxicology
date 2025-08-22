"""
ToxiRAG Agentic Pipeline
Orchestrates LLM reasoning with retrieval for evidence-based toxicology responses.
"""

import os
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Agno imports for reasoning tools
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from agno.tools.knowledge import KnowledgeTools
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

from retriever.retriever import retrieve_relevant_docs
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of toxicology queries for different reasoning strategies."""
    MECHANISM = "mechanism"  # 机制研究
    TOXICITY = "toxicity"    # 毒性评估
    DESIGN = "design"        # 实验设计
    COMPARISON = "comparison"  # 化合物比较
    GENERAL = "general"      # 一般性问题


@dataclass
class EvidencePack:
    """Structured evidence pack with citations and metadata."""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    evidence_text: str
    citation_ids: List[str]
    confidence_score: float
    has_sufficient_evidence: bool


@dataclass
class AgenticResponse:
    """Response from agentic pipeline with reasoning and citations."""
    query: str
    response_text: str
    evidence_pack: EvidencePack
    reasoning_steps: List[str]
    citations: List[str]
    confidence_score: float
    refusal_reason: Optional[str] = None


from config.settings import settings


class _LocalDummyEmbedder:
    """Local dummy embedder to avoid external API calls during tests.
    Provides minimal get_embedding(s) interface expected by LanceDb.
    """
    def __init__(self, dim: int = 3072):
        self.dim = dim
        self.id = "dummy-embedding"

    def get_embedding(self, text: str):
        return [0.0] * self.dim

    def get_embeddings(self, texts):
        return [[0.0] * self.dim for _ in texts]

    @property
    def dimensions(self) -> int:
        return self.dim


class ToxiRAGAgent:
    """Main agentic orchestrator for ToxiRAG toxicology reasoning."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 llm_provider: str = "openai",
                 model_id: str = "gpt-5-nano",
                 max_tokens: int = 2000,
                 temperature: float = 0.1,
                 top_k_docs: int = 5):
        """Initialize the agentic pipeline."""
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.google_api_key = google_api_key or settings.google_api_key
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k_docs = top_k_docs
        
        # Initialize agents
        self.reasoning_agent = None
        self.knowledge_agent = None
        self._setup_agents()
        
        logger.info(f"ToxiRAGAgent initialized with provider: {llm_provider}")

    def _setup_agents(self) -> None:
        """Setup reasoning and knowledge agents."""
        try:
            # Set environment variables
            if self.openai_api_key:
                os.environ["OPENAI_API_KEY"] = self.openai_api_key
            if self.google_api_key:
                os.environ["GOOGLE_API_KEY"] = self.google_api_key
            
            # Create base model
            if self.llm_provider == "openai":
                model = OpenAIChat(
                    id=self.model_id, 
                    api_key=self.openai_api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            else:  # gemini
                model = Gemini(
                    id=self.model_id,
                    api_key=self.google_api_key,
                    temperature=self.temperature
                )
            
            # Setup reasoning agent
            self.reasoning_agent = Agent(
                model=model,
                tools=[ReasoningTools(add_instructions=True)],
                instructions=self._get_toxicology_instructions(),
                show_tool_calls=True,
                markdown=True
            )
            
            # Setup knowledge agent with vector DB
            # Use a mockable embedder in tests if API key missing to avoid real calls
            if self.openai_api_key and self.openai_api_key != "test-key":
                embedder = OpenAIEmbedder(
                    id=settings.openai_embed_model,
                    api_key=self.openai_api_key
                )
            else:
                # Fall back to a local dummy embedder (dimension aligned with text-embedding-3-large)
                embedder = _LocalDummyEmbedder(dim=3072)
            knowledge = TextKnowledgeBase(
                vector_db=LanceDb(
                    uri=settings.lancedb_uri,
                    table_name=settings.collection_name,
                    search_type=SearchType.hybrid,
                    embedder=embedder
                )
            )
            
            self.knowledge_agent = Agent(
                model=model,
                tools=[KnowledgeTools(
                    knowledge=knowledge,
                    think=True,
                    search=True,
                    analyze=True,
                    add_few_shot=True
                )],
                instructions=self._get_toxicology_instructions(),
                show_tool_calls=True,
                markdown=True
            )
            
            logger.info("Agents setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup agents: {str(e)}")
            raise

    def _get_toxicology_instructions(self) -> str:
        """Get specialized instructions for toxicology domain."""
        return """You are a toxicology expert specializing in liver cancer research and Traditional Chinese Medicine (TCM) compounds.

CRITICAL GUIDELINES:
1. EVIDENCE-BASED RESPONSES: Only answer based on provided evidence from toxicology papers
2. CITATION REQUIREMENTS: Always include bracketed citations in format [E1 · 实验分组与给药]
3. REFUSE INSUFFICIENT EVIDENCE: If evidence is insufficient, clearly state this and refuse to speculate
4. TOXICOLOGY FOCUS: Prioritize mechanisms, dosage, safety profiles, and experimental design
5. BILINGUAL SUPPORT: Handle both Chinese and English toxicology terminology
6. UNITS NORMALIZATION: Reference standard units (mm³ for tumor volume, mg/kg for dose)

When insufficient evidence exists, respond with:
"基于当前检索到的证据不足以回答此问题。需要更多关于[具体缺失信息]的研究数据。"

Always structure responses with:
1. 毒理学机制分析
2. 实验设计建议
3. 安全性考虑
4. 证据来源引用"""

    def classify_query(self, query: str) -> QueryType:
        """Classify the query type for appropriate reasoning strategy."""
        query_lower = query.lower()
        
        # Comparison keywords (check first to avoid conflicts with design)
        if any(word in query_lower for word in ['比较', 'compare', '对比', '差异', 'differences', 'versus', 'vs']):
            return QueryType.COMPARISON
        
        # Mechanism keywords
        elif any(word in query_lower for word in ['机制', 'mechanism', '作用机理', '分子机制']):
            return QueryType.MECHANISM
        
        # Toxicity keywords
        elif any(word in query_lower for word in ['毒性', 'toxicity', '安全性', 'safety', '副作用']):
            return QueryType.TOXICITY
        
        # Design keywords (check after comparison to avoid conflicts)
        elif any(word in query_lower for word in ['实验设计', 'design', '方案', '实验方法']):
            return QueryType.DESIGN
        
        else:
            return QueryType.GENERAL

    def decompose_query(self, query: str, query_type: QueryType) -> List[str]:
        """Decompose complex queries into focused sub-queries."""
        base_subqueries = [query]  # Always include original query
        
        if query_type == QueryType.MECHANISM:
            base_subqueries.extend([
                f"What are the molecular mechanisms involved in: {query}",
                f"Which signaling pathways are affected by: {query}",
                f"What are the cellular targets of compounds mentioned in: {query}"
            ])
        
        elif query_type == QueryType.TOXICITY:
            base_subqueries.extend([
                f"What are the safety profiles and toxicity data for: {query}",
                f"What are the dose-response relationships for: {query}",
                f"What are the adverse effects and contraindications for: {query}"
            ])
        
        elif query_type == QueryType.DESIGN:
            base_subqueries.extend([
                f"What experimental models are suitable for: {query}",
                f"What are the dosing strategies and protocols for: {query}",
                f"What measurements and endpoints should be used for: {query}"
            ])
        
        elif query_type == QueryType.COMPARISON:
            base_subqueries.extend([
                f"What are the efficacy differences between compounds in: {query}",
                f"What are the safety profiles comparing compounds in: {query}",
                f"What are the mechanistic differences for: {query}"
            ])
        
        logger.info(f"Decomposed query into {len(base_subqueries)} sub-queries")
        return base_subqueries

    async def retrieve_evidence(self, sub_queries: List[str], collection_name: str = "tcm_tox") -> List[Dict[str, Any]]:
        """Retrieve evidence for multiple sub-queries with deduplication."""
        all_docs = []
        seen_content = set()
        
        for i, sub_query in enumerate(sub_queries):
            try:
                docs = await retrieve_relevant_docs(
                    query=sub_query,
                    top_k=self.top_k_docs,
                    collection_name=collection_name
                )
                
                # Deduplicate based on content similarity
                for doc in docs:
                    content_hash = hash(doc.get('content', '')[:200])  # Use first 200 chars for dedup
                    if content_hash not in seen_content:
                        doc['sub_query_index'] = i
                        doc['sub_query'] = sub_query
                        all_docs.append(doc)
                        seen_content.add(content_hash)
                
                logger.debug(f"Retrieved {len(docs)} docs for sub-query {i}")
                
            except Exception as e:
                logger.error(f"Failed to retrieve docs for sub-query {i}: {str(e)}")
        
        logger.info(f"Total retrieved documents: {len(all_docs)} (after deduplication)")
        return all_docs

    def build_evidence_pack(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> EvidencePack:
        """Build structured evidence pack with citations."""
        if not retrieved_docs:
            return EvidencePack(
                query=query,
                retrieved_docs=[],
                evidence_text="",
                citation_ids=[],
                confidence_score=0.0,
                has_sufficient_evidence=False
            )
        
        # Generate citation IDs and build evidence text
        evidence_parts = []
        citation_ids = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            citation_id = f"E{i}"
            section_tag = doc.get('section_type', '数据')
            citation_text = f"[{citation_id} · {section_tag}]"
            
            # Format evidence with citation
            evidence_part = f"{citation_text}\n{doc.get('content', '')}"
            if doc.get('source_page'):
                evidence_part += f"\n来源页面: {doc.get('source_page')}"
            
            evidence_parts.append(evidence_part)
            citation_ids.append(citation_text)
        
        evidence_text = "\n\n".join(evidence_parts)
        
        # Calculate confidence based on relevance and coverage
        confidence_score = min(1.0, len(retrieved_docs) / 3.0)  # Full confidence with 3+ docs
        has_sufficient_evidence = len(retrieved_docs) >= 2  # Minimum 2 docs for sufficient evidence
        
        logger.info(f"Built evidence pack with {len(citation_ids)} citations, confidence: {confidence_score:.2f}")
        
        return EvidencePack(
            query=query,
            retrieved_docs=retrieved_docs,
            evidence_text=evidence_text,
            citation_ids=citation_ids,
            confidence_score=confidence_score,
            has_sufficient_evidence=has_sufficient_evidence
        )

    def generate_response(self, evidence_pack: EvidencePack, use_reasoning_tools: bool = True) -> AgenticResponse:
        """Generate evidence-based response with reasoning."""
        # Check evidence sufficiency
        if not evidence_pack.has_sufficient_evidence:
            refusal_reason = f"检索到的证据不足（{len(evidence_pack.retrieved_docs)}个文档），无法提供可靠的毒理学分析"
            return AgenticResponse(
                query=evidence_pack.query,
                response_text="基于当前检索到的证据不足以回答此问题。需要更多相关的毒理学研究数据。",
                evidence_pack=evidence_pack,
                reasoning_steps=["Evidence insufficiency check: FAILED"],
                citations=[],
                confidence_score=0.0,
                refusal_reason=refusal_reason
            )
        
        # Select agent based on use_reasoning_tools
        agent = self.knowledge_agent if use_reasoning_tools else self.reasoning_agent
        
        # Build enhanced query with evidence
        enhanced_query = f"""
        基于以下检索到的毒理学文献证据，请提供详细的分析：

        证据文档:
        {evidence_pack.evidence_text}

        用户问题: {evidence_pack.query}

        请严格基于提供的证据进行分析，包括:
        1. 相关的毒理学机制（引用具体证据）
        2. 实验设计建议（基于已有研究）
        3. 安全性考虑（基于毒性数据）
        4. 引用格式：使用提供的括号引用如 [E1 · 实验分组与给药]

        如果证据不足以回答某个方面，请明确说明需要额外的研究数据。
        """
        
        try:
            # Generate response with selected agent
            response = agent.run(enhanced_query)
            response_text = response.content
            
            # Extract reasoning steps if available
            reasoning_steps = getattr(response, 'reasoning_steps', ["Response generated with evidence-based analysis"])
            
            # Extract citations from response
            citations = re.findall(r'\[E\d+ · [^\]]+\]', response_text)
            
            logger.info(f"Generated response with {len(citations)} citations")
            
            return AgenticResponse(
                query=evidence_pack.query,
                response_text=response_text,
                evidence_pack=evidence_pack,
                reasoning_steps=reasoning_steps,
                citations=citations,
                confidence_score=evidence_pack.confidence_score,
                refusal_reason=None
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return AgenticResponse(
                query=evidence_pack.query,
                response_text=f"生成回答时出现错误: {str(e)}",
                evidence_pack=evidence_pack,
                reasoning_steps=[f"Error in response generation: {str(e)}"],
                citations=[],
                confidence_score=0.0,
                refusal_reason=f"技术错误: {str(e)}"
            )


async def create_agentic_response(query: str, 
                                config: Dict[str, Any],
                                collection_name: str = "tcm_tox",
                                use_reasoning_tools: bool = True) -> AgenticResponse:
    """
    Main entry point for creating agentic responses.
    
    Args:
        query: User's toxicology question
        config: Configuration dict with API keys and parameters
        collection_name: Vector collection name
        use_reasoning_tools: Whether to use knowledge tools for deep reasoning
        
    Returns:
        AgenticResponse with evidence-based answer and citations
    """
    try:
        # Initialize agent
        agent = ToxiRAGAgent(
            openai_api_key=config.get("openai_api_key"),
            google_api_key=config.get("google_api_key"),
            llm_provider=config.get("llm_provider", "openai"),
            model_id=config.get("selected_model", "gpt-5-nano"),
            max_tokens=config.get("max_tokens", 2000),
            temperature=config.get("temperature", 0.1),
            top_k_docs=config.get("top_k_docs", 5)
        )
        
        # Classify and decompose query
        query_type = agent.classify_query(query)
        sub_queries = agent.decompose_query(query, query_type)
        
        logger.info(f"Processing query type: {query_type.value}")
        
        # Retrieve evidence
        retrieved_docs = await agent.retrieve_evidence(sub_queries, collection_name)
        
        # Build evidence pack
        evidence_pack = agent.build_evidence_pack(query, retrieved_docs)
        
        # Generate response
        response = agent.generate_response(evidence_pack, use_reasoning_tools)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to create agentic response: {str(e)}")
        # Return error response
        return AgenticResponse(
            query=query,
            response_text=f"处理查询时出现错误: {str(e)}",
            evidence_pack=EvidencePack(
                query=query,
                retrieved_docs=[],
                evidence_text="",
                citation_ids=[],
                confidence_score=0.0,
                has_sufficient_evidence=False
            ),
            reasoning_steps=[f"Error: {str(e)}"],
            citations=[],
            confidence_score=0.0,
            refusal_reason=f"系统错误: {str(e)}"
        )
