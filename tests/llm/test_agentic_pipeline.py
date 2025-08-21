"""
Unit tests for ToxiRAG agentic pipeline.
Tests query decomposition, evidence building, and response generation with mocked LLM calls.
Also includes integration tests with real Google API when available.
"""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

from llm.agentic_pipeline import (
    ToxiRAGAgent, 
    QueryType, 
    EvidencePack, 
    AgenticResponse, 
    create_agentic_response
)

# Check for real API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HAS_REAL_API_KEYS = bool(GOOGLE_API_KEY or OPENAI_API_KEY)


class TestQueryClassification:
    """Test query type classification for toxicology domains."""
    
    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_mechanism_query_classification(self, mock_setup):
        """Test classification of mechanism-related queries."""
        # Skip agent setup during testing
        mock_setup.return_value = None
        
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        mechanism_queries = [
            "请分析玉米花粉多糖的作用机制",
            "What is the molecular mechanism of this compound?",
            "化合物的分子机制是什么？",
            "explain the mechanism of action"
        ]
        
        for query in mechanism_queries:
            query_type = agent.classify_query(query)
            assert query_type == QueryType.MECHANISM

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_toxicity_query_classification(self, mock_setup):
        """Test classification of toxicity-related queries.""" 
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        toxicity_queries = [
            "化合物的毒性如何？",
            "What are the safety concerns?",
            "请评估安全性",
            "toxicity profile of the drug"
        ]
        
        for query in toxicity_queries:
            query_type = agent.classify_query(query)
            assert query_type == QueryType.TOXICITY

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_design_query_classification(self, mock_setup):
        """Test classification of experimental design queries."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        design_queries = [
            "如何设计实验方案？",
            "experimental design for testing",
            "实验方法建议",
            "study design recommendations"
        ]
        
        for query in design_queries:
            query_type = agent.classify_query(query)
            assert query_type == QueryType.DESIGN

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_comparison_query_classification(self, mock_setup):
        """Test classification of comparison queries."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        comparison_queries = [
            "比较两种化合物的效果",
            "compare the efficacy of compounds",
            "对比不同治疗方案",
            "differences between treatments"
        ]
        
        for query in comparison_queries:
            query_type = agent.classify_query(query)
            assert query_type == QueryType.COMPARISON

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_general_query_classification(self, mock_setup):
        """Test fallback to general classification."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        general_query = "What is liver cancer?"
        query_type = agent.classify_query(general_query)
        assert query_type == QueryType.GENERAL


class TestQueryDecomposition:
    """Test query decomposition into sub-queries."""
    
    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_mechanism_query_decomposition(self, mock_setup):
        """Test decomposition of mechanism queries."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        query = "请分析玉米花粉多糖对肝癌的作用机制"
        sub_queries = agent.decompose_query(query, QueryType.MECHANISM)
        
        assert len(sub_queries) == 4  # Original + 3 mechanism-specific
        assert query in sub_queries  # Original query preserved
        assert any("molecular mechanisms" in sq for sq in sub_queries)
        assert any("signaling pathways" in sq for sq in sub_queries)
        assert any("cellular targets" in sq for sq in sub_queries)

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_toxicity_query_decomposition(self, mock_setup):
        """Test decomposition of toxicity queries."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        query = "化合物的毒性评估"
        sub_queries = agent.decompose_query(query, QueryType.TOXICITY)
        
        assert len(sub_queries) == 4  # Original + 3 toxicity-specific
        assert query in sub_queries
        assert any("safety profiles" in sq for sq in sub_queries)
        assert any("dose-response" in sq for sq in sub_queries)
        assert any("adverse effects" in sq for sq in sub_queries)

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_design_query_decomposition(self, mock_setup):
        """Test decomposition of design queries."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        query = "实验设计建议"
        sub_queries = agent.decompose_query(query, QueryType.DESIGN)
        
        assert len(sub_queries) == 4  # Original + 3 design-specific
        assert query in sub_queries
        assert any("experimental models" in sq for sq in sub_queries)
        assert any("dosing strategies" in sq for sq in sub_queries)
        assert any("measurements and endpoints" in sq for sq in sub_queries)


class TestEvidenceBuilding:
    """Test evidence pack building with real toxicology data."""
    
    def get_sample_retrieved_docs(self) -> List[Dict[str, Any]]:
        """Get sample retrieved documents based on 肝癌.md data."""
        return [
            {
                "content": "玉米花粉多糖可显著提高巨噬细胞活性，增强机体免疫功能，对肝癌H22细胞具有明显的抑制作用",
                "document_title": "玉米花粉多糖的药理药效研究",
                "section_type": "机制研究结果",
                "source_page": "p.2–3，Table 1–9"
            },
            {
                "content": "实验分组：对照组(生理盐水)，阳性对照组(5-FU 20mg/kg)，玉米花粉多糖低剂量组(100mg/kg)，中剂量组(200mg/kg)，高剂量组(400mg/kg)",
                "document_title": "玉米花粉多糖的药理药效研究", 
                "section_type": "实验分组与给药",
                "source_page": "p.1，Table 1"
            },
            {
                "content": "玉米花粉多糖高剂量组肿瘤体积为1245.2±156.8 mm³，肿瘤重量为1.02±0.15 g，抑瘤率达到58.7%",
                "document_title": "玉米花粉多糖的药理药效研究",
                "section_type": "实验结果数据",
                "source_page": "p.3，Table 2"
            }
        ]

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_evidence_pack_building_sufficient_evidence(self, mock_setup):
        """Test building evidence pack with sufficient evidence."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        query = "玉米花粉多糖对肝癌的治疗效果"
        retrieved_docs = self.get_sample_retrieved_docs()
        
        evidence_pack = agent.build_evidence_pack(query, retrieved_docs)
        
        assert evidence_pack.query == query
        assert evidence_pack.has_sufficient_evidence is True
        assert len(evidence_pack.citation_ids) == 3
        assert evidence_pack.confidence_score > 0.5
        
        # Check citation format
        expected_citations = ["[E1 · 机制研究结果]", "[E2 · 实验分组与给药]", "[E3 · 实验结果数据]"]
        assert evidence_pack.citation_ids == expected_citations
        
        # Check evidence text contains all documents
        for doc in retrieved_docs:
            assert doc["content"] in evidence_pack.evidence_text

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_evidence_pack_building_insufficient_evidence(self, mock_setup):
        """Test building evidence pack with insufficient evidence."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        query = "某未知化合物的毒性"
        retrieved_docs = [self.get_sample_retrieved_docs()[0]]  # Only 1 doc
        
        evidence_pack = agent.build_evidence_pack(query, retrieved_docs)
        
        assert evidence_pack.has_sufficient_evidence is False
        assert evidence_pack.confidence_score < 0.5
        assert len(evidence_pack.citation_ids) == 1

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_evidence_pack_building_no_evidence(self, mock_setup):
        """Test building evidence pack with no evidence."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        query = "完全无关的问题"
        retrieved_docs = []
        
        evidence_pack = agent.build_evidence_pack(query, retrieved_docs)
        
        assert evidence_pack.has_sufficient_evidence is False
        assert evidence_pack.confidence_score == 0.0
        assert len(evidence_pack.citation_ids) == 0
        assert evidence_pack.evidence_text == ""


class TestResponseGeneration:
    """Test response generation with mocked LLM calls."""
    
    @patch('llm.agentic_pipeline.Agent')
    def test_response_generation_sufficient_evidence(self, mock_agent_class):
        """Test response generation with sufficient evidence."""
        # Setup mock agent
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = """基于提供的证据分析：

1. 毒理学机制分析：玉米花粉多糖通过增强机体免疫功能发挥抗肿瘤作用 [E1 · 机制研究结果]

2. 实验设计建议：建议采用分层剂量设计，包括低中高三个剂量组 [E2 · 实验分组与给药]

3. 安全性考虑：高剂量组(400mg/kg)显示良好的抑瘤效果，抑瘤率达58.7% [E3 · 实验结果数据]

证据来源充分，建议基于现有数据进行进一步研究。"""
        
        mock_agent.run.return_value = mock_response
        mock_agent_class.return_value = mock_agent
        
        # Create agent and evidence pack
        agent = ToxiRAGAgent(openai_api_key="test-key")
        agent.reasoning_agent = mock_agent
        agent.knowledge_agent = mock_agent
        
        evidence_pack = EvidencePack(
            query="玉米花粉多糖的治疗效果",
            retrieved_docs=[{"content": "test", "section_type": "test"}],
            evidence_text="[E1 · 机制研究结果]\n测试内容",
            citation_ids=["[E1 · 机制研究结果]"],
            confidence_score=0.8,
            has_sufficient_evidence=True
        )
        
        response = agent.generate_response(evidence_pack, use_reasoning_tools=False)
        
        assert response.refusal_reason is None
        assert len(response.citations) == 3  # Should extract citations from response
        assert response.confidence_score == 0.8
        assert "玉米花粉多糖" in response.response_text

    @patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents')
    def test_response_generation_insufficient_evidence(self, mock_setup):
        """Test response generation with insufficient evidence (should refuse)."""
        mock_setup.return_value = None
        agent = ToxiRAGAgent(openai_api_key="test-key")
        
        evidence_pack = EvidencePack(
            query="未知化合物的毒性",
            retrieved_docs=[],
            evidence_text="",
            citation_ids=[],
            confidence_score=0.0,
            has_sufficient_evidence=False
        )
        
        response = agent.generate_response(evidence_pack)
        
        assert response.refusal_reason is not None
        assert "证据不足" in response.refusal_reason
        assert response.confidence_score == 0.0
        assert "基于当前检索到的证据不足" in response.response_text

    @patch('llm.agentic_pipeline.Agent')
    def test_response_generation_llm_error(self, mock_agent_class):
        """Test response generation with LLM error handling."""
        # Setup mock agent to raise exception
        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("API rate limit exceeded")
        mock_agent_class.return_value = mock_agent
        
        agent = ToxiRAGAgent(openai_api_key="test-key")
        agent.reasoning_agent = mock_agent
        agent.knowledge_agent = mock_agent
        
        evidence_pack = EvidencePack(
            query="测试查询",
            retrieved_docs=[{"content": "test"}],
            evidence_text="测试证据",
            citation_ids=["[E1 · 测试]"],
            confidence_score=0.8,
            has_sufficient_evidence=True
        )
        
        response = agent.generate_response(evidence_pack)
        
        assert response.refusal_reason is not None
        assert "API rate limit exceeded" in response.refusal_reason
        assert response.confidence_score == 0.0


class TestRealAPIIntegration:
    """Integration tests using real Google API when available."""
    
    @pytest.mark.skipif(not GOOGLE_API_KEY, reason="Google API key not available")
    def test_real_agent_initialization_google(self):
        """Test real agent initialization with Google API."""
        agent = ToxiRAGAgent(
            google_api_key=GOOGLE_API_KEY,
            llm_provider="gemini",
            max_tokens=1000,
            temperature=0.1
        )
        
        assert agent.reasoning_agent is not None
        assert agent.knowledge_agent is not None
        assert agent.llm_provider == "gemini"

    @pytest.mark.skipif(not GOOGLE_API_KEY, reason="Google API key not available") 
    def test_real_response_generation_google(self):
        """Test real response generation with Google Gemini API."""
        agent = ToxiRAGAgent(
            google_api_key=GOOGLE_API_KEY,
            llm_provider="gemini",
            max_tokens=500,
            temperature=0.1
        )
        
        # Create a simple evidence pack for testing
        evidence_pack = EvidencePack(
            query="What are the basic principles of toxicology?",
            retrieved_docs=[{
                "content": "Toxicology is the study of adverse effects of chemicals on living organisms. The basic principles include dose-response relationships, individual susceptibility, and time factors.",
                "section_type": "基础知识"
            }],
            evidence_text="[E1 · 基础知识]\nToxicology is the study of adverse effects of chemicals on living organisms. The basic principles include dose-response relationships, individual susceptibility, and time factors.",
            citation_ids=["[E1 · 基础知识]"],
            confidence_score=0.8,
            has_sufficient_evidence=True
        )
        
        response = agent.generate_response(evidence_pack, use_reasoning_tools=False)
        
        assert response.refusal_reason is None
        assert len(response.response_text) > 50  # Should generate substantial response
        assert response.confidence_score == 0.8
        print(f"Real API Response: {response.response_text[:200]}...")

    @pytest.mark.skipif(not GOOGLE_API_KEY, reason="Google API key not available")
    @pytest.mark.asyncio
    async def test_real_end_to_end_google(self):
        """Test real end-to-end flow with Google API and mocked retrieval."""
        with patch('llm.agentic_pipeline.retrieve_relevant_docs', new_callable=AsyncMock) as mock_retrieve:
            # Mock retrieval but use real LLM
            mock_retrieve.return_value = [
                {
                    "content": "Traditional Chinese Medicine compounds have shown anti-tumor effects in liver cancer studies.",
                    "document_title": "TCM研究",
                    "section_type": "研究结果",
                    "source_page": "p.1"
                },
                {
                    "content": "Experimental design should include proper control groups and dosage escalation.",
                    "document_title": "实验设计指南",
                    "section_type": "方法学",
                    "source_page": "p.2"
                }
            ]
            
            config = {
                "google_api_key": GOOGLE_API_KEY,
                "llm_provider": "gemini",
                "max_tokens": 800,
                "temperature": 0.1,
                "top_k_docs": 3
            }
            
            response = await create_agentic_response(
                query="How should I design an experiment to test TCM compounds for liver cancer?",
                config=config,
                collection_name="test_collection",
                use_reasoning_tools=False
            )
            
            assert response.refusal_reason is None
            assert len(response.evidence_pack.retrieved_docs) == 2
            assert response.confidence_score > 0.5
            assert len(response.response_text) > 100
            print(f"Real E2E Response: {response.response_text[:300]}...")

    @pytest.mark.skipif(not OPENAI_API_KEY, reason="OpenAI API key not available")
    def test_real_agent_initialization_openai(self):
        """Test real agent initialization with OpenAI API."""
        agent = ToxiRAGAgent(
            openai_api_key=OPENAI_API_KEY,
            llm_provider="openai",
            max_tokens=1000,
            temperature=0.1
        )
        
        assert agent.reasoning_agent is not None
        assert agent.knowledge_agent is not None
        assert agent.llm_provider == "openai"


class TestEndToEndFlow:
    """Test end-to-end agentic response creation."""
    
    @pytest.mark.asyncio
    @patch('llm.agentic_pipeline.retrieve_relevant_docs', new_callable=AsyncMock)
    @patch('llm.agentic_pipeline.Agent')
    async def test_create_agentic_response_success(self, mock_agent_class, mock_retrieve):
        """Test successful end-to-end response creation."""
        # Mock retrieval
        mock_retrieve.return_value = [
            {
                "content": "玉米花粉多糖显示出良好的抗肿瘤活性",
                "document_title": "玉米花粉多糖研究",
                "section_type": "结果",
                "source_page": "p.1"
            },
            {
                "content": "实验采用H22肝癌细胞移植瘤模型",
                "document_title": "玉米花粉多糖研究",
                "section_type": "方法",
                "source_page": "p.2"
            }
        ]
        
        # Mock agent
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "基于证据分析，玉米花粉多糖具有抗肿瘤活性 [E1 · 结果]"
        mock_agent.run.return_value = mock_response
        mock_agent_class.return_value = mock_agent
        
        config = {
            "openai_api_key": "test-key",
            "llm_provider": "openai",
            "max_tokens": 2000,
            "temperature": 0.1,
            "top_k_docs": 5
        }
        
        response = await create_agentic_response(
            query="玉米花粉多糖的抗肿瘤效果如何？",
            config=config,
            collection_name="test_collection"
        )
        
        assert response.refusal_reason is None
        assert len(response.evidence_pack.retrieved_docs) == 2
        assert response.confidence_score > 0.5
        assert "玉米花粉多糖" in response.response_text

    @pytest.mark.asyncio
    @patch('llm.agentic_pipeline.retrieve_relevant_docs', new_callable=AsyncMock)
    async def test_create_agentic_response_no_results(self, mock_retrieve):
        """Test response creation with no retrieval results."""
        # Mock empty retrieval
        mock_retrieve.return_value = []
        
        config = {
            "openai_api_key": "test-key",
            "llm_provider": "openai"
        }
        
        response = await create_agentic_response(
            query="完全无关的查询",
            config=config
        )
        
        assert response.refusal_reason is not None
        assert response.confidence_score == 0.0
        assert len(response.evidence_pack.retrieved_docs) == 0

    @pytest.mark.asyncio
    async def test_create_agentic_response_config_error(self):
        """Test response creation with configuration errors."""
        config = {}  # Missing required keys
        
        response = await create_agentic_response(
            query="测试查询",
            config=config
        )
        
        assert response.refusal_reason is not None
        assert ("基于当前检索到的证据不足" in response.response_text or "错误" in response.response_text)


class TestDeduplication:
    """Test document deduplication in retrieval."""
    
    @pytest.mark.asyncio
    @patch('llm.agentic_pipeline.retrieve_relevant_docs')
    async def test_retrieve_evidence_deduplication(self, mock_retrieve):
        """Test deduplication of retrieved documents."""
        # Setup mock to return duplicate content
        duplicate_docs = [
            {"content": "玉米花粉多糖具有抗肿瘤活性", "title": "研究1"},
            {"content": "玉米花粉多糖具有抗肿瘤活性", "title": "研究2"},  # Duplicate
            {"content": "实验采用H22肝癌模型", "title": "研究3"}
        ]
        
        mock_retrieve.return_value = duplicate_docs
        
        with patch('llm.agentic_pipeline.ToxiRAGAgent._setup_agents') as mock_setup:
            mock_setup.return_value = None
            agent = ToxiRAGAgent(openai_api_key="test-key")
            sub_queries = ["test query"]
        
        result_docs = await agent.retrieve_evidence(sub_queries)
        
        # Should deduplicate based on content similarity
        assert len(result_docs) == 2  # Original 3 docs deduplicated to 2
        contents = [doc["content"] for doc in result_docs]
        assert len(set(contents)) == 2  # All unique content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
