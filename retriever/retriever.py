"""
ToxiRAG Hybrid Retriever
Implements hybrid search (embedding + BM25/keywords) with evidence pack building.
"""

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import lancedb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from agno.embedder.openai import OpenAIEmbedder

from config.settings import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with scoring information."""
    id: str
    content: str
    document_title: str
    section_name: str
    section_type: str
    citation_id: str
    section_tag: str
    source_page: str
    file_path: str
    metadata: Dict[str, Any]
    
    # Scoring information
    vector_score: float
    bm25_score: float
    combined_score: float
    rank: int


@dataclass
class EvidencePack:
    """Evidence pack with formatted citations and metadata."""
    query: str
    results: List[RetrievalResult]
    evidence_text: str  # Formatted with citations
    citations: List[Dict[str, Any]]  # Citation details
    total_results: int
    filters_applied: Dict[str, Any]


class BM25Scorer:
    """BM25 scoring implementation for keyword relevance."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.idf = None
        self.doc_len = None
        self.avgdl = None
        
    def fit(self, documents: List[str]):
        """Fit BM25 on document corpus."""
        # Use TF-IDF vectorizer to get term frequencies
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # Keep all words for toxicology terms
            ngram_range=(1, 2),  # Include bigrams for compound terms
            max_features=10000
        )
        
        # Fit and get term frequency matrix
        tf_matrix = self.vectorizer.fit_transform(documents)
        
        # Calculate document lengths and average length
        self.doc_len = np.array(tf_matrix.sum(axis=1)).flatten()
        self.avgdl = np.mean(self.doc_len)
        
        # Calculate IDF values
        N = len(documents)
        df = np.array((tf_matrix > 0).sum(axis=0)).flatten()
        self.idf = np.log((N - df + 0.5) / (df + 0.5))
        
        logger.info(f"BM25 fitted on {N} documents, vocab size: {len(self.vectorizer.vocabulary_)}")
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """Calculate BM25 scores for query against documents."""
        if self.vectorizer is None:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        # Get query term frequencies
        query_vec = self.vectorizer.transform([query])
        query_terms = query_vec.toarray()[0]
        
        # Get document term frequencies
        doc_vecs = self.vectorizer.transform(documents)
        doc_tf = doc_vecs.toarray()
        
        # Calculate BM25 scores
        scores = np.zeros(len(documents))
        
        for i, doc_tf_vec in enumerate(doc_tf):
            score = 0.0
            doc_length = self.doc_len[i] if i < len(self.doc_len) else np.sum(doc_tf_vec)
            
            for j, tf in enumerate(doc_tf_vec):
                if tf > 0 and query_terms[j] > 0:
                    # BM25 formula
                    idf = self.idf[j] if j < len(self.idf) else 0
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                    score += idf * (numerator / denominator)
            
            scores[i] = score
        
        return scores


class ToxiRAGRetriever:
    """Hybrid retriever with vector similarity and BM25 scoring."""
    
    def __init__(self, 
                 lancedb_uri: Optional[str] = None,
                 table_name: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        self.lancedb_uri = lancedb_uri or settings.lancedb_uri
        self.table_name = table_name or settings.collection_name
        self.embedding_model = embedding_model or settings.openai_embed_model
        
        # Initialize embedder
        self.embedder = OpenAIEmbedder(
            id=self.embedding_model,
            api_key=settings.openai_api_key
        )
        
        # Database connection (lazy)
        self._db = None
        self._table = None
        self._bm25 = None
        self._corpus_cache = None
        
    @property
    def db(self):
        """Lazy database connection."""
        if self._db is None:
            logger.info(f"Connecting to LanceDB at: {self.lancedb_uri}")
            self._db = lancedb.connect(self.lancedb_uri)
        return self._db
    
    @property
    def table(self):
        """Lazy table connection."""
        if self._table is None:
            try:
                self._table = self.db.open_table(self.table_name)
                logger.info(f"Opened table: {self.table_name}")
            except FileNotFoundError:
                raise ValueError(f"Table {self.table_name} not found. Please ingest documents first.")
        return self._table
    
    def _prepare_bm25(self, force_refresh: bool = False):
        """Prepare BM25 scorer with current corpus."""
        if self._bm25 is not None and not force_refresh:
            return
        
        # Get all documents for BM25 training
        df = self.table.to_pandas()
        if len(df) == 0:
            logger.warning("No documents found in table for BM25 preparation")
            return
        
        # Prepare corpus (content + section info for better matching)
        corpus = []
        for _, row in df.iterrows():
            # Combine content with section metadata for richer BM25 matching
            doc_text = f"{row['content']} {row['section_name']} {row['section_type']}"
            corpus.append(doc_text)
        
        self._corpus_cache = corpus
        self._bm25 = BM25Scorer()
        self._bm25.fit(corpus)
        logger.info(f"BM25 prepared with {len(corpus)} documents")
    
    async def search(self,
                    query: str,
                    top_k: int = None,
                    vector_weight: float = 0.7,
                    bm25_weight: float = 0.3,
                    section_types: Optional[List[str]] = None,
                    document_titles: Optional[List[str]] = None,
                    min_score_threshold: float = 0.0,
                    deduplicate: bool = True) -> List[RetrievalResult]:
        """
        Hybrid search with vector similarity and BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            vector_weight: Weight for vector similarity score (0-1)
            bm25_weight: Weight for BM25 score (0-1)
            section_types: Filter by section types
            document_titles: Filter by document titles
            min_score_threshold: Minimum combined score threshold
            deduplicate: Remove near-duplicate results
        """
        top_k = top_k or settings.retrieval_top_k
        
        # Prepare BM25 if needed
        self._prepare_bm25()
        
        # Get query embedding
        query_vector = self.embedder.get_embedding(query)
        
        # Vector search (specify embedding column name)
        vector_results = self.table.search(query_vector, vector_column_name="embedding").limit(top_k * 3).to_pandas()  # Get more for reranking
        
        if len(vector_results) == 0:
            logger.warning("No vector search results found")
            return []
        
        # Apply filters
        if section_types:
            vector_results = vector_results[vector_results['section_type'].isin(section_types)]
        
        if document_titles:
            vector_results = vector_results[vector_results['document_title'].isin(document_titles)]
        
        if len(vector_results) == 0:
            logger.warning("No results after filtering")
            return []
        
        # Calculate BM25 scores
        bm25_scores = np.zeros(len(vector_results))
        if self._bm25 is not None:
            # Prepare documents for BM25 scoring
            docs_for_bm25 = []
            for _, row in vector_results.iterrows():
                doc_text = f"{row['content']} {row['section_name']} {row['section_type']}"
                docs_for_bm25.append(doc_text)
            
            bm25_scores = self._bm25.score(query, docs_for_bm25)
        
        # Normalize scores
        vector_scores = vector_results['_distance'].values
        vector_scores = 1 / (1 + vector_scores)  # Convert distance to similarity
        
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)  # Normalize to 0-1
        
        # Combine scores
        combined_scores = vector_weight * vector_scores + bm25_weight * bm25_scores
        
        # Create retrieval results
        results = []
        for i, (_, row) in enumerate(vector_results.iterrows()):
            if combined_scores[i] >= min_score_threshold:
                # Parse metadata
                metadata = {}
                try:
                    if row['metadata'] and row['metadata'].strip():
                        metadata = json.loads(row['metadata'])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse metadata for {row['id']}: {e}")
                
                result = RetrievalResult(
                    id=row['id'],
                    content=row['content'],
                    document_title=row['document_title'],
                    section_name=row['section_name'],
                    section_type=row['section_type'],
                    citation_id=row['citation_id'] or f"E{i+1}",
                    section_tag=row['section_tag'] or row['section_name'],
                    source_page=row['source_page'] or "",
                    file_path=row['file_path'] or "",
                    metadata=metadata,
                    vector_score=float(vector_scores[i]),
                    bm25_score=float(bm25_scores[i]),
                    combined_score=float(combined_scores[i]),
                    rank=i + 1
                )
                results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Deduplicate if requested
        if deduplicate:
            results = self._deduplicate_results(results)
        
        # Update ranks and limit to top_k
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        logger.info(f"Retrieved {len(results[:top_k])} results for query: {query[:50]}...")
        return results[:top_k]
    
    def _deduplicate_results(self, results: List[RetrievalResult], similarity_threshold: float = 0.8) -> List[RetrievalResult]:
        """Remove near-duplicate results based on content similarity."""
        if len(results) <= 1:
            return results
        
        # Use TF-IDF for quick similarity calculation
        vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
        contents = [result.content for result in results]
        
        try:
            tfidf_matrix = vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            to_remove = set()
            for i in range(len(results)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(results)):
                    if j in to_remove:
                        continue
                    if similarity_matrix[i, j] > similarity_threshold:
                        # Keep the one with higher score
                        if results[i].combined_score >= results[j].combined_score:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
            
            deduplicated = [result for i, result in enumerate(results) if i not in to_remove]
            logger.info(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
            return deduplicated
            
        except Exception as e:
            logger.warning(f"Deduplication failed: {e}")
            return results
    
    def build_evidence_pack(self,
                           query: str,
                           results: List[RetrievalResult],
                           filters_applied: Optional[Dict[str, Any]] = None) -> EvidencePack:
        """
        Build evidence pack with formatted citations and metadata.
        
        Format: [E1 · 实验分组与给药], [E2 · 数据记录表格 - 肿瘤体积], etc.
        """
        # Generate formatted evidence text
        evidence_sections = []
        citations = []
        
        for result in results:
            # Format citation with section tag
            citation_text = f"[{result.citation_id} · {result.section_tag}]"
            
            # Format content with citation
            content_with_citation = f"{citation_text}\n{result.content}"
            if result.source_page:
                content_with_citation += f"\n来源页面: {result.source_page}"
            
            evidence_sections.append(content_with_citation)
            
            # Add citation details
            citation_detail = {
                "citation_id": result.citation_id,
                "section_tag": result.section_tag,
                "document_title": result.document_title,
                "section_name": result.section_name,
                "section_type": result.section_type,
                "source_page": result.source_page,
                "file_path": result.file_path,
                "combined_score": result.combined_score,
                "rank": result.rank
            }
            citations.append(citation_detail)
        
        # Combine all evidence
        evidence_text = "\n\n---\n\n".join(evidence_sections)
        
        return EvidencePack(
            query=query,
            results=results,
            evidence_text=evidence_text,
            citations=citations,
            total_results=len(results),
            filters_applied=filters_applied or {}
        )
    
    def get_section_types(self) -> List[str]:
        """Get all available section types in the corpus."""
        try:
            df = self.table.to_pandas()
            return sorted(df['section_type'].unique().tolist())
        except Exception as e:
            logger.error(f"Failed to get section types: {e}")
            return []
    
    def get_document_titles(self) -> List[str]:
        """Get all available document titles in the corpus."""
        try:
            df = self.table.to_pandas()
            return sorted(df['document_title'].unique().tolist())
        except Exception as e:
            logger.error(f"Failed to get document titles: {e}")
            return []
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed corpus."""
        try:
            df = self.table.to_pandas()
            
            stats = {
                "total_chunks": len(df),
                "total_documents": df['document_title'].nunique(),
                "section_types": df['section_type'].value_counts().to_dict(),
                "documents": df['document_title'].unique().tolist(),
                "avg_content_length": df['content'].str.len().mean(),
                "section_type_distribution": df.groupby(['document_title', 'section_type']).size().to_dict()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get corpus stats: {e}")
            return {"error": str(e)}


# Convenience functions
async def search_documents(query: str,
                          top_k: int = None,
                          section_types: Optional[List[str]] = None,
                          document_titles: Optional[List[str]] = None,
                          vector_weight: float = 0.7,
                          bm25_weight: float = 0.3) -> EvidencePack:
    """Convenience function for document search with evidence pack building."""
    retriever = ToxiRAGRetriever()
    
    results = await retriever.search(
        query=query,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        section_types=section_types,
        document_titles=document_titles
    )
    
    filters_applied = {
        "section_types": section_types,
        "document_titles": document_titles,
        "vector_weight": vector_weight,
        "bm25_weight": bm25_weight
    }
    
    evidence_pack = retriever.build_evidence_pack(query, results, filters_applied)
    return evidence_pack


async def retrieve_relevant_docs(query: str,
                                top_k: int = 5,
                                collection_name: str = "toxicology_docs",
                                section_types: Optional[List[str]] = None,
                                document_titles: Optional[List[str]] = None,
                                vector_weight: float = 0.7,
                                bm25_weight: float = 0.3) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents for a query.
    Returns a list of documents as dictionaries for compatibility with legacy interface.
    """
    retriever = ToxiRAGRetriever(table_name=collection_name)
    
    # Use the search method to get RetrievalResult objects
    results = await retriever.search(
        query=query,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        section_types=section_types,
        document_titles=document_titles
    )
    
    # Convert RetrievalResult objects to dictionaries for compatibility
    docs = []
    for result in results:
        doc = {
            "content": result.content,
            "document_title": result.document_title,
            "section_type": result.section_type,
            "section_name": result.section_name,
            "source_page": result.source_page,
            "citation_id": result.citation_id,
            "section_tag": result.section_tag,
            "file_path": result.file_path,
            "vector_score": result.vector_score,
            "bm25_score": result.bm25_score,
            "combined_score": result.combined_score,
            "rank": result.rank,
            "metadata": result.metadata
        }
        docs.append(doc)
    
    return docs


def get_available_filters() -> Dict[str, List[str]]:
    """Get available filter options."""
    retriever = ToxiRAGRetriever()
    return {
        "section_types": retriever.get_section_types(),
        "document_titles": retriever.get_document_titles()
    }
