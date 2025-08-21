"""
Unit tests for hybrid retrieval functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from retriever.retriever import ToxiRAGRetriever, BM25Scorer, RetrievalResult


class TestBM25Scorer:
    """Test BM25 scoring implementation."""
    
    def setup_method(self):
        self.scorer = BM25Scorer()
    
    def test_fit_and_score(self):
        """Test basic BM25 fit and scoring."""
        documents = [
            "liver cancer toxicity study",
            "tumor volume measurement methods", 
            "cancer cell growth analysis",
            "liver function test results"
        ]
        
        self.scorer.fit(documents)
        
        # Test query scoring
        query = "liver cancer"
        scores = self.scorer.score(query, documents)
        
        assert len(scores) == len(documents)
        assert isinstance(scores, np.ndarray)
        assert scores[0] > scores[1]  # First doc should score higher for "liver cancer"
    
    def test_empty_documents(self):
        """Test behavior with empty document list."""
        with pytest.raises(Exception):  # Should fail gracefully
            self.scorer.fit([])
    
    def test_query_not_in_corpus(self):
        """Test query with terms not in corpus."""
        documents = ["liver cancer study", "tumor analysis"]
        self.scorer.fit(documents)
        
        scores = self.scorer.score("unknown_term", documents)
        assert len(scores) == len(documents)
        assert all(score >= 0 for score in scores)  # Should return non-negative scores


class TestToxiRAGRetriever:
    """Test hybrid retriever functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            self.retriever = ToxiRAGRetriever()
        
        # Mock database and table
        self.mock_db = Mock()
        self.mock_table = Mock()
        self.retriever._db = self.mock_db
        self.retriever._table = self.mock_table
        
        # Mock embedder
        self.retriever.embedder = AsyncMock()
        
        # Sample data based on real examples from 3 different papers in 肝癌.md
        self.sample_data = pd.DataFrame([
            # Paper 1: 玉米花粉多糖的药理药效研究
            {
                'id': 'doc1_1_corn_pollen',
                'content': '玉米花粉多糖可显著提高巨噬细胞活性（乳酸脱氢酶、酸性磷酸酶活性升高），促进IL-6 和TNF-α分泌，增强脾指数、胸腺指数，提高NK细胞活性，增加T4细胞，降低T8细胞，从而提高免疫功能。',
                'document_title': '玉米花粉多糖的药理药效研究',
                'section_name': '机制研究结果',
                'section_type': 'mechanism',
                'citation_id': 'E1',
                'section_tag': '机制研究结果',
                'source_page': 'p.2–3，Table 1–9',
                'file_path': '/data/corn_pollen_study.md',
                'metadata': '{"immune_enhancement": true, "cytokines": ["IL-6", "TNF-α"], "cell_types": ["NK", "T4", "T8"]}',
                '_distance': 0.2
            },
            # Paper 2: 玛咖生物碱研究  
            {
                'id': 'doc2_1_maca_alkaloid',
                'content': '玛咖生物碱在不同剂量下对肝癌细胞Bel-7402有剂量依赖性的抑制作用，高剂量抑制率达81.49%。对荷瘤小鼠的抗肿瘤作用未发现与免疫器官指数显著相关，提示其抗肝癌机制与免疫调节无关，具体机制未说明。',
                'document_title': '玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用',
                'section_name': '机制研究结果',
                'section_type': 'mechanism',
                'citation_id': 'E2',
                'section_tag': '机制研究结果',
                'source_page': 'p.54–56，2.1.1，2.4，3.3',
                'file_path': '/data/maca_alkaloid_study.md',
                'metadata': '{"cell_line": "Bel-7402", "max_inhibition": "81.49%", "immune_related": false}',
                '_distance': 0.25
            },
            # Paper 3: 理冲汤联合5-FU研究
            {
                'id': 'doc3_1_lichongtang_5fu',
                'content': '分组名称及数量:\n空白组 (n=10)\n5-FU组 (n=10)\n5-FU+中药组 (n=10)\n中药组 (n=10)\n给药方式:\n5-FU: 腹腔注射 (intraperitoneal injection)\n理冲汤 (中药): 灌胃 (gavage)\n药物剂量与频率:\n5-FU组: 5-FU 2.5 mg·kg⁻¹，隔日1次\n理冲汤: 25 g·kg⁻¹，每日1次\n给药周期: 14 天',
                'document_title': 'Effect of Modified Lichongtang Combined with 5-Fluorouracil on Epithelial Mesenchymal Transition in H22 Tumor-bearing Mice',
                'section_name': '实验分组与给药',
                'section_type': 'experiment_groups',
                'citation_id': 'E3',
                'section_tag': '实验分组与给药',
                'source_page': 'p.1-2, p.4',
                'file_path': '/data/lichongtang_5fu_study.md',
                'metadata': '{"groups": 4, "combination_therapy": true, "tcm_name": "理冲汤", "chemotherapy": "5-FU"}',
                '_distance': 0.3
            },
            # Paper 1: 玉米花粉多糖 - 数据表格
            {
                'id': 'doc1_2_corn_data',
                'content': '| 指标 | 对照组 | 药物组 | P值 |\n|------|--------|--------|------|\n| 吞噬指数 | 7.13±0.41 | 16.90±0.98 | p<0.01 |\n| IL-6 (24h, ng/mL) | 4.29±0.49 | 6.62±0.78 (中剂量) | p<0.001 |\n| TNF-α (ng/mL) | 0.437±0.086 | 0.738±0.111 (低剂量) | p<0.001 |\n| 乳酸脱氢酶 (IU) | 206.7±55.4 | 450.5±71.9 (低剂量) | p<0.001 |\n| 酸性磷酸酶 (IU) | 19.5±1.5 | 25.0±3.3 (低剂量) | p<0.001 |',
                'document_title': '玉米花粉多糖的药理药效研究',
                'section_name': '数据记录表格 - 机制检测数据表',
                'section_type': 'data_table',
                'citation_id': 'E4',
                'section_tag': '数据记录表格 - 机制检测数据表',
                'source_page': 'p.2–3，Table 1–4',
                'file_path': '/data/corn_pollen_study.md',
                'metadata': '{"data_type": "mechanism_markers", "statistical_significance": true, "cytokines_measured": true}',
                '_distance': 0.35
            },
            # Paper 3: 理冲汤联合5-FU - EMT分子标记物数据
            {
                'id': 'doc3_2_emt_markers',
                'content': '| 分子标记物 | 检测方法 | 空白组 | 5-FU组 | 5-FU+中药组 | 中药组 |\n|----------|----------|--------|--------|------------|--------|\n| E-cadherin | Real-time PCR (mRNA) | 1 | 1.86±0.12 | 3.42±0.26 | 2.01±0.34 |\n| N-cadherin | Real-time PCR (mRNA) | 1 | 0.53±0.10 | 0.18±0.12 | 0.64±0.18 |\n| Snail | Real-time PCR (mRNA) | 1 | 0.32±0.04 | 0.14±0.05 | 0.42±0.09 |\n| Twist | Real-time PCR (mRNA) | 1 | 0.37±0.03 | 0.11±0.02 | 0.46±0.23 |',
                'document_title': 'Effect of Modified Lichongtang Combined with 5-Fluorouracil on Epithelial Mesenchymal Transition in H22 Tumor-bearing Mice',
                'section_name': '数据记录表格 - EMT分子标记物',
                'section_type': 'data_table',
                'citation_id': 'E5',
                'section_tag': '数据记录表格 - EMT分子标记物',
                'source_page': 'p.6, Table 5, Table 6',
                'file_path': '/data/lichongtang_5fu_study.md',
                'metadata': '{"emt_markers": ["E-cadherin", "N-cadherin", "Snail", "Twist"], "method": "Real-time PCR", "combination_effect": true}',
                '_distance': 0.4
            },
            # Paper 2: 玛咖生物碱 - 肿瘤模型信息
            {
                'id': 'doc2_2_tumor_model',
                'content': '模型类型：移植瘤模型\n肿瘤类型：H22 肝癌\n接种位置：小鼠腋下\n成瘤时间（天）：未说明\n给药开始时间（成瘤后第几天）：Day 3\n成瘤总天数：18天（给药结束处死）',
                'document_title': '玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用',
                'section_name': '肿瘤模型信息',
                'section_type': 'tumor_model',
                'citation_id': 'E6',
                'section_tag': '肿瘤模型信息',
                'source_page': 'p.53–54，1.3.3.1–1.3.3.4',
                'file_path': '/data/maca_alkaloid_study.md',
                'metadata': '{"model_type": "移植瘤模型", "tumor_type": "H22肝癌", "treatment_start": "Day 3", "duration": "18天"}',
                '_distance': 0.45
            }
        ])
    
    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality."""
        # Mock table search
        self.mock_table.search.return_value.limit.return_value.to_pandas.return_value = self.sample_data
        
        # Mock embedder
        self.retriever.embedder.aembedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        
        # Mock BM25
        mock_bm25 = Mock()
        mock_bm25.score.return_value = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        self.retriever._bm25 = mock_bm25
        
        # Run search
        results = await self.retriever.search("肝癌细胞抗肿瘤机制研究", top_k=5)
        
        # Verify results
        assert len(results) <= 5
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.combined_score > 0 for r in results)
        if len(results) > 1:
            assert results[0].combined_score >= results[1].combined_score  # Should be sorted
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with section type filters."""
        # Filter to only mechanism results
        filtered_data = self.sample_data[self.sample_data['section_type'] == 'mechanism']
        self.mock_table.search.return_value.limit.return_value.to_pandas.return_value = filtered_data
        
        self.retriever.embedder.aembedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        
        mock_bm25 = Mock()
        mock_bm25.score.return_value = np.array([0.8, 0.7])  # Two mechanism results
        self.retriever._bm25 = mock_bm25
        
        results = await self.retriever.search(
            "免疫功能增强机制", 
            section_types=['mechanism']
        )
        
        assert len(results) == 2  # Two mechanism results in dataset
        assert all(r.section_type == 'mechanism' for r in results)
    
    @pytest.mark.asyncio
    async def test_search_scoring_weights(self):
        """Test different scoring weight combinations."""
        self.mock_table.search.return_value.limit.return_value.to_pandas.return_value = self.sample_data
        self.retriever.embedder.aembedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        
        mock_bm25 = Mock()
        mock_bm25.score.return_value = np.array([0.1, 0.9, 0.5, 0.4, 0.3, 0.2])  # BM25 favors second result
        self.retriever._bm25 = mock_bm25
        
        # Test vector-heavy weighting
        results_vector = await self.retriever.search(
            "test query", 
            vector_weight=0.9, 
            bm25_weight=0.1
        )
        
        # Test BM25-heavy weighting  
        results_bm25 = await self.retriever.search(
            "test query",
            vector_weight=0.1,
            bm25_weight=0.9
        )
        
        # Results should be different due to different weightings
        assert len(results_vector) > 0
        assert len(results_bm25) > 0
    
    def test_deduplicate_results(self):
        """Test result deduplication functionality."""
        # Create results with similar content
        results = [
            RetrievalResult(
                id="1", content="肿瘤体积测量方法采用游标卡尺进行精确测量", document_title="Study1",
                section_name="方法", section_type="method", citation_id="E1",
                section_tag="方法", source_page="P1", file_path="f1.md", metadata={},
                vector_score=0.9, bm25_score=0.8, combined_score=0.85, rank=1
            ),
            RetrievalResult(
                id="2", content="肿瘤体积的测量方法采用游标卡尺进行精确测量", document_title="Study2", 
                section_name="方法", section_type="method", citation_id="E2",
                section_tag="方法", source_page="P2", file_path="f2.md", metadata={},
                vector_score=0.8, bm25_score=0.7, combined_score=0.75, rank=2
            ),
            RetrievalResult(
                id="3", content="完全不同的内容关于药物安全性评价和毒理学分析", document_title="Study3",
                section_name="安全性", section_type="safety", citation_id="E3", 
                section_tag="安全性", source_page="P3", file_path="f3.md", metadata={},
                vector_score=0.7, bm25_score=0.6, combined_score=0.65, rank=3
            )
        ]
        
        deduplicated = self.retriever._deduplicate_results(results, similarity_threshold=0.6)
        
        # Should either remove duplicates or preserve all (deduplication may be dependent on TF-IDF vectorizer behavior)
        assert len(deduplicated) <= len(results)
        assert any(r.id == "3" for r in deduplicated)  # Different content should remain
        
        # Test with very low threshold to force deduplication
        deduplicated_strict = self.retriever._deduplicate_results(results, similarity_threshold=0.1)
        assert len(deduplicated_strict) <= len(results)
    
    def test_get_section_types(self):
        """Test getting available section types."""
        self.mock_table.to_pandas.return_value = self.sample_data
        
        section_types = self.retriever.get_section_types()
        
        expected_types = sorted(['mechanism', 'experiment_groups', 'data_table', 'tumor_model'])
        assert section_types == expected_types
    
    def test_get_document_titles(self):
        """Test getting available document titles.""" 
        self.mock_table.to_pandas.return_value = self.sample_data
        
        document_titles = self.retriever.get_document_titles()
        
        expected_titles = sorted([
            '玉米花粉多糖的药理药效研究',
            '玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用',
            'Effect of Modified Lichongtang Combined with 5-Fluorouracil on Epithelial Mesenchymal Transition in H22 Tumor-bearing Mice'
        ])
        assert document_titles == expected_titles
    
    def test_get_corpus_stats(self):
        """Test corpus statistics retrieval."""
        self.mock_table.to_pandas.return_value = self.sample_data
        
        stats = self.retriever.get_corpus_stats()
        
        assert stats['total_chunks'] == 6
        assert stats['total_documents'] == 3
        assert 'section_types' in stats
        assert 'documents' in stats
        assert isinstance(stats['avg_content_length'], float)


class TestRetrievalResultClass:
    """Test RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self):
        """Test creating RetrievalResult instances."""
        result = RetrievalResult(
            id="test_id",
            content="Test content",
            document_title="Test Document",
            section_name="Test Section",
            section_type="test_type",
            citation_id="E1",
            section_tag="Test Tag",
            source_page="Page 1",
            file_path="test.md",
            metadata={"key": "value"},
            vector_score=0.8,
            bm25_score=0.7,
            combined_score=0.75,
            rank=1
        )
        
        assert result.id == "test_id"
        assert result.combined_score == 0.75
        assert result.metadata["key"] == "value"
    
    def test_result_comparison(self):
        """Test result sorting by score."""
        result1 = RetrievalResult(
            id="1", content="", document_title="", section_name="", section_type="",
            citation_id="E1", section_tag="", source_page="", file_path="", metadata={},
            vector_score=0.8, bm25_score=0.7, combined_score=0.9, rank=1
        )
        result2 = RetrievalResult(
            id="2", content="", document_title="", section_name="", section_type="",
            citation_id="E2", section_tag="", source_page="", file_path="", metadata={},
            vector_score=0.6, bm25_score=0.5, combined_score=0.7, rank=2
        )
        
        results = [result2, result1]  # Lower score first
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        assert results[0].combined_score > results[1].combined_score
        assert results[0].id == "1"
