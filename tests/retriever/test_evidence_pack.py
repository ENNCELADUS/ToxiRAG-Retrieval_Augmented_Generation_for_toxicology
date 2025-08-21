"""
Unit tests for evidence pack building and citation formatting.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from retriever.retriever import ToxiRAGRetriever, RetrievalResult, EvidencePack, search_documents


class TestEvidencePack:
    """Test evidence pack functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create sample retrieval results based on real examples from 3 different papers in 肝癌.md
        self.sample_results = [
            # Paper 1: 玉米花粉多糖的药理药效研究
            RetrievalResult(
                id="doc1_1_corn_pollen",
                content="玉米花粉多糖可显著提高巨噬细胞活性（乳酸脱氢酶、酸性磷酸酶活性升高），促进IL-6 和TNF-α分泌，增强脾指数、胸腺指数，提高NK细胞活性，增加T4细胞，降低T8细胞，从而提高免疫功能。",
                document_title="玉米花粉多糖的药理药效研究",
                section_name="机制研究结果",
                section_type="mechanism", 
                citation_id="E1",
                section_tag="机制研究结果",
                source_page="p.2–3，Table 1–9",
                file_path="/data/corn_pollen_study.md",
                metadata={"immune_enhancement": True, "cytokines": ["IL-6", "TNF-α"], "cell_types": ["NK", "T4", "T8"]},
                vector_score=0.85,
                bm25_score=0.78,
                combined_score=0.82,
                rank=1
            ),
            # Paper 2: 玛咖生物碱研究
            RetrievalResult(
                id="doc2_1_maca_alkaloid", 
                content="玛咖生物碱在不同剂量下对肝癌细胞Bel-7402有剂量依赖性的抑制作用，高剂量抑制率达81.49%。对荷瘤小鼠的抗肿瘤作用未发现与免疫器官指数显著相关，提示其抗肝癌机制与免疫调节无关，具体机制未说明。",
                document_title="玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用",
                section_name="机制研究结果",
                section_type="mechanism",
                citation_id="E2", 
                section_tag="机制研究结果",
                source_page="p.54–56，2.1.1，2.4，3.3",
                file_path="/data/maca_alkaloid_study.md",
                metadata={"cell_line": "Bel-7402", "max_inhibition": "81.49%", "immune_related": False},
                vector_score=0.79,
                bm25_score=0.73,
                combined_score=0.76,
                rank=2
            ),
            # Paper 3: 理冲汤联合5-FU研究
            RetrievalResult(
                id="doc3_1_lichongtang_5fu",
                content="| 分子标记物 | 检测方法 | 空白组 | 5-FU组 | 5-FU+中药组 | 中药组 |\n|----------|----------|--------|--------|------------|--------|\n| E-cadherin | Real-time PCR (mRNA) | 1 | 1.86±0.12 | 3.42±0.26 | 2.01±0.34 |\n| N-cadherin | Real-time PCR (mRNA) | 1 | 0.53±0.10 | 0.18±0.12 | 0.64±0.18 |\n| Snail | Real-time PCR (mRNA) | 1 | 0.32±0.04 | 0.14±0.05 | 0.42±0.09 |\n| Twist | Real-time PCR (mRNA) | 1 | 0.37±0.03 | 0.11±0.02 | 0.46±0.23 |",
                document_title="Effect of Modified Lichongtang Combined with 5-Fluorouracil on Epithelial Mesenchymal Transition in H22 Tumor-bearing Mice",
                section_name="数据记录表格 - EMT分子标记物",
                section_type="data_table",
                citation_id="E3",
                section_tag="数据记录表格 - EMT分子标记物", 
                source_page="p.6, Table 5, Table 6",
                file_path="/data/lichongtang_5fu_study.md",
                metadata={"emt_markers": ["E-cadherin", "N-cadherin", "Snail", "Twist"], "method": "Real-time PCR", "combination_effect": True},
                vector_score=0.72,
                bm25_score=0.68,
                combined_score=0.70,
                rank=3
            ),
            # Additional result from Paper 1 for more comprehensive testing
            RetrievalResult(
                id="doc1_2_corn_data",
                content="| 指标 | 对照组 | 药物组 | P值 |\n|------|--------|--------|------|\n| 吞噬指数 | 7.13±0.41 | 16.90±0.98 | p<0.01 |\n| IL-6 (24h, ng/mL) | 4.29±0.49 | 6.62±0.78 (中剂量) | p<0.001 |\n| TNF-α (ng/mL) | 0.437±0.086 | 0.738±0.111 (低剂量) | p<0.001 |\n| 乳酸脱氢酶 (IU) | 206.7±55.4 | 450.5±71.9 (低剂量) | p<0.001 |",
                document_title="玉米花粉多糖的药理药效研究",
                section_name="数据记录表格 - 机制检测数据表",
                section_type="data_table",
                citation_id="E4",
                section_tag="数据记录表格 - 机制检测数据表",
                source_page="p.2–3，Table 1–4",
                file_path="/data/corn_pollen_study.md",
                metadata={"data_type": "mechanism_markers", "statistical_significance": True, "cytokines_measured": True},
                vector_score=0.68,
                bm25_score=0.65,
                combined_score=0.67,
                rank=4
            )
        ]
    
    def test_build_evidence_pack_basic(self):
        """Test basic evidence pack building."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        query = "肝癌细胞抗肿瘤机制研究对比分析"
        evidence_pack = retriever.build_evidence_pack(query, self.sample_results)
        
        # Check basic structure
        assert isinstance(evidence_pack, EvidencePack)
        assert evidence_pack.query == query
        assert len(evidence_pack.results) == 4
        assert evidence_pack.total_results == 4
        assert len(evidence_pack.citations) == 4
    
    def test_citation_format(self):
        """Test proper citation formatting with bracketed numerals and section tags."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        evidence_pack = retriever.build_evidence_pack("test query", self.sample_results)
        
        # Check citation format in evidence text
        evidence_text = evidence_pack.evidence_text
        
        # Should contain proper citation format [E1 · section_tag] from 3 different papers
        assert "[E1 · 机制研究结果]" in evidence_text
        assert "[E2 · 机制研究结果]" in evidence_text
        assert "[E3 · 数据记录表格 - EMT分子标记物]" in evidence_text
        assert "[E4 · 数据记录表格 - 机制检测数据表]" in evidence_text
        
        # Should contain source pages from different papers
        assert "来源页面: p.2–3，Table 1–9" in evidence_text
        assert "来源页面: p.54–56，2.1.1，2.4，3.3" in evidence_text  
        assert "来源页面: p.6, Table 5, Table 6" in evidence_text
        assert "来源页面: p.2–3，Table 1–4" in evidence_text
        
        # Should contain content from different papers
        assert "玉米花粉多糖可显著提高巨噬细胞活性" in evidence_text
        assert "玛咖生物碱在不同剂量下对肝癌细胞Bel-7402有剂量依赖性" in evidence_text
        assert "E-cadherin" in evidence_text  # EMT marker
        assert "吞噬指数" in evidence_text  # Immune function marker
    
    def test_citation_details(self):
        """Test citation detail information."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        evidence_pack = retriever.build_evidence_pack("test query", self.sample_results)
        
        # Check citation details from multiple papers
        citations = evidence_pack.citations
        assert len(citations) == 4
        
        # Check first citation (Paper 1: 玉米花粉多糖)
        cite1 = citations[0]
        assert cite1["citation_id"] == "E1"
        assert cite1["section_tag"] == "机制研究结果"
        assert cite1["document_title"] == "玉米花粉多糖的药理药效研究"
        assert cite1["section_type"] == "mechanism"
        assert cite1["combined_score"] == 0.82
        assert cite1["rank"] == 1
        
        # Check second citation (Paper 2: 玛咖生物碱)
        cite2 = citations[1]
        assert cite2["citation_id"] == "E2"
        assert cite2["section_tag"] == "机制研究结果"
        assert cite2["document_title"] == "玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用"
        assert cite2["file_path"] == "/data/maca_alkaloid_study.md"
        
        # Check third citation (Paper 3: 理冲汤+5-FU)
        cite3 = citations[2]
        assert cite3["citation_id"] == "E3"
        assert cite3["section_tag"] == "数据记录表格 - EMT分子标记物"
        assert cite3["document_title"] == "Effect of Modified Lichongtang Combined with 5-Fluorouracil on Epithelial Mesenchymal Transition in H22 Tumor-bearing Mice"
        assert cite3["section_type"] == "data_table"
    
    def test_evidence_pack_with_filters(self):
        """Test evidence pack with applied filters."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        filters_applied = {
            "section_types": ["tumor_model", "experiment_groups"],
            "document_titles": None,
            "vector_weight": 0.7,
            "bm25_weight": 0.3
        }
        
        evidence_pack = retriever.build_evidence_pack(
            "test query", 
            self.sample_results[:3],  # Only first three results from different papers
            filters_applied
        )
        
        assert evidence_pack.total_results == 3
        assert evidence_pack.filters_applied == filters_applied
        assert len(evidence_pack.citations) == 3
    
    def test_evidence_text_formatting(self):
        """Test evidence text proper formatting and separation."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        evidence_pack = retriever.build_evidence_pack("test query", self.sample_results)
        
        evidence_text = evidence_pack.evidence_text
        
        # Should have proper separators between evidence sections
        sections = evidence_text.split("\n\n---\n\n")
        assert len(sections) == 4
        
        # Each section should start with citation
        for i, section in enumerate(sections):
            expected_citation = f"[E{i+1} ·"
            assert section.startswith(expected_citation)
    
    def test_empty_results(self):
        """Test evidence pack with no results."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        evidence_pack = retriever.build_evidence_pack("test query", [])
        
        assert evidence_pack.total_results == 0
        assert len(evidence_pack.results) == 0
        assert len(evidence_pack.citations) == 0
        assert evidence_pack.evidence_text == ""
    
    def test_single_result(self):
        """Test evidence pack with single result."""
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        single_result = [self.sample_results[0]]
        evidence_pack = retriever.build_evidence_pack("test query", single_result)
        
        assert evidence_pack.total_results == 1
        assert len(evidence_pack.citations) == 1
        assert "[E1 · 机制研究结果]" in evidence_pack.evidence_text
        assert "---" not in evidence_pack.evidence_text  # No separators for single result


class TestSearchDocuments:
    """Test convenience search function."""
    
    @pytest.mark.asyncio
    @patch('retriever.retriever.ToxiRAGRetriever')
    async def test_search_documents_convenience(self, mock_retriever_class):
        """Test search_documents convenience function."""
        # Mock retriever instance
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        
        # Mock search results
        mock_results = [
            RetrievalResult(
                id="test", content="test content", document_title="test doc",
                section_name="test section", section_type="test_type", 
                citation_id="E1", section_tag="test tag", source_page="P1",
                file_path="test.md", metadata={}, vector_score=0.8,
                bm25_score=0.7, combined_score=0.75, rank=1
            )
        ]
        mock_retriever.search = AsyncMock(return_value=mock_results)
        
        # Mock evidence pack
        mock_evidence_pack = EvidencePack(
            query="test query",
            results=mock_results,
            evidence_text="[E1 · test tag]\ntest content",
            citations=[{"citation_id": "E1"}],
            total_results=1,
            filters_applied={}
        )
        mock_retriever.build_evidence_pack.return_value = mock_evidence_pack
        
        # Test function call
        result = await search_documents(
            query="test query",
            top_k=5,
            section_types=["test_type"],
            vector_weight=0.8,
            bm25_weight=0.2
        )
        
        # Verify calls
        mock_retriever.search.assert_called_once_with(
            query="test query",
            top_k=5,
            vector_weight=0.8,
            bm25_weight=0.2,
            section_types=["test_type"],
            document_titles=None
        )
        
        mock_retriever.build_evidence_pack.assert_called_once()
        
        # Verify result
        assert isinstance(result, EvidencePack)
        assert result.query == "test query"
        assert result.total_results == 1


class TestEvidencePackEdgeCases:
    """Test edge cases in evidence pack building."""
    
    def test_missing_citation_fields(self):
        """Test handling of missing citation fields."""
        result_with_missing_fields = RetrievalResult(
            id="test_id",
            content="Test content",
            document_title="Test Document", 
            section_name="Test Section",
            section_type="test_type",
            citation_id="",  # Missing citation ID
            section_tag="",  # Missing section tag
            source_page="",  # Missing source page
            file_path="test.md",
            metadata={},
            vector_score=0.8,
            bm25_score=0.7,
            combined_score=0.75,
            rank=1
        )
        
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        evidence_pack = retriever.build_evidence_pack("test", [result_with_missing_fields])
        
        # Should handle missing fields gracefully
        assert evidence_pack.total_results == 1
        assert len(evidence_pack.citations) == 1
        
        # Citation should still be created
        citation = evidence_pack.citations[0]
        assert citation["citation_id"] == ""
        assert citation["section_tag"] == ""
        assert citation["source_page"] == ""
    
    def test_special_characters_in_content(self):
        """Test handling special characters in content and citations."""
        result_with_special_chars = RetrievalResult(
            id="test_id",
            content="测试内容包含特殊字符：【】、（）、'引号'、<标签>",
            document_title="特殊字符测试文档",
            section_name="特殊字符 & 符号",
            section_type="test_type",
            citation_id="E1",
            section_tag="特殊字符 & 符号",
            source_page="第10页",
            file_path="special_chars.md", 
            metadata={"special": "值"},
            vector_score=0.8,
            bm25_score=0.7,
            combined_score=0.75,
            rank=1
        )
        
        with patch('retriever.retriever.OpenAIEmbedder'):
            retriever = ToxiRAGRetriever()
        
        evidence_pack = retriever.build_evidence_pack("特殊查询", [result_with_special_chars])
        
        # Should handle special characters properly
        assert "【】、（）、'引号'、<标签>" in evidence_pack.evidence_text
        assert "[E1 · 特殊字符 & 符号]" in evidence_pack.evidence_text
        assert "来源页面: 第10页" in evidence_pack.evidence_text
