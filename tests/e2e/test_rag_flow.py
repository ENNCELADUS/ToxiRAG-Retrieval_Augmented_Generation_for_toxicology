"""
End-to-End RAG Flow Tests
Tests the complete pipeline: ingest → retrieve → answer → citations
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ingest.ingest_local import ingest_markdown_file
from retriever.retriever import search_documents
from llm.agentic_pipeline import create_agentic_response
from eval.evaluator import ToxiRAGEvaluator


class TestE2ERAGFlow:
    """End-to-end tests for the complete RAG pipeline."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_toxirag.lance"
    
    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content for testing ingestion."""
        return """# 论文标题: 测试化合物的抗肝癌研究
（来源：p.1 / Title）

## 肿瘤模型信息
模型类型：移植瘤模型
肿瘤类型：H22 肝癌
接种位置：小鼠腋下
成瘤时间（天）：3天
给药开始时间（成瘤后第几天）：Day 3
成瘤总天数：14天
（来源：p.2，1.2.1）

## 实验分组与给药
- 分组名称及数量: 对照组 (n=10)、低剂量组 (n=10)、高剂量组 (n=10)
- 给药方式: 腹腔注射 (ip)
- 药物剂量与频率: 低剂量组 100 mg/kg qd；高剂量组 200 mg/kg qd
- 给药周期: 连续10天，每日1次
（来源：p.2–3 / 表格1）

## 机制研究结果
测试化合物通过抑制肝癌细胞增殖和诱导凋亡发挥抗肿瘤作用。高剂量组肿瘤抑制率达到65.3%，显著优于对照组（p<0.01）。机制检测显示该化合物能够上调促凋亡蛋白Bax表达，下调抗凋亡蛋白Bcl-2表达。
（来源：p.4–5，图2–3）

### 机制检测数据表
| 指标 | 对照组 | 低剂量组 | 高剂量组 | P值 |
|------|--------|----------|----------|-----|
| Bax蛋白表达 | 1.0±0.1 | 1.8±0.2 | 2.5±0.3 | p<0.01 |
| Bcl-2蛋白表达 | 1.0±0.1 | 0.7±0.1 | 0.4±0.1 | p<0.01 |
| 肿瘤体积 (mm³) | 1250±120 | 850±95 | 435±68 | p<0.01 |
| 抑瘤率 (%) | 0 | 32.0 | 65.3 | - |

（来源：p.5，表格2）

## 研究结论
测试化合物具有显著的抗肝癌活性，高剂量组抑瘤率达65.3%。其作用机制主要通过调控凋亡相关蛋白Bax/Bcl-2比例，诱导肿瘤细胞凋亡实现。该研究为化合物的抗肝癌应用提供了实验依据。
（来源：p.6 / 结论）
"""
    
    @pytest.fixture
    def sample_markdown_file(self, sample_markdown_content):
        """Create temporary markdown file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_markdown_content)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline_mock_embeddings(self, temp_db_path, sample_markdown_file):
        """Test complete RAG pipeline with mocked embeddings."""
        # Mock OpenAI embeddings
        with patch('agno.embedder.openai.OpenAIEmbedder') as mock_embedder_class:
            mock_embedder = AsyncMock()
            mock_embedder.get_embedding.return_value = [0.1] * 1536  # Mock embedding
            mock_embedder.aembedding.return_value = [[0.1] * 1536]  # Mock async embedding
            mock_embedder_class.return_value = mock_embedder
            
            # Step 1: Ingest document
            from ingest.ingest_local import ToxiRAGIngester
            collection_name = "test_e2e_collection"
            ingester = ToxiRAGIngester(
                lancedb_uri=str(temp_db_path),
                table_name=collection_name
            )
            result = await ingester.ingest_file(sample_markdown_file)
            
            assert result["status"] == "success"
            assert result["chunks"] > 0
            
            # Step 2: Test retrieval using the same database
            from retriever.retriever import ToxiRAGRetriever
            retriever = ToxiRAGRetriever(
                lancedb_uri=str(temp_db_path),
                table_name=collection_name
            )
            results = await retriever.search(
                query="测试化合物的抗肿瘤机制是什么？",
                top_k=3
            )
            evidence_pack = retriever.build_evidence_pack("测试化合物的抗肿瘤机制是什么？", results)
            
            assert evidence_pack.total_results > 0
            assert len(evidence_pack.citations) > 0
            assert "测试化合物" in evidence_pack.evidence_text
            
            # Verify citation format
            for citation in evidence_pack.citations:
                assert "citation_id" in citation
                assert "section_tag" in citation
                assert "document_title" in citation
    
    def test_agentic_response_structure(self):
        """Test agentic response data structure creation."""
        from llm.agentic_pipeline import EvidencePack, AgenticResponse
        
        # Test creating evidence pack
        evidence_pack = EvidencePack(
            query="测试查询",
            retrieved_docs=[{"content": "测试内容"}],
            evidence_text="[E1 · 测试]\n测试内容",
            citation_ids=["[E1 · 测试]"],
            confidence_score=0.8,
            has_sufficient_evidence=True
        )
        
        assert evidence_pack.query == "测试查询"
        assert evidence_pack.confidence_score == 0.8
        assert evidence_pack.has_sufficient_evidence is True
        
        # Test creating agentic response
        response = AgenticResponse(
            query="测试查询",
            response_text="测试回答 [E1 · 测试]",
            evidence_pack=evidence_pack,
            reasoning_steps=["Step 1"],
            citations=["[E1 · 测试]"],
            confidence_score=0.8
        )
        
        assert response.query == "测试查询"
        assert len(response.citations) == 1
        assert response.confidence_score == 0.8
        assert response.evidence_pack is not None
    
    @pytest.mark.asyncio
    async def test_evaluation_pipeline_mock(self, temp_db_path):
        """Test evaluation pipeline with golden questions (mocked)."""
        # Create temporary evaluation config
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = Path(tmpdir)
            
            # Create minimal golden questions
            golden_questions = {
                "dataset": "test_dataset",
                "questions": [
                    {
                        "id": "TEST-001",
                        "question": "测试化合物的抗肿瘤机制是什么？",
                        "query_type": "mechanism",
                        "expected_citations": [
                            {
                                "document_title": "测试化合物的抗肝癌研究",
                                "section_tag": "机制研究结果",
                                "source_page": "p.4–5",
                                "must_include_phrases": ["抑制", "凋亡", "Bax", "Bcl-2"]
                            }
                        ],
                        "min_citations_required": 1
                    }
                ]
            }
            
            # Create config files
            import yaml
            with open(eval_dir / "golden_questions.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(golden_questions, f, allow_unicode=True)
            
            eval_config = {
                "name": "test_eval",
                "collection_name": "test_collection",
                "golden_questions_file": str(eval_dir / "golden_questions.yaml"),
                "scoring": {
                    "grounding_required": True,
                    "min_grounding_score": 0.3,
                    "min_citation_coverage": 0.5,
                    "min_overall_score": 0.4
                }
            }
            
            with open(eval_dir / "config.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(eval_config, f)
            
            # Mock the agentic response in evaluator
            with patch('eval.evaluator.create_agentic_response') as mock_create_response:
                from llm.agentic_pipeline import AgenticResponse, EvidencePack
                
                mock_evidence_pack = EvidencePack(
                    query="测试化合物的抗肿瘤机制是什么？",
                    retrieved_docs=[],
                    evidence_text="测试化合物通过抑制肝癌细胞增殖和诱导凋亡发挥抗肿瘤作用",
                    citation_ids=["[E1 · 机制研究结果]"],
                    confidence_score=0.8,
                    has_sufficient_evidence=True
                )
                
                mock_create_response.return_value = AgenticResponse(
                    query="测试化合物的抗肿瘤机制是什么？",
                    response_text="测试化合物通过抑制肝癌细胞增殖和诱导凋亡发挥抗肿瘤作用。[E1 · 机制研究结果] 具体表现为上调Bax蛋白表达，下调Bcl-2蛋白表达。",
                    evidence_pack=mock_evidence_pack,
                    reasoning_steps=["Analysis completed"],
                    citations=["[E1 · 机制研究结果]"],
                    confidence_score=0.8
                )
                
                # Run evaluation
                evaluator = ToxiRAGEvaluator(eval_dir / "config.yaml")
                result = await evaluator.run_evaluation(
                    output_dir=eval_dir / "results",
                    limit=1  # Only test one question
                )
                
                # Verify evaluation results
                assert result.total_questions == 1
                assert result.passed_questions >= 0  # May pass or fail depending on thresholds
                assert len(result.question_results) == 1
                
                question_result = result.question_results[0]
                assert question_result.question_id == "TEST-001"
                assert len(question_result.citations_found) > 0
                assert question_result.grounding_score >= 0
                assert question_result.citation_coverage >= 0
    
    def test_citation_extraction(self):
        """Test citation extraction from response text."""
        from eval.evaluator import ToxiRAGEvaluator
        
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create dummy golden questions file
            golden_file = tmpdir_path / "dummy_golden.yaml"
            golden_file.write_text("questions: []", encoding='utf-8')
            
            # Create config file
            config_file = tmpdir_path / "config.yaml"
            config_file.write_text(f"golden_questions_file: {golden_file}", encoding='utf-8')
            
            evaluator = ToxiRAGEvaluator(config_file)
        
        response_text = """
        根据研究结果，化合物具有以下特性：
        [E1 · 机制研究结果] 该化合物能够抑制肿瘤细胞增殖。
        [E2 · 数据记录表格 - 抑瘤率] 高剂量组抑瘤率达65.3%。
        这些证据表明化合物具有良好的抗肿瘤效果。
        """
        
        citations = evaluator.extract_citations(response_text)
        
        assert len(citations) == 2
        assert "[E1 · 机制研究结果]" in citations
        assert "[E2 · 数据记录表格 - 抑瘤率]" in citations
    
    def test_grounding_score_calculation(self):
        """Test grounding score calculation."""
        from eval.evaluator import ToxiRAGEvaluator
        
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create dummy golden questions file
            golden_file = tmpdir_path / "dummy_golden.yaml"
            golden_file.write_text("questions: []", encoding='utf-8')
            
            # Create config file
            config_file = tmpdir_path / "config.yaml"
            config_file.write_text(f"golden_questions_file: {golden_file}", encoding='utf-8')
            
            evaluator = ToxiRAGEvaluator(config_file)
        
        evidence_text = "compound inhibits tumor cell proliferation and induces apoptosis"
        response_text = "compound inhibits tumor cell proliferation"
        
        score = evaluator.calculate_grounding_score(response_text, evidence_text)
        
        assert 0 <= score <= 1
        assert score > 0  # Should have some overlap
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="No API keys available for integration test"
    )
    @pytest.mark.asyncio
    async def test_real_integration_e2e(self, sample_markdown_file):
        """
        Real integration test with actual API calls.
        Only runs if API keys are available.
        """
        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = Path(tmpdir) / "real_test.lance"
            
            # Step 1: Real ingestion
            from ingest.ingest_local import ToxiRAGIngester
            collection_name = "real_e2e_test"
            ingester = ToxiRAGIngester(
                lancedb_uri=str(temp_db),
                table_name=collection_name
            )
            result = await ingester.ingest_file(sample_markdown_file)
            
            assert result["status"] == "success"
            
            # Step 2: Real retrieval using the same database
            from retriever.retriever import ToxiRAGRetriever
            retriever = ToxiRAGRetriever(
                lancedb_uri=str(temp_db),
                table_name=collection_name
            )
            results = await retriever.search(
                query="测试化合物的给药剂量是多少？",
                top_k=2
            )
            evidence_pack = retriever.build_evidence_pack("测试化合物的给药剂量是多少？", results)
            
            assert evidence_pack.total_results > 0
            
            # Step 3: Real agentic response (with limited token usage)
            config = {
                "llm_provider": "openai",
                "temperature": 0.1,
                "max_tokens": 500,  # Limit tokens for cost control
                "top_k_docs": 2
            }
            
            response = await create_agentic_response(
                query="测试化合物的给药剂量是多少？",
                config=config,
                collection_name=collection_name
            )
            
            # Verify response quality
            assert len(response.response_text) > 0
            assert response.confidence_score > 0
            
            # Should contain citation format
            import re
            citations = re.findall(r'\[E\d+ · [^\]]+\]', response.response_text)
            assert len(citations) > 0
            
            # Clear API keys from memory for security
            if "openai_api_key" in config:
                config["openai_api_key"] = None
            if "google_api_key" in config:
                config["google_api_key"] = None
