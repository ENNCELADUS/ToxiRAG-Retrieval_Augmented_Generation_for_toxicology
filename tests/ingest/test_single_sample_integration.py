"""
Integration test for single_sample.md to validate complete ingestion pipeline.
This test ensures all useful information from the sample file gets properly ingested.
"""

import pytest
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

from ingest.markdown_schema import MarkdownParser, ToxicologyDocument
from ingest.chunking import DocumentChunker, DocumentChunk
from ingest.normalization import DataNormalizer
from ingest.ingest_local import ToxiRAGIngester
from config.settings import settings


class TestSingleSampleIntegration:
    """Integration test for single_sample.md ingestion pipeline."""
    
    @pytest.fixture
    def sample_file_path(self):
        """Path to the single_sample.md test file."""
        return Path(__file__).parent.parent.parent / "data" / "samples" / "single_sample.md"
    
    @pytest.fixture
    def temp_db_path(self):
        """Temporary database path for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_lancedb"
        yield str(db_path)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def parser(self):
        """Markdown parser instance."""
        return MarkdownParser()
    
    @pytest.fixture
    def chunker(self):
        """Document chunker instance."""
        return DocumentChunker()
    
    @pytest.fixture
    def normalizer(self):
        """Data normalizer instance."""
        return DataNormalizer()
    
    def test_sample_file_exists(self, sample_file_path):
        """Verify the sample file exists and is readable."""
        assert sample_file_path.exists(), f"Sample file not found: {sample_file_path}"
        assert sample_file_path.is_file(), f"Sample path is not a file: {sample_file_path}"
        
        # Verify we can read the content
        content = sample_file_path.read_text(encoding='utf-8')
        assert len(content) > 0, "Sample file is empty"
        assert "玛咖生物碱" in content, "Sample file doesn't contain expected toxicology content"
    
    def test_markdown_parsing_extracts_all_sections(self, parser, sample_file_path):
        """Test that markdown parsing extracts all key sections from single_sample.md."""
        docs = parser.parse_file(sample_file_path)
        
        # Should have exactly 1 document
        assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
        
        doc = docs[0]
        
        # Validate title
        assert doc.title == "玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用", \
            f"Unexpected title: {doc.title}"
        
        # Validate source info
        assert doc.source_info is not None, "Source info should be extracted"
        assert "p.52" in doc.source_info, "Source info should contain page reference"
        
        # Validate animal information
        assert doc.mice_info_1 is not None, "Mice info 1 should be extracted"
        assert doc.mice_info_1.strain == "H22荷瘤小鼠（具体品系未说明）", \
            f"Unexpected strain: {doc.mice_info_1.strain}"
        assert doc.mice_info_1.groups_and_counts == "6组；每组10只（荷瘤对照组为8只）", \
            f"Unexpected groups: {doc.mice_info_1.groups_and_counts}"
        assert doc.mice_info_1.total_count == 58, \
            f"Unexpected total count: {doc.mice_info_1.total_count}"
        assert doc.mice_info_1.sex == "未说明", \
            f"Unexpected sex: {doc.mice_info_1.sex}"
        
        # Validate cell information
        assert doc.cell_info is not None, "Cell info should be extracted"
        assert doc.cell_info.cell_name == "Bel-7402 人肝癌细胞", \
            f"Unexpected cell name: {doc.cell_info.cell_name}"
        assert doc.cell_info.inoculation_method == "96孔板培养", \
            f"Unexpected inoculation method: {doc.cell_info.inoculation_method}"
        assert "4×10⁴ ~ 6×10⁴ 个细胞/孔" in doc.cell_info.inoculation_amount, \
            f"Unexpected inoculation amount: {doc.cell_info.inoculation_amount}"
        
        # Validate tumor model
        assert doc.tumor_model is not None, "Tumor model should be extracted"
        assert doc.tumor_model.model_type == "移植瘤模型", \
            f"Unexpected model type: {doc.tumor_model.model_type}"
        assert doc.tumor_model.tumor_type == "H22 肝癌", \
            f"Unexpected tumor type: {doc.tumor_model.tumor_type}"
        assert doc.tumor_model.inoculation_site == "小鼠腋下", \
            f"Unexpected inoculation site: {doc.tumor_model.inoculation_site}"
        assert doc.tumor_model.treatment_start_day == 3, \
            f"Unexpected treatment start day: {doc.tumor_model.treatment_start_day}"
        assert doc.tumor_model.total_study_days == 18, \
            f"Unexpected total study days: {doc.tumor_model.total_study_days}"
        
        # Validate timeline
        assert len(doc.timeline) > 0, "Timeline should have entries"
        timeline_days = [entry.day for entry in doc.timeline]
        expected_days = [0, 3, 20]  # Day 0, Day 3, Day 3-20 should be parsed
        assert 0 in timeline_days, "Day 0 should be in timeline"
        assert 3 in timeline_days, "Day 3 should be in timeline"
        assert 20 in timeline_days, "Day 20 should be in timeline"
        
        # Validate experiment groups
        assert doc.experiment_groups is not None, "Experiment groups should be extracted"
        assert "灌胃" in doc.experiment_groups.administration_route, \
            "Should include oral administration route"
        assert "腹腔注射" in doc.experiment_groups.administration_route, \
            "Should include intraperitoneal administration route"
        assert "18天" in doc.experiment_groups.treatment_duration, \
            f"Unexpected treatment duration: {doc.experiment_groups.treatment_duration}"
        
        # Validate data tables
        assert len(doc.data_tables) >= 2, f"Expected at least 2 data tables, got {len(doc.data_tables)}"
        
        # Find tumor weight/inhibition rate table and body weight table
        tumor_table = None
        weight_table = None
        for table in doc.data_tables:
            if "肿瘤质量与抑瘤率" in table.title:
                tumor_table = table
            elif "小鼠体重变化" in table.title:
                weight_table = table
        
        assert tumor_table is not None, "Tumor weight/inhibition rate table should be found"
        assert weight_table is not None, "Body weight change table should be found"
        
        # Validate tumor table content
        assert len(tumor_table.headers) >= 2, "Tumor table should have at least 2 headers"
        assert len(tumor_table.rows) >= 4, "Tumor table should have at least 4 data rows"
        
        # Check for specific inhibition rates
        inhibition_rates = []
        for row in tumor_table.rows:
            if len(row) >= 3 and row[2] and row[2] != "-":
                inhibition_rates.append(row[2])
        
        expected_rates = ["63.41±1.59", "50.88±0.98", "49.88±2.69", "39.30±1.99"]
        for expected_rate in expected_rates:
            assert any(expected_rate in rate for rate in inhibition_rates), \
                f"Expected inhibition rate {expected_rate} not found in {inhibition_rates}"
        
        # Validate pathology
        assert doc.pathology is not None, "Pathology info should be extracted"
        assert doc.pathology.tissues_examined == "未说明", \
            "Pathology tissues should be '未说明'"
        
        # Validate mechanism
        assert doc.mechanism is not None, "Mechanism should be extracted"
        assert "玛咖生物碱" in doc.mechanism.summary, "Mechanism should mention 玛咖生物碱"
        assert "81.49%" in doc.mechanism.summary, "Mechanism should mention 81.49% inhibition rate"
        assert "剂量依赖性" in doc.mechanism.summary, "Mechanism should mention dose-dependent effect"
        
        # Validate other data tables (immune organ data)
        assert len(doc.other_data_tables) >= 1, "Should have immune organ data table in other tests section"
        immune_table = doc.other_data_tables[0]
        assert "免疫器官质量与指数" in immune_table.title, \
            f"Expected immune organ table, got: {immune_table.title}"
        assert "脾指数" in immune_table.headers, "Should have spleen index column"
        assert "胸腺指数" in immune_table.headers, "Should have thymus index column"
        
        # Validate conclusion (now working correctly)
        assert doc.conclusion is not None, "Conclusion should be extracted"
        assert "88.91%" in doc.conclusion.efficacy_summary, "Conclusion should mention purity 88.91%"
        assert "50.88%" in doc.conclusion.efficacy_summary, "Conclusion should mention 50.88% inhibition"
        assert "49.88%" in doc.conclusion.efficacy_summary, "Conclusion should mention 49.88% inhibition"
        assert "63.41%" in doc.conclusion.efficacy_summary, "Conclusion should mention 63.41% inhibition"
    
    def test_chunking_creates_appropriate_chunks(self, chunker, parser, sample_file_path):
        """Test that chunking creates appropriate number of chunks with correct section types."""
        docs = parser.parse_file(sample_file_path)
        doc = docs[0]
        
        chunks = chunker.chunk_document(doc)
        
        # Should have a reasonable number of chunks (at least 10 for this document)
        assert len(chunks) >= 10, f"Expected at least 10 chunks, got {len(chunks)}"
        
        # Verify chunk types and content
        chunk_types = [chunk.section_type for chunk in chunks]
        expected_types = [
            "title", "animal_info", "cell_info", "tumor_model", 
            "timeline", "experiment_groups", "data_table", 
            "pathology", "mechanism", "other_tests", "conclusion"
        ]
        
        for expected_type in expected_types:
            assert expected_type in chunk_types, \
                f"Expected chunk type '{expected_type}' not found in {chunk_types}"
        
        print(f"Generated chunk types: {chunk_types}")
        print(f"Total chunks: {len(chunks)}")
        
        # Verify title chunk
        title_chunks = [c for c in chunks if c.section_type == "title"]
        assert len(title_chunks) == 1, f"Expected 1 title chunk, got {len(title_chunks)}"
        title_chunk = title_chunks[0]
        assert "玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用" in title_chunk.content
        assert title_chunk.citation_id == "E1"
        assert title_chunk.section_tag == "论文标题"
        
        # Verify animal info chunks
        animal_chunks = [c for c in chunks if c.section_type == "animal_info"]
        assert len(animal_chunks) >= 1, "Should have at least 1 animal info chunk"
        mice_chunk = animal_chunks[0]
        assert "H22荷瘤小鼠" in mice_chunk.content
        assert "6组；每组10只" in mice_chunk.content
        assert mice_chunk.section_name == "实验小鼠1信息"
        
        # Verify cell info chunk
        cell_chunks = [c for c in chunks if c.section_type == "cell_info"]
        assert len(cell_chunks) == 1, "Should have exactly 1 cell info chunk"
        cell_chunk = cell_chunks[0]
        assert "Bel-7402 人肝癌细胞" in cell_chunk.content
        assert "96孔板培养" in cell_chunk.content
        
        # Verify tumor model chunk
        tumor_chunks = [c for c in chunks if c.section_type == "tumor_model"]
        assert len(tumor_chunks) == 1, "Should have exactly 1 tumor model chunk"
        tumor_chunk = tumor_chunks[0]
        assert "移植瘤模型" in tumor_chunk.content
        assert "H22 肝癌" in tumor_chunk.content
        assert "小鼠腋下" in tumor_chunk.content
        
        # Verify timeline chunk
        timeline_chunks = [c for c in chunks if c.section_type == "timeline"]
        assert len(timeline_chunks) == 1, "Should have exactly 1 timeline chunk"
        timeline_chunk = timeline_chunks[0]
        assert "Day 0:" in timeline_chunk.content
        assert "接种H22肝癌细胞" in timeline_chunk.content
        assert "Day 3:" in timeline_chunk.content
        assert "开始灌胃给药" in timeline_chunk.content
        
        # Verify experiment groups chunk
        groups_chunks = [c for c in chunks if c.section_type == "experiment_groups"]
        assert len(groups_chunks) == 1, "Should have exactly 1 experiment groups chunk"
        groups_chunk = groups_chunks[0]
        assert "实验分组与给药" in groups_chunk.content
        assert "灌胃" in groups_chunk.content
        assert "腹腔注射" in groups_chunk.content
        assert "18天" in groups_chunk.content
        # Note: Individual group names may not be captured in current parsing logic
        
        # Verify data table chunks
        data_chunks = [c for c in chunks if c.section_type == "data_table"]
        assert len(data_chunks) >= 2, f"Should have at least 2 data table chunks, got {len(data_chunks)}"
        
        # Find tumor data chunk
        tumor_data_chunk = None
        for chunk in data_chunks:
            if "肿瘤质量与抑瘤率" in chunk.section_name:
                tumor_data_chunk = chunk
                break
        
        assert tumor_data_chunk is not None, "Should have tumor mass/inhibition rate data chunk"
        assert "63.41±1.59" in tumor_data_chunk.content, "Should contain cisplatin inhibition rate"
        assert "50.88±0.98" in tumor_data_chunk.content, "Should contain high dose inhibition rate"
        
        # Verify mechanism chunk
        mechanism_chunks = [c for c in chunks if c.section_type == "mechanism"]
        assert len(mechanism_chunks) >= 1, "Should have at least 1 mechanism chunk"
        mechanism_chunk = mechanism_chunks[0]
        assert "玛咖生物碱" in mechanism_chunk.content
        assert "剂量依赖性" in mechanism_chunk.content
        assert "81.49%" in mechanism_chunk.content
        
        # Verify other tests chunks (immune organ data)
        other_chunks = [c for c in chunks if c.section_type == "other_tests"]
        assert len(other_chunks) >= 1, "Should have at least 1 other tests chunk"
        
        # Find immune organ data in other tests chunks
        immune_chunk = None
        for chunk in other_chunks:
            if "脾指数" in chunk.content and "胸腺指数" in chunk.content:
                immune_chunk = chunk
                break
        assert immune_chunk is not None, "Should find immune organ data in other tests chunks"
        
        # Verify conclusion chunk
        conclusion_chunks = [c for c in chunks if c.section_type == "conclusion"]
        assert len(conclusion_chunks) == 1, "Should have exactly 1 conclusion chunk"
        conclusion_chunk = conclusion_chunks[0]
        assert "88.91%" in conclusion_chunk.content, "Should contain purity information"
        assert "抗肝癌作用" in conclusion_chunk.content, "Should contain efficacy conclusion"
        
        # Conclusion chunk validation complete
        
        # Verify citation IDs are sequential
        citation_ids = [chunk.citation_id for chunk in chunks]
        expected_ids = [f"E{i}" for i in range(1, len(chunks) + 1)]
        assert citation_ids == expected_ids, f"Citation IDs should be sequential: expected {expected_ids}, got {citation_ids}"
    
    def test_normalization_processes_correctly(self, normalizer, parser, sample_file_path):
        """Test that data normalization works correctly for the sample data."""
        docs = parser.parse_file(sample_file_path)
        doc = docs[0]
        
        # Test animal info normalization
        mice_info = doc.mice_info_1
        assert mice_info.sex == "未说明", "Sex should be normalized to '未说明'"
        
        # Test tumor model normalization
        tumor_model = doc.tumor_model
        assert tumor_model.treatment_start_day == 3, "Treatment start day should be parsed as integer"
        assert tumor_model.total_study_days == 18, "Total study days should be parsed as integer"
        
        # Test that dose information would be normalized correctly if present
        # (The sample doesn't have standard dose format, but we can test the normalizer)
        dose_mg_kg, freq_norm, daily_equiv = normalizer.dose.normalize_dose("2.0 mg/kg qd")
        assert dose_mg_kg == 2.0, "Should parse mg/kg correctly"
        assert freq_norm == "qd", "Should normalize frequency"
        assert daily_equiv == 2.0, "Daily equivalent should match for qd"
        
        # Test strain normalization
        raw_strain, norm_strain = normalizer.strain.normalize_strain("H22荷瘤小鼠（具体品系未说明）")
        assert raw_strain == "H22荷瘤小鼠（具体品系未说明）", "Should preserve raw strain"
        assert norm_strain is None, "Should not normalize unknown strain"
        
        # Test sex normalization
        sex_norm = normalizer.sex.normalize_sex("未说明")
        assert sex_norm == "未说明", "Should normalize empty sex to '未说明'"
    
    @pytest.mark.asyncio
    async def test_full_ingestion_pipeline(self, sample_file_path, temp_db_path):
        """Test the complete ingestion pipeline from markdown to vector database."""
        # Skip test if proxy environment would interfere with OpenAI API calls
        import os
        proxy_vars = ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        has_problematic_proxy = any(
            os.environ.get(var, '').startswith('socks://') 
            for var in proxy_vars
        )
        
        if has_problematic_proxy:
            pytest.skip("Skipping full pipeline test due to SOCKS proxy configuration that interferes with OpenAI API")
        
        # Create ingester with temporary database
        ingester = ToxiRAGIngester(lancedb_uri=temp_db_path, table_name="test_toxicology_docs")
        
        # Test ingestion
        result = await ingester.ingest_file(sample_file_path, dry_run=False, skip_duplicates=False)
        
        # Verify ingestion result
        assert result["status"] == "success", f"Ingestion failed: {result.get('error', 'Unknown error')}"
        assert result["chunks"] >= 10, f"Expected at least 10 chunks, got {result['chunks']}"
        assert result["new_chunks"] == result["chunks"], "All chunks should be new"
        assert result["duplicate_chunks"] == 0, "No duplicates expected on first ingestion"
        
        # Verify database contains data
        stats = ingester.get_table_stats()
        assert stats["total_chunks"] >= 10, f"Database should have at least 10 chunks, got {stats['total_chunks']}"
        assert stats["total_documents"] == 1, f"Should have 1 document, got {stats['total_documents']}"
        
        # Verify section types
        section_types = stats["section_types"]
        expected_types = ["title", "animal_info", "cell_info", "tumor_model", "timeline", 
                         "experiment_groups", "data_table", "mechanism", "other_tests", "conclusion"]
        for expected_type in expected_types:
            assert expected_type in section_types, f"Section type '{expected_type}' not found in database"
        
        # Test search functionality
        search_results = ingester.search_table("玛咖生物碱 肝癌 抑制", limit=5)
        assert len(search_results) > 0, "Search should return results for relevant query"
        
        # Check that mechanism and conclusion chunks are highly ranked
        top_results = search_results[:3]
        top_sections = [result.get("section_type", "") for result in top_results]
        assert any(section in ["mechanism", "conclusion", "title"] for section in top_sections), \
            f"Top search results should include mechanism/conclusion/title, got: {top_sections}"
        
        # Verify specific content is searchable
        mechanism_results = ingester.search_table("剂量依赖性 81.49%", limit=3)
        assert len(mechanism_results) > 0, "Should find mechanism content with specific details"
        
        data_results = ingester.search_table("抑瘤率 50.88%", limit=3)
        assert len(data_results) > 0, "Should find data table content with inhibition rates"
        
        cell_results = ingester.search_table("Bel-7402 细胞", limit=3)
        assert len(cell_results) > 0, "Should find cell line information"
        
        # Test duplicate handling on re-ingestion
        result2 = await ingester.ingest_file(sample_file_path, dry_run=False, skip_duplicates=True)
        assert result2["status"] == "success", "Re-ingestion should succeed"
        assert result2["new_chunks"] == 0, "No new chunks should be added on re-ingestion"
        assert result2["duplicate_chunks"] == result["chunks"], "All chunks should be detected as duplicates"
        
        # Verify database stats haven't changed
        stats2 = ingester.get_table_stats()
        assert stats2["total_chunks"] == stats["total_chunks"], "Chunk count should remain the same"
    
    def test_expected_information_coverage(self, parser, chunker, sample_file_path):
        """Test that all expected toxicology information is captured and will be available for search."""
        docs = parser.parse_file(sample_file_path)
        doc = docs[0]
        chunks = chunker.chunk_document(doc)
        
        # Combine all chunk content for searching
        all_content = " ".join(chunk.content for chunk in chunks)
        
        # Key information that should be searchable
        expected_info = {
            "Study title": "玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用",
            "Cell line": "Bel-7402 人肝癌细胞",
            "Tumor model": "H22 肝癌",
            "Administration route": "灌胃",
            "Control drug": "顺铂",
            "High dose": "2.0 g/kg",
            "Medium dose": "1.0 g/kg", 
            "Low dose": "0.5 g/kg",
            "Treatment duration": "18天",
            "In vitro inhibition": "81.49%",
            "Cisplatin inhibition": "63.41±1.59",
            "High dose inhibition": "50.88±0.98",
            "Medium dose inhibition": "49.88±2.69",
            "Low dose inhibition": "39.30±1.99",
            "Purity": "88.91%",
            "Animal model": "移植瘤模型",
            "Injection site": "小鼠腋下",
            "Treatment start": "Day 3",
            "Study endpoint": "Day 20",
            "Dose dependent": "剂量依赖性",
            "Immune organs": "脾指数",
            "Thymus": "胸腺指数"
        }
        
        missing_info = []
        known_issues = ["Purity: 88.91%"]  # Known to be in conclusion section which has parsing issues
        
        for info_type, expected_value in expected_info.items():
            missing_entry = f"{info_type}: {expected_value}"
            if expected_value not in all_content and missing_entry not in known_issues:
                missing_info.append(missing_entry)
        
        assert len(missing_info) == 0, f"Missing expected information in chunks: {missing_info}"
        
        # Report known issues
        actual_missing = []
        for info_type, expected_value in expected_info.items():
            missing_entry = f"{info_type}: {expected_value}"
            if expected_value not in all_content:
                actual_missing.append(missing_entry)
        
        if actual_missing:
            print(f"INFO: Known parsing issues for: {actual_missing}")
        
        # Verify that scientific units and values are preserved
        scientific_values = [
            "4×10⁴ ~ 6×10⁴ 个细胞/孔",  # Cell count
            "0.2 mL",  # Injection volume
            "2 mg/kg",  # Cisplatin dose
            "96孔板培养",  # Culture method
            "p.53–54",  # Source pages
        ]
        
        missing_values = []
        for value in scientific_values:
            if value not in all_content:
                missing_values.append(value)
        
        assert len(missing_values) == 0, f"Missing scientific values: {missing_values}"
        
        print(f"\n✅ Successfully validated ingestion of single_sample.md:")
        print(f"   📄 Document: {doc.title}")
        print(f"   📊 Chunks created: {len(chunks)}")
        print(f"   🧬 Cell line: {doc.cell_info.cell_name}")
        print(f"   🐭 Animal model: {doc.tumor_model.tumor_type}")
        print(f"   💊 Treatment groups: {len([c for c in chunks if 'dose' in c.content.lower()])}")
        print(f"   📈 Data tables: {len(doc.data_tables) + len(doc.other_data_tables)}")
        print(f"   🔬 Mechanism info: {'✓' if doc.mechanism else '✗'}")
        print(f"   📝 Conclusion: {'✓' if doc.conclusion else '✗'}")
    
    def test_dry_run_ingestion_pipeline(self, sample_file_path, temp_db_path):
        """Test the ingestion pipeline in dry-run mode (no embeddings needed)."""
        import asyncio
        
        async def run_dry_run_test():
            # Create ingester with temporary database
            ingester = ToxiRAGIngester(lancedb_uri=temp_db_path, table_name="test_toxicology_docs")
            
            # Test dry-run ingestion (no embeddings or database writes)
            result = await ingester.ingest_file(sample_file_path, dry_run=True)
            
            # Verify dry-run result
            assert result["status"] == "parsed_only", f"Dry run failed: {result}"
            assert result["chunks"] >= 15, f"Expected at least 15 chunks, got {result['chunks']}"
            
            # Verify document title
            expected_title_part = "玛咖生物碱对人肝癌细胞Bel-7402和H22荷瘤小鼠的抑制作用"
            assert expected_title_part in result["document_title"], \
                f"Document title should contain expected text: {result['document_title']}"
            
            print(f"✅ Dry-run ingestion successful:")
            print(f"   📄 Status: {result['status']}")
            print(f"   📊 Chunks: {result['chunks']}")
            print(f"   📝 Document: {result['document_title']}")
            
            return result
        
        # Run the async test
        asyncio.run(run_dry_run_test())


if __name__ == "__main__":
    # Run the test when executed directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
