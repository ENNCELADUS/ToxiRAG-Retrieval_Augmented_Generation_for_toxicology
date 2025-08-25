"""
Test enhanced data table parsing to capture descriptive content.
Validates that the enhanced parser captures all content types from 肝癌.md patterns.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List

from ingest.markdown_schema import MarkdownParser, ToxicologyDocument, DataTable
from ingest.chunking import DocumentChunker


class TestEnhancedTableParsing:
    """Test enhanced data table parsing with descriptive content."""
    
    @pytest.fixture
    def sample_complex_content(self):
        """Complex markdown content with various data table patterns."""
        return '''# 论文标题: 测试增强解析功能
（来源：p.1）

## 数据记录表格（如有的话，全部都要）

### 肿瘤体积变化（mm³）
未说明（仅报告瘤重）
（来源：p.3–4）

### 小鼠体重变化（g）

| 分组 | Day 0 (始) | Day Z (终) |
|------|------------|------------|
| 多数实验提供始末体重变化 | 约19–30 g | 约19–24 g |

（来源：p.3–4，Table 5–14）

### 肿瘤质量与抑瘤率

| 分组 | 肿瘤质量（g） | 抑瘤率（%） |
|------|----------------|--------------|
| 对照组 | 277±96 | — |
| PPM 5 mg/kg | 74±35 | 73.29 |
| PPM 200 mg/kg | 0.444±0.09 | 48.01 |

（来源：p.3–4 / Table 10–14）

### 肝功能
未说明

### 肾功能
未说明

---

## 研究结论
测试结论内容...
'''

    @pytest.fixture
    def parser(self):
        """Create markdown parser."""
        return MarkdownParser()
    
    @pytest.fixture
    def chunker(self):
        """Create document chunker."""
        return DocumentChunker()
    
    def test_enhanced_data_table_parsing(self, parser, sample_complex_content):
        """Test that enhanced parsing captures all content types."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_complex_content)
            temp_path = Path(f.name)
        
        try:
            # Parse the content
            docs = parser.parse_file(temp_path)
            assert len(docs) == 1, "Should parse exactly one document"
            
            doc = docs[0]
            
            # Should have 5 data tables
            assert len(doc.data_tables) == 5, f"Expected 5 data tables, got {len(doc.data_tables)}"
            
            # Test each data table type
            tables_by_title = {table.title: table for table in doc.data_tables}
            
            # 1. Description-only table
            tumor_volume = tables_by_title["肿瘤体积变化（mm³）"]
            assert tumor_volume.description == "未说明（仅报告瘤重）", "Should capture description"
            assert tumor_volume.source_page == "（来源：p.3–4）", "Should capture source"
            assert len(tumor_volume.headers) == 0, "Should have no headers"
            assert len(tumor_volume.rows) == 0, "Should have no rows"
            assert tumor_volume.units == "mm³", "Should extract units from title"
            
            # 2. Table with data and source
            weight_change = tables_by_title["小鼠体重变化（g）"]
            assert weight_change.description is None, "Should have no description"
            assert weight_change.source_page == "（来源：p.3–4，Table 5–14）", "Should capture source"
            assert len(weight_change.headers) == 3, "Should have 3 headers"
            assert len(weight_change.rows) == 1, "Should have 1 data row"
            assert weight_change.units == "g", "Should extract units from title"
            assert "分组" in weight_change.headers, "Should have expected headers"
            
            # 3. Complex table with multiple rows
            tumor_mass = tables_by_title["肿瘤质量与抑瘤率"]
            assert tumor_mass.description is None, "Should have no description"
            assert tumor_mass.source_page == "（来源：p.3–4 / Table 10–14）", "Should capture source"
            assert len(tumor_mass.headers) == 3, "Should have 3 headers"
            assert len(tumor_mass.rows) == 3, "Should have 3 data rows"
            assert "对照组" in tumor_mass.rows[0], "Should have expected data"
            
            # 4. & 5. Simple description tables
            liver_func = tables_by_title["肝功能"]
            assert liver_func.description == "未说明", "Should capture simple description"
            assert liver_func.source_page is None, "Should have no source"
            assert len(liver_func.headers) == 0, "Should have no headers"
            assert len(liver_func.rows) == 0, "Should have no rows"
            
            kidney_func = tables_by_title["肾功能"]
            assert kidney_func.description == "未说明", "Should capture simple description"
            assert kidney_func.source_page is None, "Should have no source"
            assert len(kidney_func.headers) == 0, "Should have no headers"
            assert len(kidney_func.rows) == 0, "Should have no rows"
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_enhanced_chunking_includes_descriptions(self, parser, chunker, sample_complex_content):
        """Test that chunking includes description content in output."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_complex_content)
            temp_path = Path(f.name)
        
        try:
            # Parse and chunk
            docs = parser.parse_file(temp_path)
            doc = docs[0]
            chunks = chunker.chunk_document(doc)
            
            # Find data table chunks
            data_table_chunks = [c for c in chunks if c.section_type == "data_table"]
            assert len(data_table_chunks) == 5, f"Expected 5 data table chunks, got {len(data_table_chunks)}"
            
            # Test description inclusion in chunks
            chunk_by_name = {chunk.section_name.split(' - ')[1]: chunk for chunk in data_table_chunks}
            
            # Tumor volume chunk should include description
            tumor_volume_chunk = chunk_by_name["肿瘤体积变化（mm³）"]
            assert "未说明（仅报告瘤重）" in tumor_volume_chunk.content, \
                "Chunk should include description text"
            assert "（来源：p.3–4）" in tumor_volume_chunk.content, \
                "Chunk should include source information"
            
            # Weight change chunk should include table
            weight_change_chunk = chunk_by_name["小鼠体重变化（g）"]
            assert "| 分组 |" in weight_change_chunk.content, \
                "Chunk should include table headers"
            assert "多数实验提供始末体重变化" in weight_change_chunk.content, \
                "Chunk should include table data"
            
            # Liver function chunk should include description
            liver_func_chunk = chunk_by_name["肝功能"]
            assert "未说明" in liver_func_chunk.content, \
                "Chunk should include description"
            assert "### 肝功能" in liver_func_chunk.content, \
                "Chunk should include section header"
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_backward_compatibility_with_existing_files(self, parser):
        """Test that existing files still parse correctly with enhanced schema."""
        # Test with single_sample.md
        sample_path = Path(__file__).parent.parent.parent / "data" / "samples" / "single_sample.md"
        
        docs = parser.parse_file(sample_path)
        doc = docs[0]
        
        # Should still have the expected number of data tables
        assert len(doc.data_tables) >= 2, "Should still parse existing data tables"
        
        # Check that first table now has description
        tumor_volume_table = next((t for t in doc.data_tables if "肿瘤体积变化" in t.title), None)
        assert tumor_volume_table is not None, "Should find tumor volume table"
        assert tumor_volume_table.description is not None, "Should capture description from existing file"
        assert "未说明" in tumor_volume_table.description, "Should capture expected description text"
    
    def test_preserves_all_information_types(self, parser):
        """Test that no information is lost during enhanced parsing."""
        content = '''# 论文标题: 信息保存测试

## 数据记录表格（如有的话，全部都要）

### 复杂数据表（units）
描述性文本在这里
更多描述内容

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |
| 数据4 | 数据5 | 数据6 |

额外说明文本
（来源：p.10，表格3）
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            docs = parser.parse_file(temp_path)
            doc = docs[0]
            
            assert len(doc.data_tables) == 1
            table = doc.data_tables[0]
            
            # Should capture all components
            assert table.title == "复杂数据表（units）"
            assert table.units == "units"
            assert "描述性文本在这里" in table.description
            assert "更多描述内容" in table.description
            assert len(table.headers) == 3
            assert len(table.rows) == 2
            assert table.source_page == "（来源：p.10，表格3）"
            
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    # Run the test when executed directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
