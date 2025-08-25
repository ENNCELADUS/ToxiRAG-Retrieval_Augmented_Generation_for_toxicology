"""
Unit tests for markdown schema parsing.
"""

import pytest
from pathlib import Path
from ingest.markdown_schema import MarkdownParser, ToxicologyDocument


class TestMarkdownParser:
    """Test the MarkdownParser class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.parser = MarkdownParser()
    
    def test_parse_title(self):
        """Test parsing document title."""
        content = """# 论文标题: 甘草酸对肝癌小鼠的毒性研究
（来源：p.1 / Journal of TCM / DOI:xxx）

## 实验小鼠1信息
- 品系: C57BL/6J
"""
        doc = self.parser.parse_content(content)
        
        assert doc.title == "甘草酸对肝癌小鼠的毒性研究"
        assert doc.source_info == "（来源：p.1 / Journal of TCM / DOI:xxx）"
    
    def test_parse_animal_info(self):
        """Test parsing animal information."""
        content = """# 论文标题: Test Document

## 实验小鼠1信息
- 品系: C57BL/6J
- 分组数及每组数量: 4组，每组10只
- 总数: 40只
- 性别: 雌雄各半
- 体重: 20-25g
- 周龄: 6-8周
（来源：p.2 / Methods）
"""
        doc = self.parser.parse_content(content)
        
        assert doc.mice_info_1 is not None
        assert doc.mice_info_1.strain == "C57BL/6J"
        assert doc.mice_info_1.strain_norm == "C57BL/6"
        assert doc.mice_info_1.total_count == 40
        assert doc.mice_info_1.sex == "mixed"  # "雌雄各半" contains both 雌 and 雄, should be mixed
        assert doc.mice_info_1.weight == "20-25g"
        assert doc.mice_info_1.source_page == "（来源：p.2 / Methods）"
    
    def test_parse_dose_frequency(self):
        """Test dose and frequency parsing."""
        # Test daily dose
        dose, freq = self.parser._parse_dose_frequency("200 mg/kg daily")
        assert dose == 200.0
        assert freq == "qd"
        
        # Test every other day
        dose, freq = self.parser._parse_dose_frequency("100 mg/kg 隔日")
        assert dose == 100.0
        assert freq == "q2d"
        
        # Test twice daily
        dose, freq = self.parser._parse_dose_frequency("50 mg/kg bid")
        assert dose == 50.0
        assert freq == "bid"
    
    def test_normalize_weight(self):
        """Test weight normalization."""
        # Test grams
        weight = self.parser._normalize_weight_to_grams("25g")
        assert weight == 25.0
        
        # Test kilograms
        weight = self.parser._normalize_weight_to_grams("0.025kg")
        assert weight == 25.0
        
        # Test invalid
        weight = self.parser._normalize_weight_to_grams("未说明")
        assert weight is None
    
    def test_parse_timeline(self):
        """Test timeline parsing."""
        content = """# 论文标题: Test Document

## 实验时间线简表
| 时间点 | 操作内容           |
|--------|--------------------|
| Day 0  | 接种肿瘤细胞        |
| Day 7  | 成瘤完成           |
| Day 10 | 开始给药           |
| Day 21 | 终点采样           |
"""
        doc = self.parser.parse_content(content)
        
        assert len(doc.timeline) == 4
        assert doc.timeline[0].day == 0
        assert doc.timeline[0].operation == "接种肿瘤细胞"
        assert doc.timeline[3].day == 21
        assert doc.timeline[3].operation == "终点采样"
    
    def test_parse_data_table(self):
        """Test data table parsing."""
        content = """# 论文标题: Test Document

## 数据记录表格

### 肿瘤体积变化（mm³）
| 分组 | Day 7 | Day 14 | Day 21 |
|------|-------|--------|--------|
| 对照组 | 100   | 200    | 400    |
| 药物组 | 90    | 150    | 250    |
（来源：p.5 / Fig.2）
"""
        doc = self.parser.parse_content(content)
        
        assert len(doc.data_tables) == 1
        table = doc.data_tables[0]
        assert table.title == "肿瘤体积变化（mm³）"
        assert table.units == "mm³"
        assert len(table.headers) == 4
        assert len(table.rows) == 2
        assert table.rows[0][0] == "对照组"
        assert table.source_page == "（来源：p.5 / Fig.2）"
    
    def test_parse_keywords(self):
        """Test keywords parsing."""
        content = """# 论文标题: Test Document

## 关键词
`#动物实验` `#肿瘤模型` `#药效评价` `#机制研究`
"""
        doc = self.parser.parse_content(content)
        
        assert len(doc.keywords) == 4
        assert "#动物实验" in doc.keywords
        assert "#肿瘤模型" in doc.keywords
    
    def test_empty_content(self):
        """Test parsing empty or minimal content."""
        content = "# 论文标题: Empty Document"
        doc = self.parser.parse_content(content)
        
        assert doc.title == "Empty Document"
        assert doc.mice_info_1 is None
        assert len(doc.timeline) == 0
        assert len(doc.data_tables) == 0
    
    def test_parse_file(self, tmp_path):
        """Test parsing from file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("""# 论文标题: 测试文档

## 实验小鼠1信息
- 品系: KM
- 总数: 20只
""", encoding='utf-8')
        
        documents = self.parser.parse_file(test_file)
        
        assert len(documents) == 1
        doc = documents[0]
        assert doc.title == "测试文档"
        assert doc.file_path == str(test_file)
        assert doc.mice_info_1.strain == "KM"
        assert doc.mice_info_1.strain_norm == "KM"
    
    def test_parse_multiple_documents_file(self, tmp_path):
        """Test parsing multiple documents from a single file."""
        test_file = tmp_path / "multi_test.md"
        test_file.write_text("""# 论文标题: 第一个文档

## 实验小鼠1信息
- 品系: C57BL/6J
- 总数: 20只

---
# 论文标题: 第二个文档

## 实验小鼠1信息
- 品系: BALB/c
- 总数: 30只

---
#  论文标题: 第三个文档

## 实验小鼠1信息
- 品系: KM
- 总数: 40只
""", encoding='utf-8')
        
        documents = self.parser.parse_file(test_file)
        
        assert len(documents) == 3
        assert documents[0].title == "第一个文档"
        assert documents[1].title == "第二个文档"
        assert documents[2].title == "第三个文档"
        
        assert documents[0].mice_info_1.strain == "C57BL/6J"
        assert documents[1].mice_info_1.strain == "BALB/c"
        assert documents[2].mice_info_1.strain == "KM"
