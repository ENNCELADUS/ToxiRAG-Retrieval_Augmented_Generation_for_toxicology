"""
ToxiRAG Markdown Schema Parser
Parse structured toxicology markdown documents following the exact template format.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


@dataclass
class AnimalInfo:
    """Animal experiment information."""
    strain: Optional[str] = None
    strain_norm: Optional[str] = None
    groups_and_counts: Optional[str] = None
    total_count: Optional[int] = None
    sex: Optional[str] = None  # {male, female, mixed, 未说明}
    weight: Optional[str] = None
    weight_g: Optional[float] = None  # normalized to grams
    age_weeks: Optional[int] = None
    source_page: Optional[str] = None


@dataclass
class CellInfo:
    """Cell line information."""
    cell_name: Optional[str] = None
    inoculation_method: Optional[str] = None
    inoculation_amount: Optional[str] = None
    source_page: Optional[str] = None


@dataclass
class TumorModelInfo:
    """Tumor model information."""
    model_type: Optional[str] = None
    tumor_type: Optional[str] = None
    inoculation_site: Optional[str] = None
    tumor_formation_days: Optional[int] = None
    treatment_start_day: Optional[int] = None
    total_study_days: Optional[int] = None
    source_page: Optional[str] = None


@dataclass
class TimelineEntry:
    """Timeline entry with day and operation."""
    day: int
    operation: str


@dataclass
class ExperimentGroup:
    """Experimental group information."""
    group_names_and_counts: Optional[str] = None
    administration_route: Optional[str] = None
    dose_and_frequency: Optional[str] = None
    dose_mg_per_kg: Optional[float] = None
    dose_frequency_norm: Optional[str] = None  # {qd, q2d, q3d, qod, qwk, bid, tid, 未说明}
    daily_equiv_mg_per_kg: Optional[float] = None
    treatment_duration: Optional[str] = None
    source_page: Optional[str] = None


@dataclass
class DataTable:
    """Generic data table with metadata."""
    title: str
    headers: List[str]
    rows: List[List[str]]
    source_page: Optional[str] = None
    units: Optional[str] = None
    calc_method: Optional[str] = None  # for tumor volume calculations


@dataclass
class PathologyInfo:
    """Pathology detection information."""
    tissues_examined: Optional[str] = None
    staining_methods: Optional[str] = None
    positive_results: Optional[str] = None
    source_page: Optional[str] = None


@dataclass
class MechanismInfo:
    """Mechanism research results."""
    summary: Optional[str] = None
    source_page: Optional[str] = None


@dataclass
class StudyConclusion:
    """Study conclusions."""
    efficacy_summary: Optional[str] = None
    mechanism_summary: Optional[str] = None
    research_value: Optional[str] = None
    source_page: Optional[str] = None


@dataclass
class ToxicologyDocument:
    """Complete parsed toxicology document."""
    title: str
    source_info: Optional[str] = None
    
    # Animal information (multiple possible)
    mice_info_1: Optional[AnimalInfo] = None
    mice_info_2: Optional[AnimalInfo] = None
    mice_info_3: Optional[AnimalInfo] = None
    mice_info_4: Optional[AnimalInfo] = None
    rat_info: Optional[AnimalInfo] = None
    
    # Experimental setup
    cell_info: Optional[CellInfo] = None
    tumor_model: Optional[TumorModelInfo] = None
    timeline: List[TimelineEntry] = field(default_factory=list)
    experiment_groups: Optional[ExperimentGroup] = None
    
    # Data tables
    data_tables: List[DataTable] = field(default_factory=list)
    
    # Analysis
    pathology: Optional[PathologyInfo] = None
    mechanism: Optional[MechanismInfo] = None
    mechanism_data_tables: List[DataTable] = field(default_factory=list)
    other_data_tables: List[DataTable] = field(default_factory=list)
    
    # Conclusions
    conclusion: Optional[StudyConclusion] = None
    keywords: List[str] = field(default_factory=list)
    
    # Metadata
    units_version: str = "v1.0"
    file_path: Optional[str] = None


class MarkdownParser:
    """Parser for toxicology markdown documents following the exact template."""
    
    # Known strain mappings
    STRAIN_MAPPINGS = {
        "C57BL/6": "C57BL/6",
        "C57BL/6J": "C57BL/6",
        "C57BL/6N": "C57BL/6",
        "BALB/c": "BALB/c",
        "BALB/cJ": "BALB/c",
        "KM": "KM",
        "SD": "SD",
        "Sprague-Dawley": "SD",
        "Sprague Dawley": "SD"
    }
    
    # Dose frequency mappings
    FREQUENCY_MAPPINGS = {
        "daily": "qd",
        "once daily": "qd",
        "qd": "qd",
        "every day": "qd",
        "隔日": "q2d",
        "every other day": "q2d",
        "q2d": "q2d",
        "qod": "qod",
        "twice daily": "bid",
        "bid": "bid",
        "bis in die": "bid",
        "三次": "tid",
        "tid": "tid",
        "weekly": "qwk",
        "qwk": "qwk",
        "每周": "qwk"
    }
    
    def __init__(self):
        self.current_section = None
        
    def parse_file(self, file_path: Path) -> ToxicologyDocument:
        """Parse a markdown file following the toxicology template."""
        content = file_path.read_text(encoding='utf-8')
        return self.parse_content(content, str(file_path))
    
    def parse_content(self, content: str, file_path: Optional[str] = None) -> ToxicologyDocument:
        """Parse markdown content into structured document."""
        lines = content.split('\n')
        doc = ToxicologyDocument(title="", file_path=file_path)
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Title
            if line.startswith('# ') and not doc.title:
                doc.title = line[2:].strip()
                # Look for source info on next lines
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('（来源：'):
                    doc.source_info = lines[i + 1].strip()
                i += 1
                continue
            
            # Animal information sections
            if line.startswith('## 实验小鼠1信息'):
                doc.mice_info_1, i = self._parse_animal_section(lines, i)
                continue
            elif line.startswith('## 实验小鼠2信息'):
                doc.mice_info_2, i = self._parse_animal_section(lines, i)
                continue
            elif line.startswith('## 实验小鼠3信息'):
                doc.mice_info_3, i = self._parse_animal_section(lines, i)
                continue
            elif line.startswith('## 实验小鼠4信息'):
                doc.mice_info_4, i = self._parse_animal_section(lines, i)
                continue
            elif line.startswith('## 实验大鼠信息'):
                doc.rat_info, i = self._parse_animal_section(lines, i)
                continue
            
            # Cell information
            elif line.startswith('## 细胞种类'):
                doc.cell_info, i = self._parse_cell_section(lines, i)
                continue
            
            # Tumor model
            elif line.startswith('## 肿瘤模型信息'):
                doc.tumor_model, i = self._parse_tumor_model_section(lines, i)
                continue
            
            # Timeline
            elif line.startswith('## 实验时间线简表'):
                doc.timeline, i = self._parse_timeline_section(lines, i)
                continue
            
            # Experiment groups
            elif line.startswith('## 实验分组与给药'):
                doc.experiment_groups, i = self._parse_experiment_groups_section(lines, i)
                continue
            
            # Data tables
            elif line.startswith('## 数据记录表格'):
                tables, i = self._parse_data_tables_section(lines, i)
                doc.data_tables.extend(tables)
                continue
            
            # Pathology
            elif line.startswith('## 病理检测'):
                doc.pathology, i = self._parse_pathology_section(lines, i)
                continue
            
            # Mechanism
            elif line.startswith('## 机制研究结果'):
                doc.mechanism, tables, i = self._parse_mechanism_section(lines, i)
                doc.mechanism_data_tables.extend(tables)
                continue
            
            # Other tests
            elif line.startswith('## 其他检测'):
                tables, i = self._parse_other_tests_section(lines, i)
                doc.other_data_tables.extend(tables)
                continue
            
            # Conclusion
            elif line.startswith('## 研究结论'):
                doc.conclusion, i = self._parse_conclusion_section(lines, i)
                continue
            
            # Keywords
            elif line.startswith('## 关键词'):
                doc.keywords, i = self._parse_keywords_section(lines, i)
                continue
            
            i += 1
        
        return doc
    
    def _parse_animal_section(self, lines: List[str], start_idx: int) -> Tuple[AnimalInfo, int]:
        """Parse animal information section."""
        animal = AnimalInfo()
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Stop at next section or separator
            if line.startswith('#') or line.startswith('---'):
                break
            
            # Parse field lines
            if line.startswith('- 品系:'):
                strain_raw = line[5:].strip()  # Skip '- 品系:'
                animal.strain = strain_raw if strain_raw else "未说明"
                animal.strain_norm = self.STRAIN_MAPPINGS.get(strain_raw, None)
            elif line.startswith('- 分组数及每组数量:'):
                animal.groups_and_counts = line[10:].strip() or "未说明"
            elif line.startswith('- 总数:'):
                total_str = line[5:].strip()
                if total_str and total_str != "未说明":
                    try:
                        animal.total_count = int(re.search(r'\d+', total_str).group())
                    except (AttributeError, ValueError):
                        pass
            elif line.startswith('- 性别:'):
                sex_raw = line[5:].strip()  # Skip '- 性别:'
                from ingest.normalization import SexNormalizer
                sex_normalizer = SexNormalizer()
                animal.sex = sex_normalizer.normalize_sex(sex_raw)
            elif line.startswith('- 体重:'):
                weight_raw = line[5:].strip()
                animal.weight = weight_raw if weight_raw else "未说明"
                # Normalize to grams
                if weight_raw and weight_raw != "未说明":
                    animal.weight_g = self._normalize_weight_to_grams(weight_raw)
            elif line.startswith('- 周龄:'):
                age_str = line[5:].strip()
                if age_str and age_str != "未说明":
                    try:
                        animal.age_weeks = int(re.search(r'\d+', age_str).group())
                    except (AttributeError, ValueError):
                        pass
            elif line.startswith('（来源：'):
                animal.source_page = line.strip()
            
            i += 1
        
        return animal, i - 1
    
    def _parse_cell_section(self, lines: List[str], start_idx: int) -> Tuple[CellInfo, int]:
        """Parse cell information section."""
        cell = CellInfo()
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') or line.startswith('---'):
                break
            
            if line.startswith('- 细胞名称:'):
                cell.cell_name = line[6:].strip() or "未说明"
            elif line.startswith('- 接种方式:'):
                cell.inoculation_method = line[6:].strip() or "未说明"
            elif line.startswith('- 接种量:'):
                cell.inoculation_amount = line[5:].strip() or "未说明"
            elif line.startswith('（来源：'):
                cell.source_page = line.strip()
            
            i += 1
        
        return cell, i - 1
    
    def _parse_tumor_model_section(self, lines: List[str], start_idx: int) -> Tuple[TumorModelInfo, int]:
        """Parse tumor model information section."""
        model = TumorModelInfo()
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') or line.startswith('---'):
                break
            
            if line.startswith('- 模型类型：'):
                model.model_type = line[6:].strip() or "未说明"
            elif line.startswith('- 肿瘤类型：'):
                model.tumor_type = line[6:].strip() or "未说明"
            elif line.startswith('- 接种位置：'):
                model.inoculation_site = line[6:].strip() or "未说明"
            elif line.startswith('- 成瘤时间（天）：'):
                time_str = line[9:].strip()
                if time_str and time_str != "未说明":
                    try:
                        model.tumor_formation_days = int(re.search(r'\d+', time_str).group())
                    except (AttributeError, ValueError):
                        pass
            elif line.startswith('- 给药开始时间（成瘤后第几天）：'):
                time_str = line[16:].strip()
                if time_str and time_str != "未说明":
                    try:
                        model.treatment_start_day = int(re.search(r'\d+', time_str).group())
                    except (AttributeError, ValueError):
                        pass
            elif line.startswith('- 成瘤总天数：'):
                time_str = line[7:].strip()
                if time_str and time_str != "未说明":
                    try:
                        model.total_study_days = int(re.search(r'\d+', time_str).group())
                    except (AttributeError, ValueError):
                        pass
            elif line.startswith('（来源：'):
                model.source_page = line.strip()
            
            i += 1
        
        return model, i - 1
    
    def _parse_timeline_section(self, lines: List[str], start_idx: int) -> Tuple[List[TimelineEntry], int]:
        """Parse timeline table section."""
        timeline = []
        i = start_idx + 1
        
        # Skip to table content
        while i < len(lines) and not lines[i].strip().startswith('|'):
            i += 1
        
        # Skip header row
        if i < len(lines) and lines[i].strip().startswith('|'):
            i += 1
        # Skip separator row
        if i < len(lines) and lines[i].strip().startswith('|'):
            i += 1
        
        # Parse data rows
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') or line.startswith('---') or not line.startswith('|'):
                break
            
            # Parse table row
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) >= 2:
                day_str = cells[0].strip()
                operation = cells[1].strip()
                
                # Parse day (handle Day X, Day X+Y format)
                day_match = re.search(r'Day\s*(\d+)(?:\+(\d+))?', day_str)
                if day_match:
                    base_day = int(day_match.group(1))
                    extra_day = int(day_match.group(2)) if day_match.group(2) else 0
                    canonical_day = base_day + extra_day
                    timeline.append(TimelineEntry(day=canonical_day, operation=operation))
            
            i += 1
        
        return timeline, i - 1
    
    def _parse_experiment_groups_section(self, lines: List[str], start_idx: int) -> Tuple[ExperimentGroup, int]:
        """Parse experiment groups and dosing section."""
        groups = ExperimentGroup()
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') or line.startswith('---'):
                break
            
            if line.startswith('- 分组名称及数量:'):
                groups.group_names_and_counts = line[8:].strip() or "未说明"
            elif line.startswith('- 给药方式:'):
                groups.administration_route = line[6:].strip() or "未说明"
            elif line.startswith('- 药物剂量与频率:'):
                dose_freq = line[8:].strip() or "未说明"
                groups.dose_and_frequency = dose_freq
                # Parse dose and frequency
                if dose_freq != "未说明":
                    groups.dose_mg_per_kg, groups.dose_frequency_norm = self._parse_dose_frequency(dose_freq)
            elif line.startswith('- 给药周期:'):
                groups.treatment_duration = line[6:].strip() or "未说明"
            elif line.startswith('（'):
                groups.source_page = line.strip()
            
            i += 1
        
        return groups, i - 1
    
    def _parse_dose_frequency(self, dose_freq_str: str) -> Tuple[Optional[float], Optional[str]]:
        """Parse dose and frequency from text."""
        dose_mg_per_kg = None
        freq_norm = None
        
        # Extract dose (mg/kg)
        dose_match = re.search(r'(\d+(?:\.\d+)?)\s*mg/kg', dose_freq_str, re.IGNORECASE)
        if dose_match:
            dose_mg_per_kg = float(dose_match.group(1))
        
        # Extract frequency
        freq_str = dose_freq_str.lower()
        for pattern, norm in self.FREQUENCY_MAPPINGS.items():
            if pattern in freq_str:
                freq_norm = norm
                break
        
        if freq_norm is None:
            freq_norm = "未说明"
        
        return dose_mg_per_kg, freq_norm
    
    def _normalize_weight_to_grams(self, weight_str: str) -> Optional[float]:
        """Normalize weight to grams."""
        if not weight_str or weight_str == "未说明":
            return None
        
        # Extract number
        weight_match = re.search(r'(\d+(?:\.\d+)?)', weight_str)
        if not weight_match:
            return None
        
        weight_val = float(weight_match.group(1))
        
        # Check units and convert
        if 'kg' in weight_str.lower():
            return weight_val * 1000  # kg to g
        else:
            return weight_val  # assume g
    
    def _parse_data_tables_section(self, lines: List[str], start_idx: int) -> Tuple[List[DataTable], int]:
        """Parse data recording tables section."""
        tables = []
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') and not line.startswith('###'):
                break
            
            # Look for subsection headers
            if line.startswith('###'):
                table_title = line[3:].strip()
                table, i = self._parse_single_table(lines, i, table_title)
                if table:
                    tables.append(table)
                continue
            
            i += 1
        
        return tables, i - 1
    
    def _parse_single_table(self, lines: List[str], start_idx: int, title: str) -> Tuple[Optional[DataTable], int]:
        """Parse a single markdown table."""
        i = start_idx + 1
        headers = []
        rows = []
        source_page = None
        units = None
        
        # Extract units from title
        if '（' in title and '）' in title:
            units_match = re.search(r'（([^）]+)）', title)
            if units_match:
                units = units_match.group(1)
        
        # Find table start
        while i < len(lines) and not lines[i].strip().startswith('|'):
            line = lines[i].strip()
            if line.startswith('（来源：'):
                source_page = line
            i += 1
        
        # Parse headers
        if i < len(lines) and lines[i].strip().startswith('|'):
            header_line = lines[i].strip()
            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
            i += 1
            
            # Skip separator
            if i < len(lines) and lines[i].strip().startswith('|'):
                i += 1
        
        # Parse data rows
        while i < len(lines):
            line = lines[i].strip()
            
            if not line.startswith('|') or line.startswith('（来源：'):
                if line.startswith('（来源：'):
                    source_page = line
                    i += 1
                break
            
            row_data = [cell.strip() for cell in line.split('|')[1:-1]]
            if row_data:
                rows.append(row_data)
            
            i += 1
        
        if headers and rows:
            return DataTable(
                title=title,
                headers=headers,
                rows=rows,
                source_page=source_page,
                units=units
            ), i - 1
        
        return None, i - 1
    
    def _parse_pathology_section(self, lines: List[str], start_idx: int) -> Tuple[PathologyInfo, int]:
        """Parse pathology detection section."""
        pathology = PathologyInfo()
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') or line.startswith('---'):
                break
            
            if line.startswith('- 检测组织：'):
                pathology.tissues_examined = line[6:].strip() or "未说明"
            elif line.startswith('- 染色方法：'):
                pathology.staining_methods = line[6:].strip() or "未说明"
            elif line.startswith('- 阳性结果表现：'):
                pathology.positive_results = line[8:].strip() or "未说明"
            elif line.startswith('（来源：'):
                pathology.source_page = line.strip()
            
            i += 1
        
        return pathology, i - 1
    
    def _parse_mechanism_section(self, lines: List[str], start_idx: int) -> Tuple[MechanismInfo, List[DataTable], int]:
        """Parse mechanism research section."""
        mechanism = MechanismInfo()
        tables = []
        i = start_idx + 1
        
        # Parse summary text first
        summary_lines = []
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('###') or line.startswith('#') or line.startswith('---'):
                break
            
            if line.startswith('（来源：'):
                mechanism.source_page = line
                i += 1
                continue
            
            if line and not line.startswith('|'):
                summary_lines.append(line)
            
            i += 1
        
        mechanism.summary = '\n'.join(summary_lines) if summary_lines else "未说明"
        
        # Parse mechanism data tables
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') and not line.startswith('###'):
                break
            
            if line.startswith('###'):
                table_title = line[3:].strip()
                table, i = self._parse_single_table(lines, i, table_title)
                if table:
                    tables.append(table)
                continue
            
            i += 1
        
        return mechanism, tables, i - 1
    
    def _parse_other_tests_section(self, lines: List[str], start_idx: int) -> Tuple[List[DataTable], int]:
        """Parse other tests section with multiple tables."""
        tables = []
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') and not line.startswith('###'):
                break
            
            if line.startswith('###'):
                table_title = line[3:].strip()
                table, i = self._parse_single_table(lines, i, table_title)
                if table:
                    tables.append(table)
                continue
            
            i += 1
        
        return tables, i - 1
    
    def _parse_conclusion_section(self, lines: List[str], start_idx: int) -> Tuple[StudyConclusion, int]:
        """Parse study conclusion section."""
        conclusion = StudyConclusion()
        i = start_idx + 1
        
        conclusion_lines = []
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') or line.startswith('---'):
                break
            
            if line.startswith('（来源：'):
                conclusion.source_page = line
                i += 1
                continue
            
            if line:
                conclusion_lines.append(line)
            
            i += 1
        
        # Combine all conclusion text
        full_conclusion = '\n'.join(conclusion_lines) if conclusion_lines else "未说明"
        conclusion.efficacy_summary = full_conclusion
        
        return conclusion, i - 1
    
    def _parse_keywords_section(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """Parse keywords section."""
        keywords = []
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#'):
                break
            
            # Extract keywords from backticks
            keyword_matches = re.findall(r'`([^`]+)`', line)
            keywords.extend(keyword_matches)
            
            i += 1
        
        return keywords, i - 1
