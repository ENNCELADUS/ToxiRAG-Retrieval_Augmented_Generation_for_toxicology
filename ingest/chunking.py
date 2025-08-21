"""
ToxiRAG Document Chunking Strategy
Chunk toxicology documents by section with token-limit fallback for retrieval.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ingest.markdown_schema import ToxicologyDocument
from config.settings import settings


@dataclass
class DocumentChunk:
    """A chunk of a toxicology document with metadata."""
    content: str
    section_name: str
    section_type: str  # e.g., 'animal_info', 'tumor_model', 'data_table', 'mechanism'
    chunk_index: int
    document_title: str
    file_path: Optional[str] = None
    source_page: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Normalized citation format for evidence packs
    citation_id: Optional[str] = None  # e.g., "E1", "E2"
    section_tag: Optional[str] = None  # e.g., "实验分组与给药"


class DocumentChunker:
    """Chunk toxicology documents for vector storage and retrieval."""
    
    def __init__(self, max_chunk_size: int = None, overlap_size: int = None):
        self.max_chunk_size = max_chunk_size or settings.chunk_size
        self.overlap_size = overlap_size or settings.chunk_overlap
        
        # Section type mappings for consistent categorization
        self.section_types = {
            'title': '论文标题',
            'animal_info': '实验动物信息',
            'cell_info': '细胞种类',
            'tumor_model': '肿瘤模型信息',
            'timeline': '实验时间线',
            'experiment_groups': '实验分组与给药',
            'data_table': '数据记录表格',
            'pathology': '病理检测',
            'mechanism': '机制研究结果',
            'other_tests': '其他检测',
            'conclusion': '研究结论',
            'keywords': '关键词'
        }
    
    def chunk_document(self, doc: ToxicologyDocument) -> List[DocumentChunk]:
        """Chunk a parsed toxicology document."""
        chunks = []
        chunk_counter = 1
        
        # 1. Title and source chunk
        if doc.title:
            content = f"# {doc.title}\n"
            if doc.source_info:
                content += f"{doc.source_info}\n"
            
            chunks.append(self._create_chunk(
                content=content.strip(),
                section_name="论文标题",
                section_type="title",
                chunk_index=chunk_counter,
                doc=doc,
                citation_id=f"E{chunk_counter}",
                section_tag="论文标题"
            ))
            chunk_counter += 1
        
        # 2. Animal information chunks
        for i, animal_info in enumerate([doc.mice_info_1, doc.mice_info_2, doc.mice_info_3, doc.mice_info_4], 1):
            if animal_info:
                content = self._format_animal_info(animal_info, f"实验小鼠{i}信息")
                chunks.append(self._create_chunk(
                    content=content,
                    section_name=f"实验小鼠{i}信息",
                    section_type="animal_info",
                    chunk_index=chunk_counter,
                    doc=doc,
                    source_page=animal_info.source_page,
                    citation_id=f"E{chunk_counter}",
                    section_tag=f"实验小鼠{i}信息"
                ))
                chunk_counter += 1
        
        if doc.rat_info:
            content = self._format_animal_info(doc.rat_info, "实验大鼠信息")
            chunks.append(self._create_chunk(
                content=content,
                section_name="实验大鼠信息",
                section_type="animal_info",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=doc.rat_info.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag="实验大鼠信息"
            ))
            chunk_counter += 1
        
        # 3. Cell information chunk
        if doc.cell_info:
            content = self._format_cell_info(doc.cell_info)
            chunks.append(self._create_chunk(
                content=content,
                section_name="细胞种类",
                section_type="cell_info",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=doc.cell_info.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag="细胞种类"
            ))
            chunk_counter += 1
        
        # 4. Tumor model chunk
        if doc.tumor_model:
            content = self._format_tumor_model(doc.tumor_model)
            chunks.append(self._create_chunk(
                content=content,
                section_name="肿瘤模型信息",
                section_type="tumor_model",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=doc.tumor_model.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag="肿瘤模型信息"
            ))
            chunk_counter += 1
        
        # 5. Timeline chunk
        if doc.timeline:
            content = self._format_timeline(doc.timeline)
            chunks.append(self._create_chunk(
                content=content,
                section_name="实验时间线简表",
                section_type="timeline",
                chunk_index=chunk_counter,
                doc=doc,
                citation_id=f"E{chunk_counter}",
                section_tag="实验时间线"
            ))
            chunk_counter += 1
        
        # 6. Experiment groups chunk
        if doc.experiment_groups:
            content = self._format_experiment_groups(doc.experiment_groups)
            chunks.append(self._create_chunk(
                content=content,
                section_name="实验分组与给药",
                section_type="experiment_groups",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=doc.experiment_groups.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag="实验分组与给药"
            ))
            chunk_counter += 1
        
        # 7. Data tables chunks (one per table or group if large)
        for table in doc.data_tables:
            content = self._format_data_table(table)
            
            # Split large tables if needed
            table_chunks = self._split_if_needed(content, f"数据记录表格 - {table.title}")
            for i, chunk_content in enumerate(table_chunks):
                suffix = f" (部分 {i+1})" if len(table_chunks) > 1 else ""
                chunks.append(self._create_chunk(
                    content=chunk_content,
                    section_name=f"数据记录表格 - {table.title}{suffix}",
                    section_type="data_table",
                    chunk_index=chunk_counter,
                    doc=doc,
                    source_page=table.source_page,
                    citation_id=f"E{chunk_counter}",
                    section_tag=f"数据记录表格 - {table.title}",
                    metadata={"table_title": table.title, "units": table.units}
                ))
                chunk_counter += 1
        
        # 8. Pathology chunk
        if doc.pathology:
            content = self._format_pathology(doc.pathology)
            chunks.append(self._create_chunk(
                content=content,
                section_name="病理检测",
                section_type="pathology",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=doc.pathology.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag="病理检测"
            ))
            chunk_counter += 1
        
        # 9. Mechanism chunks
        if doc.mechanism:
            content = self._format_mechanism(doc.mechanism)
            chunks.append(self._create_chunk(
                content=content,
                section_name="机制研究结果",
                section_type="mechanism",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=doc.mechanism.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag="机制研究结果"
            ))
            chunk_counter += 1
        
        # Mechanism data tables
        for table in doc.mechanism_data_tables:
            content = self._format_data_table(table)
            chunks.append(self._create_chunk(
                content=content,
                section_name=f"机制检测数据 - {table.title}",
                section_type="mechanism",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=table.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag=f"机制检测数据 - {table.title}",
                metadata={"table_title": table.title, "units": table.units}
            ))
            chunk_counter += 1
        
        # 10. Other tests chunks
        for table in doc.other_data_tables:
            content = self._format_data_table(table)
            chunks.append(self._create_chunk(
                content=content,
                section_name=f"其他检测 - {table.title}",
                section_type="other_tests",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=table.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag=f"其他检测 - {table.title}",
                metadata={"table_title": table.title, "units": table.units}
            ))
            chunk_counter += 1
        
        # 11. Conclusion chunk
        if doc.conclusion:
            content = self._format_conclusion(doc.conclusion)
            chunks.append(self._create_chunk(
                content=content,
                section_name="研究结论",
                section_type="conclusion",
                chunk_index=chunk_counter,
                doc=doc,
                source_page=doc.conclusion.source_page,
                citation_id=f"E{chunk_counter}",
                section_tag="研究结论"
            ))
            chunk_counter += 1
        
        # 12. Keywords chunk
        if doc.keywords:
            content = f"关键词: {', '.join(doc.keywords)}"
            chunks.append(self._create_chunk(
                content=content,
                section_name="关键词",
                section_type="keywords",
                chunk_index=chunk_counter,
                doc=doc,
                citation_id=f"E{chunk_counter}",
                section_tag="关键词"
            ))
            chunk_counter += 1
        
        return chunks
    
    def _create_chunk(self, content: str, section_name: str, section_type: str, 
                     chunk_index: int, doc: ToxicologyDocument, 
                     source_page: Optional[str] = None, citation_id: Optional[str] = None,
                     section_tag: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> DocumentChunk:
        """Create a document chunk with metadata."""
        return DocumentChunk(
            content=content,
            section_name=section_name,
            section_type=section_type,
            chunk_index=chunk_index,
            document_title=doc.title,
            file_path=doc.file_path,
            source_page=source_page,
            citation_id=citation_id,
            section_tag=section_tag,
            metadata=metadata or {}
        )
    
    def _format_animal_info(self, animal_info, section_title: str) -> str:
        """Format animal information into readable text."""
        content = [f"## {section_title}"]
        
        if animal_info.strain:
            strain_text = animal_info.strain
            if animal_info.strain_norm:
                strain_text += f" (标准化: {animal_info.strain_norm})"
            content.append(f"品系: {strain_text}")
        
        if animal_info.groups_and_counts:
            content.append(f"分组数及每组数量: {animal_info.groups_and_counts}")
        
        if animal_info.total_count:
            content.append(f"总数: {animal_info.total_count}")
        
        if animal_info.sex:
            content.append(f"性别: {animal_info.sex}")
        
        if animal_info.weight:
            weight_text = animal_info.weight
            if animal_info.weight_g:
                weight_text += f" (标准化: {animal_info.weight_g}g)"
            content.append(f"体重: {weight_text}")
        
        if animal_info.age_weeks:
            content.append(f"周龄: {animal_info.age_weeks}周")
        
        if animal_info.source_page:
            content.append(animal_info.source_page)
        
        return "\n".join(content)
    
    def _format_cell_info(self, cell_info) -> str:
        """Format cell information into readable text."""
        content = ["## 细胞种类"]
        
        if cell_info.cell_name:
            content.append(f"细胞名称: {cell_info.cell_name}")
        
        if cell_info.inoculation_method:
            content.append(f"接种方式: {cell_info.inoculation_method}")
        
        if cell_info.inoculation_amount:
            content.append(f"接种量: {cell_info.inoculation_amount}")
        
        if cell_info.source_page:
            content.append(cell_info.source_page)
        
        return "\n".join(content)
    
    def _format_tumor_model(self, tumor_model) -> str:
        """Format tumor model information into readable text."""
        content = ["## 肿瘤模型信息"]
        
        if tumor_model.model_type:
            content.append(f"模型类型: {tumor_model.model_type}")
        
        if tumor_model.tumor_type:
            content.append(f"肿瘤类型: {tumor_model.tumor_type}")
        
        if tumor_model.inoculation_site:
            content.append(f"接种位置: {tumor_model.inoculation_site}")
        
        if tumor_model.tumor_formation_days:
            content.append(f"成瘤时间: {tumor_model.tumor_formation_days}天")
        
        if tumor_model.treatment_start_day:
            content.append(f"给药开始时间: 成瘤后第{tumor_model.treatment_start_day}天")
        
        if tumor_model.total_study_days:
            content.append(f"成瘤总天数: {tumor_model.total_study_days}天")
        
        if tumor_model.source_page:
            content.append(tumor_model.source_page)
        
        return "\n".join(content)
    
    def _format_timeline(self, timeline) -> str:
        """Format timeline into readable text."""
        content = ["## 实验时间线简表"]
        
        for entry in timeline:
            content.append(f"Day {entry.day}: {entry.operation}")
        
        return "\n".join(content)
    
    def _format_experiment_groups(self, groups) -> str:
        """Format experiment groups into readable text."""
        content = ["## 实验分组与给药"]
        
        if groups.group_names_and_counts:
            content.append(f"分组名称及数量: {groups.group_names_and_counts}")
        
        if groups.administration_route:
            content.append(f"给药方式: {groups.administration_route}")
        
        if groups.dose_and_frequency:
            dose_text = groups.dose_and_frequency
            if groups.dose_mg_per_kg:
                dose_text += f" (剂量: {groups.dose_mg_per_kg} mg/kg)"
            if groups.dose_frequency_norm:
                dose_text += f" (频率: {groups.dose_frequency_norm})"
            if groups.daily_equiv_mg_per_kg:
                dose_text += f" (日等效剂量: {groups.daily_equiv_mg_per_kg} mg/kg)"
            content.append(f"药物剂量与频率: {dose_text}")
        
        if groups.treatment_duration:
            content.append(f"给药周期: {groups.treatment_duration}")
        
        if groups.source_page:
            content.append(groups.source_page)
        
        return "\n".join(content)
    
    def _format_data_table(self, table) -> str:
        """Format data table into readable text."""
        content = [f"### {table.title}"]
        
        if table.units:
            content[-1] += f" ({table.units})"
        
        # Format table
        if table.headers and table.rows:
            # Header row
            header_row = "| " + " | ".join(table.headers) + " |"
            content.append(header_row)
            
            # Separator row
            separator = "| " + " | ".join(["---"] * len(table.headers)) + " |"
            content.append(separator)
            
            # Data rows
            for row in table.rows:
                # Pad row to match header length
                padded_row = row + [""] * (len(table.headers) - len(row))
                data_row = "| " + " | ".join(padded_row[:len(table.headers)]) + " |"
                content.append(data_row)
        
        if table.calc_method:
            content.append(f"计算方法: {table.calc_method}")
        
        if table.source_page:
            content.append(table.source_page)
        
        return "\n".join(content)
    
    def _format_pathology(self, pathology) -> str:
        """Format pathology information into readable text."""
        content = ["## 病理检测"]
        
        if pathology.tissues_examined:
            content.append(f"检测组织: {pathology.tissues_examined}")
        
        if pathology.staining_methods:
            content.append(f"染色方法: {pathology.staining_methods}")
        
        if pathology.positive_results:
            content.append(f"阳性结果表现: {pathology.positive_results}")
        
        if pathology.source_page:
            content.append(pathology.source_page)
        
        return "\n".join(content)
    
    def _format_mechanism(self, mechanism) -> str:
        """Format mechanism information into readable text."""
        content = ["## 机制研究结果"]
        
        if mechanism.summary:
            content.append(mechanism.summary)
        
        if mechanism.source_page:
            content.append(mechanism.source_page)
        
        return "\n".join(content)
    
    def _format_conclusion(self, conclusion) -> str:
        """Format conclusion into readable text."""
        content = ["## 研究结论"]
        
        if conclusion.efficacy_summary:
            content.append(conclusion.efficacy_summary)
        
        if conclusion.mechanism_summary:
            content.append(f"机制总结: {conclusion.mechanism_summary}")
        
        if conclusion.research_value:
            content.append(f"研究价值: {conclusion.research_value}")
        
        if conclusion.source_page:
            content.append(conclusion.source_page)
        
        return "\n".join(content)
    
    def _split_if_needed(self, content: str, section_name: str) -> List[str]:
        """Split content if it exceeds max chunk size."""
        if len(content) <= self.max_chunk_size:
            return [content]
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds limit, save current chunk
            if len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If still too large, split by lines
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split very large chunks by lines
                lines = chunk.split('\n')
                current_subchunk = ""
                
                for line in lines:
                    if len(current_subchunk) + len(line) + 1 > self.max_chunk_size and current_subchunk:
                        final_chunks.append(current_subchunk.strip())
                        current_subchunk = line
                    else:
                        if current_subchunk:
                            current_subchunk += "\n" + line
                        else:
                            current_subchunk = line
                
                if current_subchunk:
                    final_chunks.append(current_subchunk.strip())
        
        return final_chunks
    
    def estimate_token_count(self, text: str) -> int:
        """Rough estimate of token count (1 token ≈ 4 characters for Chinese/English mixed)."""
        return len(text) // 4
