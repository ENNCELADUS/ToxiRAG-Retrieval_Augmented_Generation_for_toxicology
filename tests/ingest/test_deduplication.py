"""
Unit tests for ingestion deduplication functionality.
Tests the real-time duplicate detection and handling during markdown file ingestion.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, Mock
import pandas as pd
from datetime import datetime

from ingest.ingest_local import ToxiRAGIngester, ingest_markdown_file


@pytest.fixture
def sample_markdown_content():
    """Sample toxicology markdown content for testing."""
    return """# 论文标题: 测试化合物A的抗肝癌研究
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
测试化合物A通过抑制肝癌细胞增殖和诱导凋亡发挥抗肿瘤作用。高剂量组肿瘤抑制率达到65.3%，显著优于对照组（p<0.01）。
（来源：p.4–5，图2–3）

## 研究结论
测试化合物A具有显著的抗肝癌活性，高剂量组抑瘤率达65.3%。
（来源：p.6 / 结论）
"""


@pytest.fixture
def sample_markdown_file_1(tmp_path, sample_markdown_content):
    """Create first sample markdown file."""
    file_path = tmp_path / "sample_study_1.md"
    file_path.write_text(sample_markdown_content, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_markdown_file_2(tmp_path, sample_markdown_content):
    """Create second identical sample markdown file."""
    file_path = tmp_path / "sample_study_2.md"  
    file_path.write_text(sample_markdown_content, encoding='utf-8')
    return file_path


@pytest.fixture  
def mock_ingester():
    """Create mocked ingester with empty database."""
    with patch('ingest.ingest_local.lancedb.connect') as mock_connect:
        # Mock database and table
        mock_db = Mock()
        mock_table = Mock()
        mock_connect.return_value = mock_db
        
        # Create empty dataframe for initial state
        empty_df = pd.DataFrame({
            'content_hash': [],
            'content': [],
            'document_title': [],
            'section_name': [],
            'id': [],
            'embedding': [],
            'file_path': [],
            'section_type': [],
            'chunk_index': [],
            'citation_id': [],
            'section_tag': [],
            'source_page': [],
            'metadata': [],
            'units_version': [],
            'ingestion_timestamp': []
        })
        
        mock_table.to_pandas.return_value = empty_df
        mock_table.add = Mock()
        mock_db.open_table.return_value = mock_table
        mock_db.create_table.return_value = mock_table
        mock_db.drop_table = Mock()
        
        # Mock embedder
        with patch('ingest.ingest_local.OpenAIEmbedder') as mock_embedder_class:
            mock_embedder = Mock()
            mock_embedder.get_embedding.return_value = [0.1] * 3072  # Mock embedding vector
            mock_embedder_class.return_value = mock_embedder
            
            ingester = ToxiRAGIngester()
            ingester._db = mock_db
            ingester._table = mock_table
            ingester.embedder = mock_embedder
            
            yield ingester, mock_table, empty_df


class TestDeduplicationLogic:
    """Test the core deduplication logic."""
    
    @pytest.mark.asyncio
    async def test_handle_duplicates_skip_mode(self, mock_ingester):
        """Test duplicate detection with skip_duplicates=True."""
        ingester, mock_table, _ = mock_ingester
        
        # Create mock data with one duplicate and one new item
        existing_hash = "existing_hash_123"
        new_hash = "new_hash_456"
        
        # Set up existing data in mock table
        existing_df = pd.DataFrame({
            'content_hash': [existing_hash],
            'content': ['Existing content'],
            'document_title': ['Existing Doc'],
            'section_name': ['Existing Section'],
            'id': ['existing_id'],
            'embedding': [[0.1] * 3072],
            'file_path': ['existing.md'],
            'section_type': ['mechanism'],
            'chunk_index': [0],
            'citation_id': ['E1'],
            'section_tag': ['机制研究'],
            'source_page': ['p.1'],
            'metadata': ['{}'],
            'units_version': ['v1.0'],
            'ingestion_timestamp': [datetime.utcnow()]
        })
        mock_table.to_pandas.return_value = existing_df
        
        # Mock ingestion data with duplicate and new content
        mock_ingestion_data = [
            {
                'content_hash': existing_hash,  # Duplicate
                'content': 'Existing content',
                'section_name': 'Duplicate Section',
                'document_title': 'Test Document'
            },
            {
                'content_hash': new_hash,  # New
                'content': 'New content',
                'section_name': 'New Section', 
                'document_title': 'Test Document'
            }
        ]
        
        # Test skip duplicates
        result = await ingester._handle_duplicates(
            mock_ingestion_data, 
            skip_duplicates=True, 
            overwrite_duplicates=False
        )
        
        assert len(result['new_chunks']) == 1
        assert len(result['duplicates']) == 1
        assert len(result['overwritten']) == 0
        assert result['new_chunks'][0]['content_hash'] == new_hash
        assert result['duplicates'][0]['content_hash'] == existing_hash

    @pytest.mark.asyncio
    async def test_handle_duplicates_overwrite_mode(self, mock_ingester):
        """Test duplicate detection with overwrite_duplicates=True."""
        ingester, mock_table, _ = mock_ingester
        
        existing_hash = "existing_hash_123"
        new_hash = "new_hash_456"
        
        # Set up existing data
        existing_df = pd.DataFrame({
            'content_hash': [existing_hash, 'other_hash'],
            'content': ['Existing content', 'Other content'],
            'document_title': ['Existing Doc', 'Other Doc'],
            'section_name': ['Existing Section', 'Other Section'],
            'id': ['existing_id', 'other_id'],
            'embedding': [[0.1] * 3072, [0.2] * 3072],
            'file_path': ['existing.md', 'other.md'],
            'section_type': ['mechanism', 'data'],
            'chunk_index': [0, 0],
            'citation_id': ['E1', 'E2'],
            'section_tag': ['机制研究', '数据表'],
            'source_page': ['p.1', 'p.2'],
            'metadata': ['{}', '{}'],
            'units_version': ['v1.0', 'v1.0'],
            'ingestion_timestamp': [datetime.utcnow(), datetime.utcnow()]
        })
        mock_table.to_pandas.return_value = existing_df
        
        mock_ingestion_data = [
            {
                'content_hash': existing_hash,  # Will be overwritten
                'content': 'Updated content',
                'section_name': 'Updated Section',
                'document_title': 'Test Document'
            },
            {
                'content_hash': new_hash,  # New
                'content': 'New content',
                'section_name': 'New Section',
                'document_title': 'Test Document'
            }
        ]
        
        # Test overwrite duplicates
        result = await ingester._handle_duplicates(
            mock_ingestion_data,
            skip_duplicates=False,
            overwrite_duplicates=True
        )
        
        assert len(result['new_chunks']) == 2  # Both should be added
        assert len(result['duplicates']) == 0
        assert len(result['overwritten']) == 1
        assert result['overwritten'][0]['content_hash'] == existing_hash

    @pytest.mark.asyncio
    async def test_handle_duplicates_allow_mode(self, mock_ingester):
        """Test duplicate detection with allow duplicates (no deduplication)."""
        ingester, mock_table, _ = mock_ingester
        
        existing_hash = "existing_hash_123"
        
        # Set up existing data
        existing_df = pd.DataFrame({
            'content_hash': [existing_hash],
            'content': ['Existing content'],
            'document_title': ['Existing Doc'],
            'section_name': ['Existing Section'],
            'id': ['existing_id'],
            'embedding': [[0.1] * 3072],
            'file_path': ['existing.md'],
            'section_type': ['mechanism'],
            'chunk_index': [0],
            'citation_id': ['E1'],
            'section_tag': ['机制研究'],
            'source_page': ['p.1'],
            'metadata': ['{}'],
            'units_version': ['v1.0'],
            'ingestion_timestamp': [datetime.utcnow()]
        })
        mock_table.to_pandas.return_value = existing_df
        
        mock_ingestion_data = [
            {
                'content_hash': existing_hash,  # Allow duplicate
                'content': 'Existing content',
                'section_name': 'Duplicate Section',
                'document_title': 'Test Document'
            }
        ]
        
        # Test allow duplicates
        result = await ingester._handle_duplicates(
            mock_ingestion_data,
            skip_duplicates=False,
            overwrite_duplicates=False
        )
        
        assert len(result['new_chunks']) == 1  # Should add anyway
        assert len(result['duplicates']) == 0
        assert len(result['overwritten']) == 0


class TestFullIngestionDeduplication:
    """Test full ingestion pipeline with deduplication."""
    
    @pytest.mark.asyncio
    async def test_ingest_identical_files_skip_duplicates(self, sample_markdown_file_1, sample_markdown_file_2):
        """Test ingesting two identical files with skip_duplicates=True."""
        with patch('ingest.ingest_local.lancedb.connect') as mock_connect, \
             patch('ingest.ingest_local.OpenAIEmbedder') as mock_embedder_class:
            
            # Mock database setup
            mock_db = Mock()
            mock_table = Mock()
            mock_connect.return_value = mock_db
            
            # Start with empty database
            empty_df = pd.DataFrame({
                'content_hash': [],
                'content': [],
                'document_title': [],
                'section_name': [],
                'id': [],
                'embedding': [],
                'file_path': [],
                'section_type': [],
                'chunk_index': [],
                'citation_id': [],
                'section_tag': [],
                'source_page': [],
                'metadata': [],
                'units_version': [],
                'ingestion_timestamp': []
            })
            
            # Mock embedder
            mock_embedder = Mock()
            mock_embedder.get_embedding.return_value = [0.1] * 3072
            mock_embedder_class.return_value = mock_embedder
            
            # Track database state across calls
            database_data = []
            
            def mock_to_pandas():
                if database_data:
                    return pd.DataFrame(database_data)
                return empty_df
            
            def mock_add(data):
                database_data.extend(data)
            
            mock_table.to_pandas.side_effect = mock_to_pandas
            mock_table.add.side_effect = mock_add
            mock_db.open_table.return_value = mock_table
            mock_db.create_table.return_value = mock_table
            
            # First ingestion - should succeed
            result1 = await ingest_markdown_file(
                file_path=str(sample_markdown_file_1),
                skip_duplicates=True,
                overwrite_duplicates=False
            )
            
            assert result1['status'] == 'success'
            assert result1['new_chunks'] > 0
            assert result1['duplicate_chunks'] == 0
            first_ingestion_chunks = result1['new_chunks']
            
            # Second ingestion - should detect duplicates
            result2 = await ingest_markdown_file(
                file_path=str(sample_markdown_file_2),
                skip_duplicates=True,
                overwrite_duplicates=False
            )
            
            assert result2['status'] == 'success'
            assert result2['new_chunks'] == 0  # No new chunks
            assert result2['duplicate_chunks'] == first_ingestion_chunks  # All duplicates
            assert len(result2['duplicates']) == first_ingestion_chunks
    
    @pytest.mark.asyncio
    async def test_ingest_identical_files_overwrite(self, sample_markdown_file_1, sample_markdown_file_2):
        """Test ingesting two identical files with overwrite_duplicates=True."""
        with patch('ingest.ingest_local.lancedb.connect') as mock_connect, \
             patch('ingest.ingest_local.OpenAIEmbedder') as mock_embedder_class:
            
            # Mock database setup
            mock_db = Mock()
            mock_table = Mock()
            mock_connect.return_value = mock_db
            
            # Mock embedder
            mock_embedder = Mock()
            mock_embedder.get_embedding.return_value = [0.1] * 3072
            mock_embedder_class.return_value = mock_embedder
            
            # Track database state
            database_data = []
            
            def mock_to_pandas():
                if database_data:
                    return pd.DataFrame(database_data)
                return pd.DataFrame({
                    'content_hash': [], 'content': [], 'document_title': [], 
                    'section_name': [], 'id': [], 'embedding': [], 'file_path': [],
                    'section_type': [], 'chunk_index': [], 'citation_id': [],
                    'section_tag': [], 'source_page': [], 'metadata': [],
                    'units_version': [], 'ingestion_timestamp': []
                })
            
            def mock_add(data):
                database_data.extend(data)
            
            def mock_drop_table(name):
                database_data.clear()
            
            mock_table.to_pandas.side_effect = mock_to_pandas
            mock_table.add.side_effect = mock_add
            mock_db.open_table.return_value = mock_table
            mock_db.create_table.return_value = mock_table
            mock_db.drop_table.side_effect = mock_drop_table
            
            # First ingestion
            result1 = await ingest_markdown_file(
                file_path=str(sample_markdown_file_1),
                skip_duplicates=True,
                overwrite_duplicates=False
            )
            
            assert result1['status'] == 'success'
            first_ingestion_chunks = result1['new_chunks']
            
            # Second ingestion with overwrite
            result2 = await ingest_markdown_file(
                file_path=str(sample_markdown_file_2),
                skip_duplicates=False,
                overwrite_duplicates=True
            )
            
            assert result2['status'] == 'success'
            assert result2['new_chunks'] == first_ingestion_chunks  # Should add all chunks
            assert result2['overwritten_chunks'] == first_ingestion_chunks  # Should overwrite all

    @pytest.mark.asyncio
    async def test_convenience_function_deduplication(self, sample_markdown_file_1):
        """Test the convenience function includes deduplication parameters."""
        with patch('ingest.ingest_local.ToxiRAGIngester') as mock_ingester_class:
            mock_ingester = Mock()
            mock_ingester_class.return_value = mock_ingester
            
            # Mock async ingest_file method
            async def mock_ingest_file(file_path, dry_run=False, skip_duplicates=True, overwrite_duplicates=False):
                return {
                    'status': 'success',
                    'chunks': 5,
                    'new_chunks': 3,
                    'duplicate_chunks': 2,
                    'overwritten_chunks': 0,
                    'duplicates': []
                }
            
            mock_ingester.ingest_file = mock_ingest_file
            
            # Test with custom deduplication parameters
            result = await ingest_markdown_file(
                file_path=str(sample_markdown_file_1),
                skip_duplicates=False,
                overwrite_duplicates=True
            )
            
            assert result['status'] == 'success'
            assert 'duplicate_chunks' in result
            assert 'overwritten_chunks' in result


class TestDeduplicationPerformance:
    """Test performance aspects of deduplication."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_deduplication(self, mock_ingester):
        """Test deduplication performance with larger dataset."""
        ingester, mock_table, _ = mock_ingester
        
        # Create mock data with many duplicates
        base_hashes = [f"hash_{i}" for i in range(100)]
        duplicate_hashes = base_hashes * 5  # 5x duplication
        
        # Set up existing data
        existing_data = []
        for i, hash_val in enumerate(base_hashes):
            existing_data.append({
                'content_hash': hash_val,
                'content': f'Content {i}',
                'document_title': 'Existing Doc',
                'section_name': f'Section {i}',
                'id': f'id_{i}',
                'embedding': [0.1] * 3072,
                'file_path': 'existing.md',
                'section_type': 'mechanism',
                'chunk_index': i,
                'citation_id': f'E{i}',
                'section_tag': '机制研究',
                'source_page': f'p.{i}',
                'metadata': '{}',
                'units_version': 'v1.0',
                'ingestion_timestamp': datetime.utcnow()
            })
        
        existing_df = pd.DataFrame(existing_data)
        mock_table.to_pandas.return_value = existing_df
        
        # Create ingestion data with duplicates
        ingestion_data = []
        for i, hash_val in enumerate(duplicate_hashes):
            ingestion_data.append({
                'content_hash': hash_val,
                'content': f'Content {i % 100}',
                'section_name': f'New Section {i}',
                'document_title': 'New Document'
            })
        
        # Test deduplication performance
        result = await ingester._handle_duplicates(
            ingestion_data,
            skip_duplicates=True,
            overwrite_duplicates=False
        )
        
        # Should find 500 duplicates (all 500 items match the 100 existing hashes)
        assert len(result['duplicates']) == 500
        assert len(result['new_chunks']) == 0
        assert len(result['overwritten']) == 0


# Integration test that can be run manually with API keys
@pytest.mark.skipif(
    True,  # Skip by default to avoid API calls in CI
    reason="Integration test - requires API keys and longer runtime"
)
class TestRealDeduplicationIntegration:
    """Real integration tests with actual embedding generation (manual execution)."""
    
    @pytest.mark.asyncio
    async def test_real_duplicate_detection(self, sample_markdown_file_1, sample_markdown_file_2):
        """Test with real embedding generation (requires API keys)."""
        # This test can be run manually by setting the skip condition to False
        # and ensuring API keys are available
        
        ingester = ToxiRAGIngester()
        
        # First ingestion
        result1 = await ingester.ingest_file(
            sample_markdown_file_1,
            skip_duplicates=True,
            overwrite_duplicates=False
        )
        
        # Second ingestion - should detect duplicates
        result2 = await ingester.ingest_file(
            sample_markdown_file_2,
            skip_duplicates=True,
            overwrite_duplicates=False
        )
        
        assert result1['status'] == 'success'
        assert result2['status'] == 'success'
        assert result2['duplicate_chunks'] > 0
        assert result2['new_chunks'] == 0
