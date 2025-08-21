"""
ToxiRAG Local Ingestion Pipeline
Ingest markdown documents into LanceDB with OpenAI embeddings.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

import lancedb
import pyarrow as pa
from agno.embedder.openai import OpenAIEmbedder

from config.settings import settings
from utils.logging_setup import get_logger
from ingest.markdown_schema import MarkdownParser, ToxicologyDocument
from ingest.chunking import DocumentChunker, DocumentChunk
from ingest.normalization import DataNormalizer

logger = get_logger(__name__)


class ToxiRAGIngester:
    """Ingest toxicology documents into LanceDB with embeddings."""
    
    def __init__(self, 
                 lancedb_uri: Optional[str] = None,
                 table_name: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        self.lancedb_uri = lancedb_uri or settings.lancedb_uri
        self.table_name = table_name or settings.collection_name
        self.embedding_model = embedding_model or settings.openai_embed_model
        
        # Initialize components
        self.parser = MarkdownParser()
        self.chunker = DocumentChunker()
        self.normalizer = DataNormalizer()
        
        # Initialize embedder
        self.embedder = OpenAIEmbedder(
            id=self.embedding_model,
            api_key=settings.openai_api_key
        )
        
        # Database connection (lazy)
        self._db = None
        self._table = None
    
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
                logger.info(f"Opened existing table: {self.table_name}")
            except Exception as e:
                logger.info(f"Table {self.table_name} not found, creating new table: {e}")
                self._table = self._create_table()
        return self._table
    
    def _create_table(self):
        """Create LanceDB table with schema."""
        # Define schema for toxicology documents
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), 3072)),  # OpenAI text-embedding-3-large fixed size
            pa.field("document_title", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("section_name", pa.string()),
            pa.field("section_type", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("citation_id", pa.string()),
            pa.field("section_tag", pa.string()),
            pa.field("source_page", pa.string()),
            pa.field("metadata", pa.string()),  # JSON string
            pa.field("units_version", pa.string()),
            pa.field("ingestion_timestamp", pa.timestamp('us')),
            pa.field("content_hash", pa.string())
        ])
        
        # Create empty table with schema
        empty_data = []
        return self.db.create_table(self.table_name, data=empty_data, schema=schema)
    
    async def ingest_file(self, file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Ingest a single markdown file."""
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Parse document
            doc = self.parser.parse_file(file_path)
            logger.info(f"Parsed document: {doc.title}")
            
            # Chunk document
            chunks = self.chunker.chunk_document(doc)
            logger.info(f"Generated {len(chunks)} chunks")
            
            if dry_run:
                return {
                    "file_path": str(file_path),
                    "document_title": doc.title,
                    "chunks": len(chunks),
                    "status": "parsed_only"
                }
            
            # Generate embeddings and prepare data
            ingestion_data = await self._prepare_ingestion_data(chunks, doc)
            
            # Insert into database
            self.table.add(ingestion_data)
            logger.info(f"Inserted {len(ingestion_data)} chunks into {self.table_name}")
            
            return {
                "file_path": str(file_path),
                "document_title": doc.title,
                "chunks": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "error": str(e),
                "status": "failed"
            }
    
    async def ingest_directory(self, directory_path: Path, pattern: str = "*.md", dry_run: bool = False) -> Dict[str, Any]:
        """Ingest all markdown files in a directory."""
        md_files = list(directory_path.glob(pattern))
        logger.info(f"Found {len(md_files)} markdown files in {directory_path}")
        
        results = []
        for file_path in md_files:
            result = await self.ingest_file(file_path, dry_run=dry_run)
            results.append(result)
        
        # Summarize results
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        parsed_only = [r for r in results if r.get("status") == "parsed_only"]
        
        total_chunks = sum(r.get("chunks", 0) for r in successful + parsed_only)
        
        return {
            "directory": str(directory_path),
            "total_files": len(md_files),
            "successful": len(successful),
            "failed": len(failed),
            "parsed_only": len(parsed_only),
            "total_chunks": total_chunks,
            "results": results
        }
    
    async def _prepare_ingestion_data(self, chunks: List[DocumentChunk], doc: ToxicologyDocument) -> List[Dict[str, Any]]:
        """Prepare chunk data for ingestion including embeddings."""
        ingestion_data = []
        timestamp = datetime.utcnow()
        
        # Generate embeddings for all chunks
        contents = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(contents)} chunks...")
        
        # Generate embeddings synchronously (OpenAI embedder is not async)
        embeddings = []
        for content in contents:
            embedding = self.embedder.get_embedding(content)
            embeddings.append(embedding)
        
        for chunk, embedding in zip(chunks, embeddings):
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(chunk.content)
            
            # Prepare metadata as JSON
            metadata = {
                **(chunk.metadata or {}),
                "units_version": doc.units_version,
                "document_keywords": doc.keywords,
                "original_file_path": doc.file_path
            }
            
            record = {
                "id": f"{Path(doc.file_path).stem}_{chunk.chunk_index}_{content_hash[:8]}",
                "content": chunk.content,
                "embedding": embedding,
                "document_title": chunk.document_title,
                "file_path": chunk.file_path or "",
                "section_name": chunk.section_name,
                "section_type": chunk.section_type,
                "chunk_index": chunk.chunk_index,
                "citation_id": chunk.citation_id or "",
                "section_tag": chunk.section_tag or "",
                "source_page": chunk.source_page or "",
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "units_version": doc.units_version,
                "ingestion_timestamp": timestamp,
                "content_hash": content_hash
            }
            
            ingestion_data.append(record)
        
        return ingestion_data
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def search_table(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the table using vector similarity."""
        if not query.strip():
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.get_embedding(query)
            
            # Search table
            results = self.table.search(query_embedding).limit(limit).to_pandas()
            
            # Convert to dict format
            return results.to_dict('records')
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingested data."""
        try:
            df = self.table.to_pandas()
            
            return {
                "total_chunks": len(df),
                "total_documents": df['document_title'].nunique(),
                "section_types": df['section_type'].value_counts().to_dict(),
                "documents": df['document_title'].unique().tolist(),
                "last_ingestion": df['ingestion_timestamp'].max().isoformat() if len(df) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
            return {"error": str(e)}
    
    def clear_document(self, document_title: str) -> int:
        """Remove all chunks for a specific document."""
        try:
            # Get current count
            df = self.table.to_pandas()
            doc_chunks = df[df['document_title'] == document_title]
            count = len(doc_chunks)
            
            if count > 0:
                # Delete chunks for this document
                remaining_df = df[df['document_title'] != document_title]
                
                # Recreate table with remaining data
                self._table = None  # Reset cached table
                self.db.drop_table(self.table_name)
                
                if len(remaining_df) > 0:
                    self._table = self.db.create_table(self.table_name, data=remaining_df.to_dict('records'))
                else:
                    self._table = self._create_table()
                
                logger.info(f"Removed {count} chunks for document: {document_title}")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to clear document {document_title}: {e}")
            return 0


# Convenience functions for CLI usage
async def ingest_markdown_file(file_path: str, collection_name: str = None, dry_run: bool = False) -> Dict[str, Any]:
    """Convenience function to ingest a single markdown file."""
    ingester = ToxiRAGIngester(table_name=collection_name)
    return await ingester.ingest_file(Path(file_path), dry_run=dry_run)


async def ingest_markdown_files(input_path: str, collection_name: str = None, dry_run: bool = False) -> Dict[str, Any]:
    """Convenience function to ingest markdown files from a path."""
    ingester = ToxiRAGIngester(table_name=collection_name)
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        return await ingester.ingest_file(input_path, dry_run=dry_run)
    elif input_path.is_dir():
        return await ingester.ingest_directory(input_path, dry_run=dry_run)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def get_ingester_stats(collection_name: str = None) -> Dict[str, Any]:
    """Get statistics about ingested data."""
    ingester = ToxiRAGIngester(table_name=collection_name)
    return ingester.get_table_stats()
