#!/usr/bin/env python3
"""
ToxiRAG Markdown Ingestion CLI
Ingest Markdown files following the strict toxicology template into LanceDB.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logging_setup import setup_logging


def main():
    """Main CLI entry point for Markdown ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest toxicology Markdown files into ToxiRAG knowledge base"
    )
    
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to Markdown file or directory containing .md files"
    )
    
    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"LanceDB collection name (default: {settings.collection_name})"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Chunk size for splitting documents (default: {settings.chunk_size})"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate files without ingesting to database"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else settings.log_level
    logger = setup_logging(log_level)
    
    # Validate API keys
    if not args.dry_run and not settings.has_openai_key():
        logger.error("OpenAI API key required for embedding generation. Set OPENAI_API_KEY in .env")
        sys.exit(1)
    
    # Validate input path
    if not args.input_path.exists():
        logger.error(f"Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    logger.info(f"Starting ingestion from: {args.input_path}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Dry run: {args.dry_run}")
    
    try:
        # Import and call actual ingestion logic
        from ingest.ingest_local import ingest_markdown_files
        import asyncio
        
        result = asyncio.run(ingest_markdown_files(
            input_path=str(args.input_path),
            collection_name=args.collection,
            dry_run=args.dry_run
        ))
        
        logger.info(f"Ingestion completed: {result}")
        
        # Print summary
        if result.get("status") == "success":
            logger.info(f"✅ Successfully ingested document: {result['document_title']}")
            logger.info(f"   Generated {result['chunks']} chunks")
        elif result.get("status") == "parsed_only":
            logger.info(f"✅ Successfully parsed document: {result['document_title']}")
            logger.info(f"   Generated {result['chunks']} chunks (dry run)")
        elif "total_files" in result:
            # Directory ingestion
            logger.info(f"✅ Processed {result['total_files']} files")
            logger.info(f"   Successful: {result['successful']}")
            logger.info(f"   Failed: {result['failed']}")
            logger.info(f"   Total chunks: {result['total_chunks']}")
        else:
            logger.error(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
