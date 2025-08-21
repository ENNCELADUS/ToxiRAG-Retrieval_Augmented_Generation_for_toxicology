#!/usr/bin/env python3
"""
ToxiRAG Reindexing CLI
Rebuild vector embeddings and indices for existing knowledge base.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logging_setup import setup_logging


def main():
    """Main CLI entry point for reindexing."""
    parser = argparse.ArgumentParser(
        description="Rebuild ToxiRAG vector embeddings and search indices"
    )
    
    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"LanceDB collection name (default: {settings.collection_name})"
    )
    
    parser.add_argument(
        "--embedding-model",
        default=settings.openai_embed_model,
        help=f"Embedding model to use (default: {settings.openai_embed_model})"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex even if embeddings already exist"
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
    if not settings.has_openai_key():
        logger.error("OpenAI API key required for embedding generation. Set OPENAI_API_KEY in .env")
        sys.exit(1)
    
    logger.info(f"Starting reindexing for collection: {args.collection}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Force reindex: {args.force}")
    
    try:
        # TODO: Import and call actual reindexing logic once implemented
        # from retriever.reindex import rebuild_embeddings
        # 
        # result = rebuild_embeddings(
        #     collection_name=args.collection,
        #     embedding_model=args.embedding_model,
        #     force=args.force
        # )
        # 
        # logger.info(f"Reindexing completed: {result}")
        
        logger.warning("Reindexing logic not yet implemented - this is a placeholder")
        
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
