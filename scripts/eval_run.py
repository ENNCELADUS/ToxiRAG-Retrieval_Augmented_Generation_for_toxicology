#!/usr/bin/env python3
"""
ToxiRAG Evaluation CLI
Run evaluation tests against the knowledge base using golden questions.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logging_setup import setup_logging


async def main():
    """Main CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Run ToxiRAG evaluation against golden questions"
    )
    
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("eval/config.yaml"),
        help="Path to evaluation configuration file"
    )
    
    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"LanceDB collection name (default: {settings.collection_name})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini"],
        default=settings.default_llm_provider,
        help=f"LLM provider for evaluation (default: {settings.default_llm_provider})"
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
    if not settings.has_any_llm_key():
        logger.error("At least one LLM API key required. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env")
        sys.exit(1)
    
    # Validate eval config
    if not args.eval_config.exists():
        logger.error(f"Evaluation config not found: {args.eval_config}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting evaluation with config: {args.eval_config}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"LLM provider: {args.llm_provider}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Import and run evaluation
        from eval.evaluator import run_evaluation
        
        result = await run_evaluation(
            eval_config=args.eval_config,
            collection_name=args.collection,
            llm_provider=args.llm_provider,
            output_dir=args.output_dir
        )
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  Questions: {result.passed_questions}/{result.total_questions} passed")
        logger.info(f"  Pass rate: {result.overall_pass_rate:.1%}")
        logger.info(f"  Average grounding score: {result.average_grounding_score:.3f}")
        logger.info(f"  Average citation coverage: {result.average_citation_coverage:.3f}")
        logger.info(f"  Results saved to: {args.output_dir}")
        
        # Return appropriate exit code
        if result.overall_pass_rate < 0.7:  # Less than 70% pass rate
            logger.warning(f"Pass rate {result.overall_pass_rate:.1%} below recommended 70%")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
