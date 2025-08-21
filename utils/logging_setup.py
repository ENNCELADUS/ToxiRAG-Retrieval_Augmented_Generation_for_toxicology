"""
ToxiRAG Logging Configuration
Centralized logging setup using loguru with file and console outputs.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_rotation: str = "10 MB",
    file_retention: str = "7 days"
) -> logger:
    """
    Setup loguru logger with console and file outputs.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        file_rotation: File rotation size/time
        file_retention: How long to keep old log files
    
    Returns:
        Configured logger instance
    """
    # Remove default logger
    logger.remove()
    
    # Console output with color
    if console_output:
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
    
    # File output
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=file_rotation,
            retention=file_retention,
            compression="zip"
        )
    
    return logger


def get_logger(name: str) -> logger:
    """
    Get a named logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
