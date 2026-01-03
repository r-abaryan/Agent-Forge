"""
Logging configuration for AgentForge
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "agentforge", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure logger for AgentForge.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "agentforge.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "agentforge") -> logging.Logger:
    """Get or create logger instance"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger

