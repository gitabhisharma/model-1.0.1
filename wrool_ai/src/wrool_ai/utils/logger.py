from loguru import logger
import sys
import json
from pathlib import Path

def setup_logger(config: dict = None):
    """Setup application logging"""
    
    # Remove default logger
    logger.remove()
    
    # Add console logging
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    
    # Add file logging if configured
    if config and config.get("logging", {}).get("file"):
        log_path = Path(config["logging"]["file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            rotation="10 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=config["logging"].get("level", "INFO")
        )
    
    return logger