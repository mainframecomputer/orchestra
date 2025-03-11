# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
import logging

# ANSI escape codes for colors
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Regular colors
    GRAY = '\033[38;5;240m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""

    FORMATS = {
        logging.DEBUG: Colors.GRAY + '%(asctime)s [%(levelname)s] Orchestra: %(message)s' + Colors.RESET,
        logging.INFO: Colors.GREEN + '%(asctime)s [%(levelname)s] Orchestra: %(message)s' + Colors.RESET,
        logging.WARNING: Colors.YELLOW + '%(asctime)s [%(levelname)s] Orchestra: %(message)s' + Colors.RESET,
        logging.ERROR: Colors.RED + '%(asctime)s [%(levelname)s] Orchestra: %(message)s' + Colors.RESET,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD + '%(asctime)s [%(levelname)s] Orchestra: %(message)s' + Colors.RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def configure_logger():
    """Configure the main orchestra logger with colored output."""
    logger = logging.getLogger("mainframe-orchestra")

    # Set log level from environment variable
    log_level = os.getenv("ORCHESTRA_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # Setup file logging if ORCHESTRA_LOG_FILE is set
    log_file = os.getenv("ORCHESTRA_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        file_handler.setLevel(logging.DEBUG)  # File logs capture everything
        logger.addHandler(file_handler)

    # Configure third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)
    logging.getLogger("groq._base_client").setLevel(logging.WARNING)

    return logger


# Configure the logger when this module is imported
logger = configure_logger()