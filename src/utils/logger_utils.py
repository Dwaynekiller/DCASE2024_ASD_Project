#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logger_utils.py

Centralized logging configuration with file rotation.

Provides:
  - configure_logger: sets up a rotating logger with different levels
    for file vs console.
  - handle_errors: decorator for centralized exception handling.

Author: [Your Name]
Date: 2025-03-12
"""

import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Callable, Any
from pathlib import Path
from src import config


LOG_FILE = config.LOG_FILE
DEBUG_MODE = config.DEBUG_MODE

def configure_logger(
    log_file: Path = LOG_FILE,
    file_log_level: int = logging.DEBUG if DEBUG_MODE else logging.WARNING,
    console_log_level: int = logging.INFO if DEBUG_MODE else logging.WARNING
) -> logging.Logger:
    """
    Configures a centralized logger with rotation.
    - file_log_level : logs at least this level to file
    - console_log_level : logs at least this level to console

    Args:
        log_file (Path): Path to the log file.
        file_log_level (int, optional): Logging level for the file handler. Defaults to INFO.
        console_log_level (int, optional): Logging level for the console handler. Defaults to WARNING.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger("DCASE_Project_Logger")

    # Mettre le logger global au niveau le plus bas 
    # afin que les handlers puissent filtrer correctement
    logger.setLevel(logging.DEBUG)

    # Éviter de dupliquer les handlers si configure_logger est appelé plusieurs fois
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # --- File handler ---
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8',
            delay=True
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)

        # --- Console handler ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Décorateur pour la gestion des erreurs (exemple simplifié)
def handle_errors(func: Callable) -> Callable:
    """
    Decorator for centralized exception handling.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: Wrapped function with error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger("DCASE_Project_Logger")
        try:
            return func(*args, **kwargs)
        except ValueError as ve:
            logger.error(f"Validation error in {func.__name__}: {str(ve)}")
        except Exception as e:
            logger.critical(f"Critical error in {func.__name__}: {str(e)}")
            logger.error(f"Unhandled error in {func.__name__}: {e}", exc_info=True)
            raise e
    return wrapper

def close_logger_handlers(logger: logging.Logger) -> None:
    """
    Closes all handlers of a given logger to release file handles.

    Args:
        logger (logging.Logger): The logger instance to clean up.
    """
    for handler in logger.handlers[:]:
        try:
            handler.close()
            logger.removeHandler(handler)
        except Exception as e:
            print(f"Error while closing logger handler: {e}")

# Initialize the logger at module load with default levels
logger = configure_logger()
