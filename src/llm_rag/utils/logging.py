"""Centralized logging configuration for the LLM-RAG system.

This module provides a standardized way to configure and access logging
throughout the codebase. It includes predefined formatters and handlers
for various logging scenarios including console output, file logging,
and structured JSON logging for easier parsing.
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Union

# Default log format with timestamps, level, module, and message
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
JSON_LOG_FORMAT = 'json'  # Special identifier for JSON formatting

# Log levels dictionary for easier configuration
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
            }

        # Add extra attributes if any
        if hasattr(record, 'extra') and record.extra:
            log_data['extra'] = record.extra

        return json.dumps(log_data)


def setup_logging(
    level: Union[str, int] = 'info',
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    module_levels: Optional[Dict[str, Union[str, int]]] = None,
) -> None:
    """Configure the logging system with the specified settings.

    Args:
        level: The base logging level (debug, info, warning, error, critical or
            corresponding integer values)
        log_format: A log format string or 'json' for JSON formatting
        log_file: Optional path to a log file
        max_file_size: Maximum size of log file before rotation (in bytes)
        backup_count: Number of backup files to keep
        module_levels: Dict mapping module names to specific logging levels
            (e.g., {"llm_rag.rag": "debug", "llm_rag.vectorstore": "warning"})

    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Set formatter based on format string
    if log_format == JSON_LOG_FORMAT:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(log_format)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific levels for modules if specified
    if module_levels:
        for module_name, module_level in module_levels.items():
            if isinstance(module_level, str):
                module_level = LOG_LEVELS.get(module_level.lower(), logging.INFO)

            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_level)

    # Log the configuration
    logging.info(f'Logging configured with level: {logging.getLevelName(level)}')
    if log_file:
        logging.info(f'Log file: {log_file}')


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger with the specified name.

    This is the recommended way to get a logger throughout the codebase.

    Args:
        name: The name of the logger, typically __name__ from the calling module

    Returns:
        A configured logger instance

    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra context to log messages."""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """Initialize the adapter with a logger and optional context.

        Args:
            logger: The logger to adapt
            extra: Dictionary of extra context to add to all log messages

        """
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        """Process the log message by adding the extra context."""
        kwargs.setdefault('extra', {}).update(self.extra)
        return msg, kwargs


def get_contextual_logger(name: str, **context) -> LoggerAdapter:
    """Get a logger with additional context information.

    This is useful for adding context like request_id, user_id, etc. to all log messages.

    Args:
        name: The name of the logger
        **context: Key-value pairs to include in all log messages

    Returns:
        A logger adapter with the specified context

    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context)
