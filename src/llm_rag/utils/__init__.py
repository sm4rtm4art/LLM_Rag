"""Utility functions for the LLM RAG system.

This package provides utility modules for logging, error handling,
configuration management, and other cross-cutting concerns.
"""

# Import essential functions from modules for easier access
from llm_rag.utils.config import (
    get_env,
    load_config,
    merge_configs,
    validate_config,
)
from llm_rag.utils.errors import (
    ConfigurationError,
    DataAccessError,
    DocumentProcessingError,
    ErrorCode,
    ExternalServiceError,
    LLMRagError,
    ModelError,
    PipelineError,
    ValidationError,
    VectorstoreError,
    convert_exception,
    handle_exceptions,
)
from llm_rag.utils.logging import (
    get_contextual_logger,
    get_logger,
    setup_logging,
)

__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger",
    "get_contextual_logger",
    # Error handling utilities
    "LLMRagError",
    "ConfigurationError",
    "DataAccessError",
    "ExternalServiceError",
    "ValidationError",
    "ModelError",
    "PipelineError",
    "VectorstoreError",
    "DocumentProcessingError",
    "ErrorCode",
    "handle_exceptions",
    "convert_exception",
    # Configuration utilities
    "load_config",
    "validate_config",
    "merge_configs",
    "get_env",
]
