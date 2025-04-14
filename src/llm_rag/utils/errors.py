"""Centralized error handling for the LLM-RAG system.

This module provides standardized error handling mechanisms, including custom
exception classes, error codes, and utilities for graceful error handling.
It helps maintain consistency in how errors are reported and handled throughout
the codebase.
"""

import functools
import inspect
import sys
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for function return types
T = TypeVar("T")


# Define error categories and codes
class ErrorCategory:
    """Enumeration of error categories for grouping related errors."""

    CONFIGURATION = "CONFIGURATION"
    DATA_ACCESS = "DATA_ACCESS"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    INPUT_VALIDATION = "INPUT_VALIDATION"
    INTERNAL = "INTERNAL"
    MODEL = "MODEL"
    PIPELINE = "PIPELINE"
    VECTORSTORE = "VECTORSTORE"
    DOCUMENT_PROCESSING = "DOCUMENT_PROCESSING"


class ErrorCode:
    """Error codes with categories for specific error conditions."""

    # Configuration errors (1xx)
    INVALID_CONFIG = f"{ErrorCategory.CONFIGURATION}:101"
    MISSING_CONFIG = f"{ErrorCategory.CONFIGURATION}:102"
    ENV_VAR_NOT_SET = f"{ErrorCategory.CONFIGURATION}:103"

    # Data access errors (2xx)
    FILE_NOT_FOUND = f"{ErrorCategory.DATA_ACCESS}:201"
    PERMISSION_DENIED = f"{ErrorCategory.DATA_ACCESS}:202"
    INVALID_FILE_FORMAT = f"{ErrorCategory.DATA_ACCESS}:203"

    # External service errors (3xx)
    SERVICE_UNAVAILABLE = f"{ErrorCategory.EXTERNAL_SERVICE}:301"
    API_ERROR = f"{ErrorCategory.EXTERNAL_SERVICE}:302"
    RATE_LIMIT_EXCEEDED = f"{ErrorCategory.EXTERNAL_SERVICE}:303"

    # Input validation errors (4xx)
    INVALID_INPUT = f"{ErrorCategory.INPUT_VALIDATION}:401"
    MISSING_PARAMETER = f"{ErrorCategory.INPUT_VALIDATION}:402"
    UNSUPPORTED_OPERATION = f"{ErrorCategory.INPUT_VALIDATION}:403"

    # Internal errors (5xx)
    UNEXPECTED_ERROR = f"{ErrorCategory.INTERNAL}:501"
    NOT_IMPLEMENTED = f"{ErrorCategory.INTERNAL}:502"

    # Model errors (6xx)
    MODEL_NOT_FOUND = f"{ErrorCategory.MODEL}:601"
    MODEL_LOAD_ERROR = f"{ErrorCategory.MODEL}:602"
    INFERENCE_ERROR = f"{ErrorCategory.MODEL}:603"

    # Pipeline errors (7xx)
    PIPELINE_CONFIG_ERROR = f"{ErrorCategory.PIPELINE}:701"
    RETRIEVAL_ERROR = f"{ErrorCategory.PIPELINE}:702"
    GENERATION_ERROR = f"{ErrorCategory.PIPELINE}:703"

    # Vectorstore errors (8xx)
    VECTORSTORE_CONNECTION_ERROR = f"{ErrorCategory.VECTORSTORE}:801"
    EMBEDDING_ERROR = f"{ErrorCategory.VECTORSTORE}:802"
    QUERY_ERROR = f"{ErrorCategory.VECTORSTORE}:803"

    # Document processing errors (9xx)
    DOCUMENT_PARSE_ERROR = f"{ErrorCategory.DOCUMENT_PROCESSING}:901"
    CHUNK_ERROR = f"{ErrorCategory.DOCUMENT_PROCESSING}:902"


class LLMRagError(Exception):
    """Base exception class for all LLM-RAG errors.

    This class provides a consistent error interface with support for
    error codes, detailed messages, and capturing the original exception.
    """

    def __init__(
        self,
        message: str,
        error_code: str = ErrorCode.UNEXPECTED_ERROR,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception with detailed information.

        Args:
            message: Human-readable error message
            error_code: Error code from ErrorCode class
            original_exception: The original exception that was caught
            details: Additional context about the error

        """
        self.error_code = error_code
        self.original_exception = original_exception
        self.details = details or {}

        # Build the full message
        full_message = f"[{error_code}] {message}"
        if original_exception:
            full_message += f" (Original error: {str(original_exception)})"

        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary representation.

        This is useful for serializing errors for API responses or logging.

        Returns:
            Dictionary containing error details

        """
        result = {
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details,
        }

        if self.original_exception:
            result["original_error"] = {
                "type": type(self.original_exception).__name__,
                "message": str(self.original_exception),
            }

        return result


# Specific exception classes
class ConfigurationError(LLMRagError):
    """Exception raised for errors in the configuration."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the configuration error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.INVALID_CONFIG, original_exception, details)


class DataAccessError(LLMRagError):
    """Exception raised for errors accessing data sources."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the data source error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.FILE_NOT_FOUND, original_exception, details)


class ExternalServiceError(LLMRagError):
    """Exception raised for errors in external service calls."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the service error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.SERVICE_UNAVAILABLE, original_exception, details)


class ValidationError(LLMRagError):
    """Exception raised for input validation errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the validation error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.INVALID_INPUT, original_exception, details)


class ModelError(LLMRagError):
    """Exception raised for errors related to language models."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the model error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.INFERENCE_ERROR, original_exception, details)


class PipelineError(LLMRagError):
    """Exception raised for errors in the RAG pipeline."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the pipeline error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.PIPELINE_CONFIG_ERROR, original_exception, details)


class VectorstoreError(LLMRagError):
    """Exception raised for errors related to vector stores."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the vectorstore error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.VECTORSTORE_CONNECTION_ERROR, original_exception, details)


class DocumentProcessingError(LLMRagError):
    """Exception raised for errors in document processing."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the document processing error.

        Args:
            message: Error message
            error_code: Optional error code
            original_exception: Original exception that caused this error
            details: Additional error details

        """
        super().__init__(message, error_code or ErrorCode.DOCUMENT_PARSE_ERROR, original_exception, details)


def handle_exceptions(
    error_type: Type[LLMRagError] = LLMRagError,
    error_code: str = ErrorCode.UNEXPECTED_ERROR,
    default_message: str = "An unexpected error occurred",
    log_exception: bool = True,
    reraise: bool = False,
    reraise_for_testing: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., Union[T, Optional[Dict[str, Any]]]]]:
    """Handle exceptions consistently in functions.

    This decorator catches exceptions and wraps them in appropriate LLMRagError
    subclasses. It also provides logging and optional reraising.

    Args:
        error_type: The type of LLMRagError to raise
        error_code: Error code to use
        default_message: Default error message
        log_exception: Whether to log the exception
        reraise: Whether to reraise the wrapped exception
        reraise_for_testing: Whether to reraise exceptions during tests
            (determined by checking if the function is called from a test)

    Returns:
        Decorated function that handles exceptions

    """

    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Optional[Dict[str, Any]]]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[T, Optional[Dict[str, Any]]]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Skip wrapping if it's already the target error type
                if isinstance(e, error_type):
                    if log_exception:
                        logger.error(f"Error in {func.__name__}: {str(e)}")
                    if reraise:
                        raise
                    return None

                # Get function details for better error reporting
                module = inspect.getmodule(func)
                module_name = module.__name__ if module else "unknown_module"

                # Build error message and details
                message = default_message
                if hasattr(e, "__str__"):
                    message = f"{default_message}: {str(e)}"

                details = {
                    "function": func.__name__,
                    "module": module_name,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }

                # Create and log the wrapped exception
                wrapped_error = error_type(
                    message=message,
                    error_code=error_code,
                    original_exception=e,
                    details=details,
                )

                if log_exception:
                    logger.error(
                        f"Error in {func.__name__}: {str(wrapped_error)}",
                        exc_info=True,
                    )

                # Check if we should reraise for testing
                should_reraise_for_testing = reraise_for_testing and (
                    "test_" in module_name or module_name.startswith("tests.")
                )

                if reraise or should_reraise_for_testing:
                    raise wrapped_error from e

                return None

        return wrapper

    return decorator


def convert_exception(
    exception: Exception,
    error_type: Type[LLMRagError] = LLMRagError,
    error_code: str = ErrorCode.UNEXPECTED_ERROR,
    message: Optional[str] = None,
) -> LLMRagError:
    """Convert a standard exception to a LLMRagError.

    Args:
        exception: The exception to convert
        error_type: The type of LLMRagError to create
        error_code: Error code to use
        message: Optional custom message

    Returns:
        A LLMRagError instance

    """
    default_message = message or f"An error occurred: {str(exception)}"

    return error_type(
        message=default_message,
        error_code=error_code,
        original_exception=exception,
    )


def get_exception_details() -> Dict[str, Any]:
    """Get detailed information about the current exception.

    Returns:
        Dictionary with exception type, message, traceback, etc.

    """
    exc_type, exc_value, exc_traceback = sys.exc_info()

    if not exc_type or not exc_value:
        return {"error": "No active exception"}

    tb_list = traceback.extract_tb(exc_traceback)
    frames = []

    for tb_frame in tb_list:
        frames.append(
            {
                "filename": tb_frame.filename,
                "line": tb_frame.lineno,
                "name": tb_frame.name,
                "line_content": tb_frame.line,
            }
        )

    return {
        "type": exc_type.__name__,
        "message": str(exc_value),
        "traceback": frames,
    }
