"""Tests for error handling utilities.

This module contains comprehensive tests for the error handling functionality
provided by the utils.errors module.
"""

import unittest
from unittest.mock import patch

import pytest

from llm_rag.utils.errors import (
    ConfigurationError,
    DataAccessError,
    DocumentProcessingError,
    ErrorCategory,
    ErrorCode,
    ExternalServiceError,
    LLMRagError,
    ModelError,
    PipelineError,
    ValidationError,
    VectorstoreError,
    convert_exception,
    get_exception_details,
    handle_exceptions,
)


class TestErrorCategories(unittest.TestCase):
    """Tests for error categories."""

    def test_error_categories_exist(self):
        """Test that all error categories are defined."""
        # Verify that all expected categories exist
        self.assertEqual(ErrorCategory.CONFIGURATION, 'CONFIGURATION')
        self.assertEqual(ErrorCategory.DATA_ACCESS, 'DATA_ACCESS')
        self.assertEqual(ErrorCategory.EXTERNAL_SERVICE, 'EXTERNAL_SERVICE')
        self.assertEqual(ErrorCategory.INPUT_VALIDATION, 'INPUT_VALIDATION')
        self.assertEqual(ErrorCategory.INTERNAL, 'INTERNAL')
        self.assertEqual(ErrorCategory.MODEL, 'MODEL')
        self.assertEqual(ErrorCategory.PIPELINE, 'PIPELINE')
        self.assertEqual(ErrorCategory.VECTORSTORE, 'VECTORSTORE')
        self.assertEqual(ErrorCategory.DOCUMENT_PROCESSING, 'DOCUMENT_PROCESSING')


class TestErrorCodes(unittest.TestCase):
    """Tests for error codes."""

    def test_error_codes_format(self):
        """Test that error codes follow the expected format."""
        # Check a sample of error codes from different categories
        self.assertEqual(ErrorCode.INVALID_CONFIG, 'CONFIGURATION:101')
        self.assertEqual(ErrorCode.FILE_NOT_FOUND, 'DATA_ACCESS:201')
        self.assertEqual(ErrorCode.SERVICE_UNAVAILABLE, 'EXTERNAL_SERVICE:301')
        self.assertEqual(ErrorCode.INVALID_INPUT, 'INPUT_VALIDATION:401')
        self.assertEqual(ErrorCode.UNEXPECTED_ERROR, 'INTERNAL:501')
        self.assertEqual(ErrorCode.MODEL_NOT_FOUND, 'MODEL:601')
        self.assertEqual(ErrorCode.PIPELINE_CONFIG_ERROR, 'PIPELINE:701')
        self.assertEqual(ErrorCode.VECTORSTORE_CONNECTION_ERROR, 'VECTORSTORE:801')
        self.assertEqual(ErrorCode.DOCUMENT_PARSE_ERROR, 'DOCUMENT_PROCESSING:901')


class TestLLMRagError(unittest.TestCase):
    """Tests for the base LLMRagError class."""

    def test_init_basic(self):
        """Test basic initialization without optional parameters."""
        error = LLMRagError('Test error message')
        self.assertEqual(str(error), f'[{ErrorCode.UNEXPECTED_ERROR}] Test error message')
        self.assertEqual(error.error_code, ErrorCode.UNEXPECTED_ERROR)
        self.assertIsNone(error.original_exception)
        self.assertEqual(error.details, {})

    def test_init_with_error_code(self):
        """Test initialization with custom error code."""
        error = LLMRagError('Test error message', error_code=ErrorCode.INVALID_INPUT)
        self.assertEqual(str(error), f'[{ErrorCode.INVALID_INPUT}] Test error message')
        self.assertEqual(error.error_code, ErrorCode.INVALID_INPUT)

    def test_init_with_original_exception(self):
        """Test initialization with original exception."""
        original = ValueError('Original error')
        error = LLMRagError('Test error message', original_exception=original)
        self.assertIn('Test error message', str(error))
        self.assertIn('Original error: Original error', str(error))
        self.assertEqual(error.original_exception, original)

    def test_init_with_details(self):
        """Test initialization with error details."""
        details = {'param': 'value', 'code': 123}
        error = LLMRagError('Test error message', details=details)
        self.assertEqual(error.details, details)

    def test_to_dict(self):
        """Test conversion to dictionary representation."""
        original = ValueError('Original error')
        details = {'param': 'value'}
        error = LLMRagError(
            'Test error message',
            error_code=ErrorCode.INVALID_INPUT,
            original_exception=original,
            details=details,
        )

        error_dict = error.to_dict()

        self.assertEqual(error_dict['error_code'], ErrorCode.INVALID_INPUT)
        self.assertIn('Test error message', error_dict['message'])
        self.assertEqual(error_dict['details'], details)
        self.assertEqual(error_dict['original_error']['type'], 'ValueError')
        self.assertEqual(error_dict['original_error']['message'], 'Original error')


class TestSpecificErrorClasses(unittest.TestCase):
    """Tests for specific error subclasses."""

    def test_configuration_error(self):
        """Test ConfigurationError initialization."""
        error = ConfigurationError('Config error')
        self.assertEqual(error.error_code, ErrorCode.INVALID_CONFIG)

        # Test with custom error code
        error = ConfigurationError('Config error', error_code=ErrorCode.MISSING_CONFIG)
        self.assertEqual(error.error_code, ErrorCode.MISSING_CONFIG)

    def test_data_access_error(self):
        """Test DataAccessError initialization."""
        error = DataAccessError('Data error')
        self.assertEqual(error.error_code, ErrorCode.FILE_NOT_FOUND)

        # Test with custom error code
        error = DataAccessError('Data error', error_code=ErrorCode.PERMISSION_DENIED)
        self.assertEqual(error.error_code, ErrorCode.PERMISSION_DENIED)

    def test_external_service_error(self):
        """Test ExternalServiceError initialization."""
        error = ExternalServiceError('Service error')
        self.assertEqual(error.error_code, ErrorCode.SERVICE_UNAVAILABLE)

        # Test with custom error code
        error = ExternalServiceError('Service error', error_code=ErrorCode.API_ERROR)
        self.assertEqual(error.error_code, ErrorCode.API_ERROR)

    def test_validation_error(self):
        """Test ValidationError initialization."""
        error = ValidationError('Validation error')
        self.assertEqual(error.error_code, ErrorCode.INVALID_INPUT)

        # Test with custom error code
        error = ValidationError('Validation error', error_code=ErrorCode.MISSING_PARAMETER)
        self.assertEqual(error.error_code, ErrorCode.MISSING_PARAMETER)

    def test_model_error(self):
        """Test ModelError initialization."""
        error = ModelError('Model error')
        self.assertEqual(error.error_code, ErrorCode.INFERENCE_ERROR)

        # Test with custom error code
        error = ModelError('Model error', error_code=ErrorCode.MODEL_LOAD_ERROR)
        self.assertEqual(error.error_code, ErrorCode.MODEL_LOAD_ERROR)

    def test_pipeline_error(self):
        """Test PipelineError initialization."""
        error = PipelineError('Pipeline error')
        self.assertEqual(error.error_code, ErrorCode.PIPELINE_CONFIG_ERROR)

        # Test with custom error code
        error = PipelineError('Pipeline error', error_code=ErrorCode.RETRIEVAL_ERROR)
        self.assertEqual(error.error_code, ErrorCode.RETRIEVAL_ERROR)

    def test_vectorstore_error(self):
        """Test VectorstoreError initialization."""
        error = VectorstoreError('Vectorstore error')
        self.assertEqual(error.error_code, ErrorCode.VECTORSTORE_CONNECTION_ERROR)

        # Test with custom error code
        error = VectorstoreError('Vectorstore error', error_code=ErrorCode.EMBEDDING_ERROR)
        self.assertEqual(error.error_code, ErrorCode.EMBEDDING_ERROR)

    def test_document_processing_error(self):
        """Test DocumentProcessingError initialization."""
        error = DocumentProcessingError('Document error')
        self.assertEqual(error.error_code, ErrorCode.DOCUMENT_PARSE_ERROR)

        # Test with custom error code
        error = DocumentProcessingError('Document error', error_code=ErrorCode.CHUNK_ERROR)
        self.assertEqual(error.error_code, ErrorCode.CHUNK_ERROR)


class TestHandleExceptions(unittest.TestCase):
    """Tests for the handle_exceptions decorator."""

    def test_function_without_error(self):
        """Test decorated function that executes without errors."""

        @handle_exceptions()
        def test_func(value):
            return value * 2

        result = test_func(5)
        self.assertEqual(result, 10)

    def test_function_with_error(self):
        """Test decorated function that raises an error."""

        @handle_exceptions(
            error_type=ValidationError,
            error_code=ErrorCode.INVALID_INPUT,
            default_message='Invalid input',
            reraise=False,
        )
        def test_func(value):
            if value < 0:
                raise ValueError('Value must be positive')
            return value * 2

        # When reraise=False, should return None or error dict
        with patch('llm_rag.utils.errors.logger') as mock_logger:
            result = test_func(-5)
            self.assertIsNone(result)
            mock_logger.error.assert_called_once()

    def test_function_with_error_and_reraise(self):
        """Test decorated function with reraise=True."""

        @handle_exceptions(
            error_type=ValidationError,
            error_code=ErrorCode.INVALID_INPUT,
            default_message='Invalid input',
            reraise=True,
        )
        def test_func(value):
            if value < 0:
                raise ValueError('Value must be positive')
            return value * 2

        # When reraise=True, should raise the wrapped exception
        with self.assertRaises(ValidationError):
            test_func(-5)

    def test_function_with_specific_error(self):
        """Test handling of specific errors differently from others."""

        @handle_exceptions(error_type=ValidationError, reraise=False)
        def test_func(error_type=None):
            if error_type == 'validation':
                raise ValueError('Validation error')
            elif error_type == 'key':
                raise KeyError('Key error')
            return 'success'

        # ValueError should be converted to ValidationError
        result1 = test_func('validation')
        self.assertIsNone(result1)

        # KeyError should be converted to LLMRagError (the base class)
        result2 = test_func('key')
        self.assertIsNone(result2)


class TestConvertException(unittest.TestCase):
    """Tests for the convert_exception function."""

    def test_convert_standard_exception(self):
        """Test converting a standard exception to LLMRagError."""
        original = ValueError('Original error')
        converted = convert_exception(original)

        self.assertIsInstance(converted, LLMRagError)
        self.assertEqual(converted.error_code, ErrorCode.UNEXPECTED_ERROR)
        self.assertEqual(converted.original_exception, original)
        self.assertIn('Original error', str(converted))

    def test_convert_with_custom_message(self):
        """Test converting with custom message."""
        original = ValueError('Original error')
        converted = convert_exception(
            original,
            message='Custom error message',
        )

        self.assertIn('Custom error message', str(converted))

    def test_convert_to_specific_type(self):
        """Test converting to a specific error type."""
        original = ValueError('Original error')
        converted = convert_exception(
            original,
            error_type=ValidationError,
            error_code=ErrorCode.INVALID_INPUT,
        )

        self.assertIsInstance(converted, ValidationError)
        self.assertEqual(converted.error_code, ErrorCode.INVALID_INPUT)


class TestGetExceptionDetails(unittest.TestCase):
    """Tests for the get_exception_details function."""

    @patch('traceback.format_exc')
    def test_get_exception_details(self, mock_format_exc):
        """Test getting exception details."""
        mock_format_exc.return_value = 'Traceback: ...\nValueError: test error'

        # Create a test exception
        try:
            raise ValueError('test error')
        except Exception:
            details = get_exception_details()

        self.assertEqual(details['type'], 'ValueError')
        self.assertEqual(details['message'], 'test error')
        self.assertIn('traceback', details)


@pytest.mark.parametrize(
    'error_class,default_code',
    [
        (ConfigurationError, ErrorCode.INVALID_CONFIG),
        (DataAccessError, ErrorCode.FILE_NOT_FOUND),
        (ExternalServiceError, ErrorCode.SERVICE_UNAVAILABLE),
        (ValidationError, ErrorCode.INVALID_INPUT),
        (ModelError, ErrorCode.INFERENCE_ERROR),
        (PipelineError, ErrorCode.PIPELINE_CONFIG_ERROR),
        (VectorstoreError, ErrorCode.VECTORSTORE_CONNECTION_ERROR),
        (DocumentProcessingError, ErrorCode.DOCUMENT_PARSE_ERROR),
    ],
)
def test_error_default_codes(error_class, default_code):
    """Test that error classes use the correct default error codes."""
    error = error_class('Test error')
    assert error.error_code == default_code


if __name__ == '__main__':
    unittest.main()
