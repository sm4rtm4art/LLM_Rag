"""Unit tests for the logging module."""

import io
import json
import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from unittest.mock import patch

import pytest

from llm_rag.utils.logging import (
    LOG_LEVELS,
    JSONFormatter,
    LoggerAdapter,
    get_contextual_logger,
    get_logger,
    setup_logging,
)


class TestLogging:
    """Test cases for the logging utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        # Save original handlers to restore later
        self.root_logger = logging.getLogger()
        self.original_handlers = self.root_logger.handlers.copy()
        self.original_level = self.root_logger.level

        # Clear all handlers before each test
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)

    def teardown_method(self):
        """Tear down test fixtures."""
        # Restore original logging configuration
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)

        for handler in self.original_handlers:
            self.root_logger.addHandler(handler)

        self.root_logger.setLevel(self.original_level)

    def test_json_formatter(self):
        """Test the JSONFormatter correctly formats log records."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test_file.py',
            lineno=123,
            msg='Test message',
            args=(),
            exc_info=None,
        )

        # Format the record
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Check all required fields are present
        assert log_data['level'] == 'INFO'
        assert log_data['name'] == 'test_logger'
        assert log_data['message'] == 'Test message'
        assert log_data['module'] == 'test_file'
        # Note: The actual value might be None depending on how the formatter is implemented
        assert 'function' in log_data
        assert log_data['line'] == 123
        assert 'timestamp' in log_data

    def test_json_formatter_with_exception(self):
        """Test JSONFormatter correctly formats exception information."""
        formatter = JSONFormatter()
        try:
            raise ValueError('Test error')
        except ValueError as e:
            record = logging.LogRecord(
                name='test_logger',
                level=logging.ERROR,
                pathname='test_file.py',
                lineno=123,
                msg='Error occurred',
                args=(),
                exc_info=(type(e), e, e.__traceback__),
            )

        # Format the record
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Check exception info is present
        assert 'exception' in log_data
        assert log_data['exception']['type'] == 'ValueError'
        assert log_data['exception']['message'] == 'Test error'

    def test_json_formatter_with_extra(self):
        """Test JSONFormatter correctly includes extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test_file.py',
            lineno=123,
            msg='Test message',
            args=(),
            exc_info=None,
        )
        record.extra = {'request_id': '123', 'user_id': 'user123'}

        # Format the record
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Check extra fields are included
        assert 'extra' in log_data
        assert log_data['extra']['request_id'] == '123'
        assert log_data['extra']['user_id'] == 'user123'

    def test_setup_logging_defaults(self):
        """Test setup_logging with default parameters."""
        # Capture the standard output
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            setup_logging()

            # Check logger configuration
            root_logger = logging.getLogger()
            assert root_logger.level == logging.INFO
            assert len(root_logger.handlers) == 1
            assert isinstance(root_logger.handlers[0], logging.StreamHandler)

            # Test logging output
            logging.info('Test message')
            log_output = mock_stdout.getvalue()
            assert 'INFO' in log_output
            assert 'Test message' in log_output

    def test_setup_logging_custom_level(self):
        """Test setup_logging with a custom level."""
        setup_logging(level='debug')
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # Also test with integer level
        setup_logging(level=logging.WARNING)
        assert root_logger.level == logging.WARNING

    def test_setup_logging_json_format(self):
        """Test setup_logging with JSON formatting."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            setup_logging(log_format='json')

            # Log a test message
            logging.info('JSON test')
            log_output = mock_stdout.getvalue()

            # The output might contain multiple JSON objects, one per line
            # Get the last line which should contain our test message
            log_lines = log_output.strip().split('\n')
            for line in log_lines:
                try:
                    log_data = json.loads(line)
                    if log_data.get('message') == 'JSON test':
                        # Found our message
                        assert log_data['level'] == 'INFO'
                        return
                except json.JSONDecodeError:
                    continue

            # If we're here, we didn't find our message
            pytest.fail('Expected JSON log message not found')

    def test_setup_logging_with_file(self):
        """Test setup_logging with a log file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            log_file_path = temp_file.name

        try:
            # Configure logging with file output
            setup_logging(log_file=log_file_path)
            root_logger = logging.getLogger()

            # Verify handlers are correctly configured
            assert len(root_logger.handlers) == 2
            assert any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers)

            # Test logging to file
            test_message = 'File logging test'
            logging.info(test_message)

            # Verify file contents
            with open(log_file_path, 'r') as f:
                log_content = f.read()
                assert test_message in log_content

        finally:
            # Clean up
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)

    def test_setup_logging_with_module_levels(self):
        """Test setup_logging with specific module log levels."""
        module_levels = {
            'llm_rag.rag': 'debug',
            'llm_rag.vectorstore': logging.WARNING,
        }

        setup_logging(module_levels=module_levels)

        # Check module-specific levels
        assert logging.getLogger('llm_rag.rag').level == logging.DEBUG
        assert logging.getLogger('llm_rag.vectorstore').level == logging.WARNING

    def test_get_logger(self):
        """Test get_logger returns the correct logger."""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_module'

    def test_logger_adapter_initialization(self):
        """Test LoggerAdapter initialization."""
        base_logger = logging.getLogger('test_adapter')
        extra = {'request_id': '123', 'user_id': 'user456'}
        adapter = LoggerAdapter(base_logger, extra)

        assert adapter.logger == base_logger
        assert adapter.extra == extra

    def test_logger_adapter_process(self):
        """Test LoggerAdapter process method."""
        logger = logging.getLogger('test_adapter')
        extra = {'request_id': '123', 'user_id': 'user456'}
        adapter = LoggerAdapter(logger, extra)

        # Call process method
        msg = 'Test message'
        kwargs = {}
        processed_msg, processed_kwargs = adapter.process(msg, kwargs)

        # Check results
        assert processed_msg == msg
        assert 'extra' in processed_kwargs
        assert processed_kwargs['extra']['request_id'] == '123'
        assert processed_kwargs['extra']['user_id'] == 'user456'

    def test_logger_adapter_process_with_existing_extra(self):
        """Test LoggerAdapter process with existing extra in kwargs."""
        logger = logging.getLogger('test_adapter')
        adapter = LoggerAdapter(logger, {'request_id': '123'})

        # Call process with existing extra
        kwargs = {'extra': {'session_id': 'abc'}}
        _, processed_kwargs = adapter.process('Test', kwargs)

        # Check results - both extras should be merged
        assert processed_kwargs['extra']['request_id'] == '123'
        assert processed_kwargs['extra']['session_id'] == 'abc'

    def test_get_contextual_logger(self):
        """Test get_contextual_logger returns correct adapter."""
        contextual_logger = get_contextual_logger('test_contextual', request_id='req123', user='test_user')

        # Check type and properties
        assert isinstance(contextual_logger, LoggerAdapter)
        assert contextual_logger.extra['request_id'] == 'req123'
        assert contextual_logger.extra['user'] == 'test_user'

    def test_log_levels_dict(self):
        """Test LOG_LEVELS dictionary contains correct mappings."""
        assert LOG_LEVELS['debug'] == logging.DEBUG
        assert LOG_LEVELS['info'] == logging.INFO
        assert LOG_LEVELS['warning'] == logging.WARNING
        assert LOG_LEVELS['error'] == logging.ERROR
        assert LOG_LEVELS['critical'] == logging.CRITICAL
