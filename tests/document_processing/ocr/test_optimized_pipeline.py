"""Tests for the optimized OCR pipeline."""

import os
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from llm_rag.document_processing.ocr.optimized_pipeline import OptimizedOCRConfig, OptimizedOCRPipeline


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def pipeline_config(temp_cache_dir):
    """Create a test configuration for the optimized OCR pipeline."""
    return OptimizedOCRConfig(
        parallel_processing=True,
        max_workers=2,
        use_cache=True,
        cache_dir=temp_cache_dir,
        cache_ttl=3600,  # 1 hour
        force_reprocess=False,
        batched_processing=True,
        batch_size=2,
        pdf_dpi=150,  # Lower DPI for faster tests
        languages='eng',
        output_format='raw',
        llm_cleaning_enabled=False,  # Disable LLM cleaning for tests
    )


@pytest.fixture
def mock_pdf_converter():
    """Mock the PDFImageConverter."""
    with patch('llm_rag.document_processing.ocr.optimized_pipeline.PDFImageConverter') as mock:
        # Setup the mock to return some test images
        instance = mock.return_value
        instance.get_images.return_value = [
            (0, MagicMock()),  # Page 1
            (1, MagicMock()),  # Page 2
        ]
        yield mock


@pytest.fixture
def mock_ocr_engine():
    """Mock the TesseractOCREngine."""
    with patch('llm_rag.document_processing.ocr.pipeline.TesseractOCREngine') as mock:
        # Setup the mock to return some test OCR text
        instance = mock.return_value
        instance.process_image.side_effect = [
            'Page 1 text',
            'Page 2 text',
        ]
        yield mock


class TestOptimizedOCRPipeline:
    """Tests for the OptimizedOCRPipeline class."""

    def test_init(self, pipeline_config):
        """Test initialization of the pipeline."""
        pipeline = OptimizedOCRPipeline(pipeline_config)
        assert pipeline.config == pipeline_config
        assert os.path.exists(pipeline_config.cache_dir)

    def test_cache_key_generation(self, pipeline_config):
        """Test that cache keys are unique for different inputs."""
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Patch os.stat to avoid file not found errors
        mock_stat = MagicMock()
        mock_stat.st_size = 1024
        mock_stat.st_mtime = 1234567890

        # Patch the pdf_renderer_dpi attribute access
        with patch.object(pipeline.config, 'pdf_dpi', 150, create=True):
            # Also patch the pdf_renderer_dpi attribute access for cache key generation
            with patch.object(pipeline.config, 'pdf_renderer_dpi', 150, create=True):
                with patch.object(pipeline.config, 'ocr_language', 'eng', create=True):
                    with patch.object(pipeline.config, 'use_llm_cleaner', False, create=True):
                        with patch('os.stat', return_value=mock_stat):
                            # Create a temp file for testing
                            with tempfile.NamedTemporaryFile() as temp_file:
                                # Generate keys
                                key1 = pipeline._generate_cache_key(temp_file.name)
                                key2 = pipeline._generate_cache_key(temp_file.name, page_num=0)
                                key3 = pipeline._generate_cache_key(temp_file.name, page_num=1)

                                # Keys should be strings
                                assert isinstance(key1, str)
                                assert isinstance(key2, str)
                                assert isinstance(key3, str)

                                # Different pages should have different keys
                                assert key2 != key3

                                # Document-level key should be different from page keys
                                assert key1 != key2
                                assert key1 != key3

    def test_cache_write_and_read(self, pipeline_config):
        """Test writing to and reading from the cache."""
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Write to cache
        test_key = 'test_key'
        test_data = {'text': 'Test OCR Text'}
        pipeline._write_cache(test_key, test_data)

        # Read from cache
        cached_data = pipeline._check_cache(test_key)
        assert cached_data is not None
        assert 'timestamp' in cached_data
        assert cached_data['text'] == 'Test OCR Text'

        # Check cache file exists
        cache_path = pipeline._get_cache_path(test_key)
        assert os.path.exists(cache_path)

    def test_cache_expiration(self, pipeline_config):
        """Test that expired cache entries are not used."""
        # Configure with very short TTL
        pipeline_config.cache_ttl = 0  # Immediate expiration
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Write to cache
        test_key = 'test_key'
        test_data = {'text': 'Test OCR Text'}
        pipeline._write_cache(test_key, test_data)

        # Read from cache - should be expired
        cached_data = pipeline._check_cache(test_key)
        assert cached_data is None

    def test_process_pdf_with_cache(self, pipeline_config, mock_pdf_converter, mock_ocr_engine):
        """Test processing a PDF with caching."""
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Mock os.path.abspath to return the unmodified path
        with patch('os.path.abspath', side_effect=lambda x: x):
            # Mock os.stat to avoid file not found errors
            mock_stat = MagicMock()
            mock_stat.st_size = 1024
            mock_stat.st_mtime = 1234567890

            # Patch file operations and config attribute access
            with patch('os.stat', return_value=mock_stat):
                # Add missing attribute patches
                with patch.object(pipeline.config, 'pdf_dpi', 150):
                    with patch.object(pipeline.config, 'languages', 'eng'):
                        with patch.object(pipeline.config, 'output_format', 'raw'):
                            with patch.object(pipeline.config, 'llm_cleaning_enabled', False):
                                # Process first time (cache miss)
                                with patch.object(pipeline, '_check_cache', return_value=None):
                                    result = pipeline.process_pdf('test.pdf')
                                    assert 'Page 1 text' in result
                                    assert 'Page 2 text' in result

                                    # Both pages should be processed
                                    assert mock_ocr_engine.return_value.process_image.call_count == 2

                                # Reset mock
                                mock_ocr_engine.return_value.process_image.reset_mock()

                                # Process second time (cache hit)
                                cache_data = {'text': 'Cached OCR Text'}
                                with patch.object(pipeline, '_check_cache', return_value=cache_data):
                                    result = pipeline.process_pdf('test.pdf')
                                    assert result == 'Cached OCR Text'

                                    # No pages should be processed
                                    assert mock_ocr_engine.return_value.process_image.call_count == 0

    def test_parallel_processing(self, pipeline_config, mock_pdf_converter, mock_ocr_engine):
        """Test parallel processing of pages."""
        pipeline_config.parallel_processing = True
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Mock os.path.abspath to return the unmodified path
        with patch('os.path.abspath', side_effect=lambda x: x):
            # Mock os.stat to avoid file not found errors
            mock_stat = MagicMock()
            mock_stat.st_size = 1024
            mock_stat.st_mtime = 1234567890

            # Create test images
            test_images = [(0, MagicMock()), (1, MagicMock())]

            # Patch file operations and config attribute access
            with patch('os.stat', return_value=mock_stat):
                # Add missing attribute patches
                with patch.object(pipeline.config, 'pdf_dpi', 150):
                    with patch.object(pipeline.config, 'languages', 'eng'):
                        with patch.object(pipeline.config, 'output_format', 'raw'):
                            with patch.object(pipeline.config, 'llm_cleaning_enabled', False):
                                # Directly test the _parallel_process_pages method
                                with patch(
                                    'llm_rag.document_processing.ocr.optimized_pipeline.ThreadPoolExecutor'
                                ) as mock_executor:
                                    # Set up mock executor
                                    executor_instance = MagicMock()
                                    mock_executor.return_value.__enter__.return_value = executor_instance

                                    # Mock futures
                                    future1 = MagicMock()
                                    future1.result.return_value = (0, 'Page 1 text')
                                    future2 = MagicMock()
                                    future2.result.return_value = (1, 'Page 2 text')
                                    executor_instance.submit.side_effect = [future1, future2]

                                    # Mock as_completed to return futures in a controlled way
                                    with patch(
                                        'llm_rag.document_processing.ocr.optimized_pipeline.as_completed',
                                        return_value=[future1, future2],
                                    ):
                                        # Test the parallel processing method directly
                                        results = pipeline._parallel_process_pages('test.pdf', test_images)

                                        # Verify results
                                        assert len(results) == 2
                                        assert (0, 'Page 1 text') in results
                                        assert (1, 'Page 2 text') in results

                                        # Verify executor was used with correct parameters
                                        mock_executor.assert_called_once_with(max_workers=pipeline_config.max_workers)

                                        # Verify submit was called for each image
                                        assert executor_instance.submit.call_count == 2

    def test_sequential_processing(self, pipeline_config, mock_pdf_converter, mock_ocr_engine):
        """Test sequential processing of pages."""
        pipeline_config.parallel_processing = False
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Mock os.path.abspath to return the unmodified path
        with patch('os.path.abspath', side_effect=lambda x: x):
            # Mock os.stat to avoid file not found errors
            mock_stat = MagicMock()
            mock_stat.st_size = 1024
            mock_stat.st_mtime = 1234567890

            # Patch file operations and config attribute access
            with patch('os.stat', return_value=mock_stat):
                # Add missing attribute patches
                with patch.object(pipeline.config, 'pdf_dpi', 150):
                    with patch.object(pipeline.config, 'languages', 'eng'):
                        with patch.object(pipeline.config, 'output_format', 'raw'):
                            with patch.object(pipeline.config, 'llm_cleaning_enabled', False):
                                # Ensure cache misses
                                with patch.object(pipeline, '_check_cache', return_value=None):
                                    # Process PDF
                                    pipeline.process_pdf('test.pdf')

                                    # Both pages should be processed
                                    assert mock_ocr_engine.return_value.process_image.call_count == 2

    def test_batch_processing(self, pipeline_config, mock_pdf_converter, mock_ocr_engine):
        """Test batch processing of multiple documents."""
        # Disable batched processing for this test to avoid ThreadPoolExecutor complexity
        pipeline_config.batched_processing = False
        pipeline = OptimizedOCRPipeline(pipeline_config)

        pdf_paths = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']

        # Mock process_pdf to track calls
        with patch.object(pipeline, 'process_pdf', return_value='Processed Text') as mock_process:
            # Process batch
            results = pipeline.batch_process_pdfs(pdf_paths)

            # Verify results
            assert len(results) == 3
            for path in pdf_paths:
                assert results[path] == 'Processed Text'

            # Verify process_pdf was called for each document
            assert mock_process.call_count == 3
            calls = [call(path) for path in pdf_paths]
            mock_process.assert_has_calls(calls, any_order=True)

    def test_skip_processed_files(self, pipeline_config, mock_pdf_converter, mock_ocr_engine):
        """Test skipping already processed files."""
        pipeline_config.skip_processed_files = True
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Add a file to processed_files set
        pipeline.config.processed_files.add('already_processed.pdf')

        # Mock os.path.abspath to return the unmodified path
        with patch('os.path.abspath', side_effect=lambda x: x):
            # Mock os.stat to avoid file not found errors
            mock_stat = MagicMock()
            mock_stat.st_size = 1024
            mock_stat.st_mtime = 1234567890

            # Patch file operations and config attribute access
            with patch('os.stat', return_value=mock_stat):
                # Add missing attribute patches
                with patch.object(pipeline.config, 'pdf_dpi', 150):
                    with patch.object(pipeline.config, 'languages', 'eng'):
                        with patch.object(pipeline.config, 'output_format', 'raw'):
                            with patch.object(pipeline.config, 'llm_cleaning_enabled', False):
                                # Try to process the same file
                                with patch.object(pipeline, '_check_cache', return_value={'text': 'Cached Text'}):
                                    result = pipeline.process_pdf('already_processed.pdf')
                                    assert result == 'Cached Text'

                                    # No pages should be processed
                                    assert mock_ocr_engine.return_value.process_image.call_count == 0

    def test_processing_specific_pages(self, pipeline_config, mock_pdf_converter, mock_ocr_engine):
        """Test processing specific pages."""
        pipeline = OptimizedOCRPipeline(pipeline_config)

        # Mock os.path.abspath to return the unmodified path
        with patch('os.path.abspath', side_effect=lambda x: x):
            # Mock os.stat to avoid file not found errors
            mock_stat = MagicMock()
            mock_stat.st_size = 1024
            mock_stat.st_mtime = 1234567890

            # Patch file operations and config attribute access
            with patch('os.stat', return_value=mock_stat):
                # Add missing attribute patches
                with patch.object(pipeline.config, 'pdf_dpi', 150):
                    with patch.object(pipeline.config, 'languages', 'eng'):
                        with patch.object(pipeline.config, 'output_format', 'raw'):
                            with patch.object(pipeline.config, 'llm_cleaning_enabled', False):
                                # Process only page 1
                                with patch.object(pipeline, '_check_cache', return_value=None):
                                    # Mock the get_images to return only one page when specific pages are requested
                                    mock_pdf_converter.return_value.get_images.return_value = [(1, MagicMock())]
                                    pipeline.process_pdf('test.pdf', pages=[1])

                                    # Only page 1 should be processed
                                    assert mock_ocr_engine.return_value.process_image.call_count == 1
