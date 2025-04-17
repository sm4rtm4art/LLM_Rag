"""Performance benchmark tests for the OptimizedOCRPipeline.

This module contains tests that measure the performance improvements from various
optimization features:
1. Parallel processing
2. Caching
3. Batch processing
4. Incremental processing (skipping files)

These tests compare performance with and without optimizations and provide
metrics on time and memory usage.
"""

import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from llm_rag.document_processing.ocr.optimized_pipeline import OptimizedOCRConfig, OptimizedOCRPipeline


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_pdf_path():
    """Return a path to a sample PDF file for testing."""
    # This is a mock path, adjust as needed for actual testing with real PDFs
    return "tests/document_processing/ocr/data/sample.pdf"


@pytest.fixture
def mock_images():
    """Create mock images for testing."""
    return [(i, Image.new("RGB", (100, 100), color=(255, 255, 255))) for i in range(5)]


@pytest.fixture
def mock_pdf_converter():
    """Mock PDFImageConverter to return predefined images."""
    with patch("llm_rag.document_processing.ocr.optimized_pipeline.PDFImageConverter") as mock:
        mock_instance = MagicMock()
        mock_instance.get_images.return_value = [
            (0, Image.new("RGB", (100, 100))),
            (1, Image.new("RGB", (100, 100))),
        ]
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_ocr_engine():
    """Mock OCR engine to return predefined text."""
    with patch("llm_rag.document_processing.ocr.pipeline.OCREngine") as mock:
        mock_instance = MagicMock()
        mock_instance.process_image.side_effect = lambda image: f"Page {image[0] + 1} text"
        mock.return_value = mock_instance
        yield mock


class TestOptimizedPipelinePerformance:
    """Performance benchmark tests for OptimizedOCRPipeline."""

    def test_parallel_vs_sequential_performance(self, temp_cache_dir, mock_pdf_converter, mock_ocr_engine, mock_images):
        """Compare performance of parallel vs sequential processing."""

        # Configure OCR engine to take some time to simulate real processing
        def delayed_process(image):
            time.sleep(0.05)  # Simulate processing time
            return f"Page {image[0] + 1} text"

        mock_ocr_engine.return_value.process_image = delayed_process

        # Create pipelines with different configurations
        parallel_config = OptimizedOCRConfig(
            parallel_processing=True,
            max_workers=4,
            use_cache=False,  # Disable cache for fair comparison
            cache_dir=temp_cache_dir,
            pdf_renderer_dpi=150,
        )

        sequential_config = OptimizedOCRConfig(
            parallel_processing=False,  # Use sequential processing
            use_cache=False,  # Disable cache for fair comparison
            cache_dir=temp_cache_dir,
            pdf_renderer_dpi=150,
        )

        # Test with mock converter that returns multiple pages
        mock_pdf_converter.return_value.get_images.return_value = mock_images

        # Measure parallel processing time
        parallel_pipeline = OptimizedOCRPipeline(parallel_config)
        with patch.object(parallel_pipeline, "_check_cache", return_value=None):
            start_time = time.time()
            parallel_pipeline._parallel_process_pages("test.pdf", mock_images)
            parallel_time = time.time() - start_time

        # Measure sequential processing time
        sequential_pipeline = OptimizedOCRPipeline(sequential_config)
        with patch.object(sequential_pipeline, "_check_cache", return_value=None):
            start_time = time.time()
            sequential_pipeline._sequential_process_pages("test.pdf", mock_images)
            sequential_time = time.time() - start_time

        # Assert parallel is faster than sequential
        assert parallel_time < sequential_time

        # Calculate speedup factor
        speedup = sequential_time / parallel_time
        print(f"Parallel speedup factor: {speedup:.2f}x")

        # With 4 workers and 5 pages, we should see a speedup close to 4x
        # but less due to overhead
        assert 1.5 < speedup < 5.0

    def test_caching_performance(self, temp_cache_dir, mock_pdf_converter, mock_ocr_engine):
        """Test performance improvement from caching."""

        # Configure OCR engine to take some time to simulate real processing
        def delayed_process(image):
            time.sleep(0.1)  # Simulate processing time
            return f"Page {image[0] + 1} text"

        mock_ocr_engine.return_value.process_image = delayed_process

        config = OptimizedOCRConfig(
            parallel_processing=False,  # Disable parallel for consistent measurement
            use_cache=True,
            cache_dir=temp_cache_dir,
            pdf_renderer_dpi=150,
        )

        pipeline = OptimizedOCRPipeline(config)

        # First run without cache
        start_time = time.time()
        pipeline.process_pdf("test.pdf")
        first_run_time = time.time() - start_time

        # Second run with cache
        start_time = time.time()
        pipeline.process_pdf("test.pdf")
        second_run_time = time.time() - start_time

        # Cache should make second run much faster
        assert second_run_time < first_run_time

        # Calculate speedup factor
        speedup = first_run_time / second_run_time
        print(f"Cache speedup factor: {speedup:.2f}x")

        # Cache access should be at least 5x faster than OCR processing
        assert speedup > 5.0

    def test_batch_processing_performance(self, temp_cache_dir, mock_pdf_converter, mock_ocr_engine):
        """Test performance of batch processing vs sequential."""

        # Mock process_pdf to simulate processing time
        def delayed_process(path):
            time.sleep(0.1)  # Simulate processing time
            return f"Processed {path}"

        # Create test PDF paths
        pdf_paths = [f"test{i}.pdf" for i in range(10)]

        # Test sequential processing (batched_processing=False)
        sequential_config = OptimizedOCRConfig(
            batched_processing=False,
            use_cache=False,
            cache_dir=temp_cache_dir,
        )
        sequential_pipeline = OptimizedOCRPipeline(sequential_config)

        with patch.object(sequential_pipeline, "process_pdf", side_effect=delayed_process):
            start_time = time.time()
            sequential_pipeline.batch_process_pdfs(pdf_paths)
            sequential_time = time.time() - start_time

        # Test batch processing (batched_processing=True)
        batch_config = OptimizedOCRConfig(
            batched_processing=True,
            batch_size=5,
            max_workers=5,
            use_cache=False,
            cache_dir=temp_cache_dir,
        )
        batch_pipeline = OptimizedOCRPipeline(batch_config)

        with patch.object(batch_pipeline, "process_pdf", side_effect=delayed_process):
            start_time = time.time()
            batch_pipeline.batch_process_pdfs(pdf_paths)
            batch_time = time.time() - start_time

        # Batch processing should be faster
        assert batch_time < sequential_time

        # Calculate speedup factor
        speedup = sequential_time / batch_time
        print(f"Batch processing speedup factor: {speedup:.2f}x")

        # With 5 workers, should see a speedup close to 5x
        assert 1.5 < speedup < 6.0

    def test_skip_processed_files(self, temp_cache_dir, mock_pdf_converter, mock_ocr_engine):
        """Test skip_processed_files functionality and its performance impact."""
        config = OptimizedOCRConfig(
            skip_processed_files=True,
            use_cache=True,
            cache_dir=temp_cache_dir,
        )

        pipeline = OptimizedOCRPipeline(config)

        # Mock _check_cache to return None first, then cached data
        mock_check_cache = MagicMock()
        mock_check_cache.side_effect = [None, {"text": "Cached result"}]

        with patch.object(pipeline, "_check_cache", mock_check_cache):
            # First process a file
            pipeline.process_pdf("test.pdf")

            # Now it should be in processed_files
            assert "test.pdf" in pipeline.config.processed_files

            # Process the same file again
            result2 = pipeline.process_pdf("test.pdf")

            # Should return cached result without processing
            assert result2 == "Cached result"

            # Verify the _generate_cache_key was called twice
            # Once for the initial check, once for the skip check
            assert pipeline._generate_cache_key.call_count == 2

    def test_memory_usage_parallel_processing(self, temp_cache_dir, mock_pdf_converter, mock_ocr_engine):
        """Test memory usage during parallel processing."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
        except ImportError:
            pytest.skip("psutil not installed, skipping memory usage test")

        # Create large mock images to stress memory
        large_images = [(i, Image.new("RGB", (1000, 1000), color=(255, 255, 255))) for i in range(10)]

        mock_pdf_converter.return_value.get_images.return_value = large_images

        # Record baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Test with different worker counts
        memory_usage = {}

        for workers in [1, 2, 4]:
            config = OptimizedOCRConfig(
                parallel_processing=True,
                max_workers=workers,
                use_cache=False,
                cache_dir=temp_cache_dir,
            )

            pipeline = OptimizedOCRPipeline(config)

            with patch.object(pipeline, "_check_cache", return_value=None):
                # Clear any previous allocated memory
                import gc

                gc.collect()

                # Process PDF and measure peak memory
                pipeline.process_pdf("test.pdf")
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage[workers] = current_memory - baseline_memory

        # Print memory usage for different worker counts
        for workers, usage in memory_usage.items():
            print(f"Memory usage with {workers} workers: {usage:.2f} MB")

        # Memory usage should increase with more workers, but not linearly
        assert memory_usage[4] > memory_usage[1]
        # But should be less than 4x the memory of 1 worker due to shared resources
        assert memory_usage[4] < 4 * memory_usage[1]

    def test_end_to_end_optimized_pipeline(self, temp_cache_dir, mock_pdf_converter, mock_ocr_engine):
        """Test full pipeline with all optimizations enabled."""
        # Create test PDF paths
        pdf_paths = [f"test{i}.pdf" for i in range(5)]

        # Configure fully optimized pipeline
        optimized_config = OptimizedOCRConfig(
            parallel_processing=True,
            max_workers=4,
            use_cache=True,
            cache_dir=temp_cache_dir,
            batched_processing=True,
            batch_size=3,
            skip_processed_files=True,
            pdf_renderer_dpi=150,
        )

        pipeline = OptimizedOCRPipeline(optimized_config)

        # Process the batch
        with patch.object(pipeline, "_check_cache", return_value=None):
            start_time = time.time()
            results = pipeline.batch_process_pdfs(pdf_paths)
            first_run_time = time.time() - start_time

            # All files should be processed
            assert len(results) == 5

            # All files should be marked as processed
            for path in pdf_paths:
                assert path in pipeline.config.processed_files

            # Process the same batch again
            start_time = time.time()
            results = pipeline.batch_process_pdfs(pdf_paths)
            second_run_time = time.time() - start_time

        # Second run should be much faster due to caching and skipping
        assert second_run_time < first_run_time / 5

        print(f"First run time: {first_run_time:.2f}s")
        print(f"Second run time: {second_run_time:.2f}s")
        print(f"Speedup factor: {first_run_time / second_run_time:.2f}x")
