"""Optimized OCR pipeline with parallel processing and caching capabilities.

This module enhances the base OCR pipeline with performance optimizations such as:
1. Parallel processing of pages for multi-page documents
2. Caching of OCR results to avoid redundant processing
3. Batch processing capabilities for multiple documents
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverter
from llm_rag.document_processing.ocr.pipeline import OCRPipeline, OCRPipelineConfig
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizedOCRConfig(OCRPipelineConfig):
    """Configuration for optimized OCR processing.

    Extends the base OCRPipelineConfig with additional options for performance.

    Attributes:
        parallel_processing: Whether to process pages in parallel
        max_workers: Maximum number of worker threads for parallel processing
        use_cache: Whether to cache OCR results
        cache_dir: Directory to store cache files
        cache_ttl: Time-to-live for cache entries in seconds (default: 1 week)
        force_reprocess: Force reprocessing even if cache exists

    """

    parallel_processing: bool = True
    max_workers: int = 4
    use_cache: bool = True
    cache_dir: str = ".ocr_cache"
    cache_ttl: int = 7 * 24 * 60 * 60  # 1 week in seconds
    force_reprocess: bool = False
    batched_processing: bool = False
    batch_size: int = 5
    skip_processed_files: bool = True
    processed_files: Set[str] = field(default_factory=set)


class OptimizedOCRPipeline(OCRPipeline):
    """Enhanced OCR pipeline with performance optimizations.

    This class extends the base OCRPipeline with:
    - Parallel processing of pages
    - Result caching
    - Batch processing for multiple documents
    """

    def __init__(self, config: Optional[OptimizedOCRConfig] = None):
        """Initialize the optimized OCR pipeline.

        Args:
            config: Configuration for the OCR pipeline

        """
        self.config = config or OptimizedOCRConfig()
        super().__init__(self.config)

        # Initialize cache directory if needed
        if self.config.use_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            logger.info(f"Using OCR cache directory: {self.config.cache_dir}")

    def _generate_cache_key(self, file_path: str, page_num: Optional[int] = None) -> str:
        """Generate a unique cache key for a file or page.

        Args:
            file_path: Path to the PDF file
            page_num: Optional page number (if None, applies to whole document)

        Returns:
            A unique hash string to use as cache key

        """
        file_stat = os.stat(file_path)
        file_info = {
            "path": str(file_path),
            "size": file_stat.st_size,
            "mtime": file_stat.st_mtime,
            "page": page_num,
            "config": {
                "dpi": self.config.pdf_dpi,
                "ocr_lang": self.config.ocr_language,
                "output_format": self.config.output_format,
                "use_llm_cleaner": self.config.use_llm_cleaner,
            },
        }

        key_str = json.dumps(file_info, sort_keys=True)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry.

        Args:
            cache_key: The cache key

        Returns:
            Path to the cache file

        """
        return Path(self.config.cache_dir) / f"{cache_key}.json"

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if a valid cache entry exists.

        Args:
            cache_key: The cache key

        Returns:
            Cached data if found and valid, None otherwise

        """
        if not self.config.use_cache or self.config.force_reprocess:
            return None

        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            # Read cache file
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Check if cache is expired
            timestamp = cache_data.get("timestamp", 0)
            current_time = time.time()
            if current_time - timestamp > self.config.cache_ttl:
                logger.info(f"Cache expired for key {cache_key}")
                return None

            logger.info(f"Using cached OCR result for key {cache_key}")
            return cache_data
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}")
            return None

    def _write_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Write data to cache.

        Args:
            cache_key: The cache key
            data: Data to cache

        """
        if not self.config.use_cache:
            return

        # Add timestamp to cache data
        cache_data = {"timestamp": time.time(), **data}

        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            logger.debug(f"Wrote cache file: {cache_path}")
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")

    def _process_page_with_cache(self, pdf_path: str, image, page_num: int, total_pages: int) -> Tuple[int, str]:
        """Process a single page with caching support.

        Args:
            pdf_path: Path to the PDF file
            image: PIL image of the page
            page_num: Page number (0-based)
            total_pages: Total number of pages

        Returns:
            Tuple of (page_num, ocr_text)

        """
        # Check cache first
        cache_key = self._generate_cache_key(pdf_path, page_num)
        cache_data = self._check_cache(cache_key)

        if cache_data and "text" in cache_data:
            logger.info(f"Using cached OCR for page {page_num + 1}/{total_pages}")
            return page_num, cache_data["text"]

        # Process the page using the base implementation
        logger.info(f"Processing page {page_num + 1}/{total_pages}")
        ocr_text = self.ocr_engine.process_image(image)

        # Clean text with LLM if configured
        if self.config.use_llm_cleaner and self.llm_cleaner:
            confidence = getattr(self.ocr_engine, "last_confidence", None)
            metadata = {
                "page_number": page_num + 1,
                "total_pages": total_pages,
                "document_type": "pdf",
            }

            # If language detection is enabled, it will be performed in the cleaner
            ocr_text = self.llm_cleaner.clean_text(ocr_text, confidence_score=confidence, metadata=metadata)

        # Write to cache
        self._write_cache(cache_key, {"text": ocr_text})

        return page_num, ocr_text

    def process_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> str:
        """Process a PDF document with OCR, with caching and parallel processing.

        Args:
            pdf_path: Path to the PDF file
            pages: Optional list of specific pages to process (0-based)

        Returns:
            OCR text for the entire document or specified pages

        """
        # Convert relative paths to absolute
        pdf_path = os.path.abspath(pdf_path)

        # Track processed files for batch processing
        if self.config.skip_processed_files:
            if pdf_path in self.config.processed_files:
                logger.info(f"Skipping already processed file: {pdf_path}")
                # Try to load from document-level cache if available
                cache_key = self._generate_cache_key(pdf_path)
                cache_data = self._check_cache(cache_key)
                if cache_data and "text" in cache_data:
                    return cache_data["text"]

        # Check if we have a document-level cache
        if pages is None:
            cache_key = self._generate_cache_key(pdf_path)
            cache_data = self._check_cache(cache_key)
            if cache_data and "text" in cache_data:
                logger.info(f"Using cached OCR for entire document: {pdf_path}")
                return cache_data["text"]

        # Load PDF converter and get images
        converter = PDFImageConverter(pdf_path, self.config.pdf_dpi)
        page_images = list(converter.get_images(pages))
        total_pages = len(page_images)

        if total_pages == 0:
            logger.warning(f"No pages found in PDF: {pdf_path}")
            return ""

        # Process pages based on configuration
        if self.config.parallel_processing and total_pages > 1:
            # Process pages in parallel
            results = self._parallel_process_pages(pdf_path, page_images)
        else:
            # Process pages sequentially
            results = self._sequential_process_pages(pdf_path, page_images)

        # Combine results
        all_text = self._combine_page_results(results, total_pages)

        # Format the output based on configuration
        formatted_text = self._format_output(all_text)

        # Cache the complete document result
        if pages is None:
            cache_key = self._generate_cache_key(pdf_path)
            self._write_cache(cache_key, {"text": formatted_text})

        # Mark as processed
        self.config.processed_files.add(pdf_path)

        return formatted_text

    def _parallel_process_pages(self, pdf_path: str, page_images: List[Tuple[int, Any]]) -> List[Tuple[int, str]]:
        """Process pages in parallel using a thread pool.

        Args:
            pdf_path: Path to the PDF file
            page_images: List of (page_num, image) tuples

        Returns:
            List of (page_num, ocr_text) tuples

        """
        total_pages = len(page_images)
        results = []

        logger.info(f"Processing {total_pages} pages in parallel with {self.config.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(self._process_page_with_cache, pdf_path, image, page_num, total_pages): page_num
                for page_num, image in page_images
            }

            # Process results as they complete
            for future in as_completed(future_to_page):
                try:
                    page_num, text = future.result()
                    results.append((page_num, text))
                except Exception as e:
                    page = future_to_page[future]
                    logger.error(f"Error processing page {page + 1}: {str(e)}")

        return results

    def _sequential_process_pages(self, pdf_path: str, page_images: List[Tuple[int, Any]]) -> List[Tuple[int, str]]:
        """Process pages sequentially.

        Args:
            pdf_path: Path to the PDF file
            page_images: List of (page_num, image) tuples

        Returns:
            List of (page_num, ocr_text) tuples

        """
        total_pages = len(page_images)
        results = []

        logger.info(f"Processing {total_pages} pages sequentially")

        for page_num, image in page_images:
            try:
                page_num, text = self._process_page_with_cache(pdf_path, image, page_num, total_pages)
                results.append((page_num, text))
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {str(e)}")

        return results

    def _combine_page_results(self, results: List[Tuple[int, str]], total_pages: int) -> List[str]:
        """Combine page results in the correct order.

        Args:
            results: List of (page_num, ocr_text) tuples
            total_pages: Total number of pages

        Returns:
            List of text strings in page order

        """
        # Sort results by page number
        sorted_results = sorted(results, key=lambda x: x[0])
        return [text for _, text in sorted_results]

    def batch_process_pdfs(self, pdf_paths: List[str]) -> Dict[str, str]:
        """Process multiple PDF documents in batches.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            Dictionary mapping file paths to OCR results

        """
        if not self.config.batched_processing:
            # Process sequentially if batching is disabled
            logger.info(f"Processing {len(pdf_paths)} documents sequentially")
            return {path: self.process_pdf(path) for path in pdf_paths}

        logger.info(f"Batch processing {len(pdf_paths)} documents with batch size {self.config.batch_size}")

        results = {}
        batches = [pdf_paths[i : i + self.config.batch_size] for i in range(0, len(pdf_paths), self.config.batch_size)]

        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)}")

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_path = {executor.submit(self.process_pdf, path): path for path in batch}

                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[path] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing {path}: {str(e)}")
                        results[path] = f"ERROR: {str(e)}"

        return results
