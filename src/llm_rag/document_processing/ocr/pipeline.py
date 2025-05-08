"""Module for orchestrating the OCR pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from llm_rag.document_processing.ocr.llm_processor import LLMCleaner, LLMCleanerConfig
from llm_rag.document_processing.ocr.ocr_engine import TesseractOCREngine
from llm_rag.document_processing.ocr.output_formatter import JSONFormatter, MarkdownFormatter
from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverter, PDFImageConverterConfig
from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCRPipelineConfig:
    """Configuration settings for OCR pipeline.

    This class encapsulates all configuration options for the OCR pipeline,
    including PDF conversion settings and OCR engine settings.
    """

    # PDF Converter Settings
    pdf_dpi: int = 300

    # Image Preprocessing Settings
    preprocessing_enabled: bool = False
    deskew_enabled: bool = False
    threshold_enabled: bool = False
    threshold_method: str = 'adaptive'
    contrast_adjust: float = 1.0
    sharpen_enabled: bool = False
    denoise_enabled: bool = False

    # OCR Engine Settings
    tesseract_path: Optional[str] = None
    languages: Union[str, List[str]] = 'eng'
    psm: int = 3  # Page Segmentation Mode
    oem: int = 3  # OCR Engine Mode
    config_params: Optional[Dict[str, str]] = None

    # Pipeline Settings
    process_pages: Optional[List[int]] = None  # Pages to process (None = all)
    parallelize: bool = False  # Future enhancement for parallel processing

    # Output Formatting Settings
    output_format: str = 'markdown'  # "markdown", "json", or "raw"
    detect_headings: bool = True
    detect_lists: bool = True
    detect_tables: bool = False

    # LLM Cleaning Settings
    llm_cleaning_enabled: bool = False
    llm_model_name: str = 'gemma-2b'
    llm_model_backend: str = 'ollama'  # "ollama", "huggingface", or "llama_cpp"
    llm_confidence_threshold: float = 0.8
    llm_min_error_rate: float = 0.05
    llm_preserve_layout: bool = True

    # Additional settings can be added here in future phases


class OCRPipeline:
    """Pipeline for converting PDFs to text using OCR.

    This class orchestrates the complete OCR workflow:
    1. Converting PDF pages to high-quality images
    2. Processing those images with OCR to extract text
    3. Cleaning text with LLM (optional)
    4. Formatting the extracted text (optional)

    The pipeline can be configured via OCRPipelineConfig to customize
    both the PDF conversion and OCR processing steps.
    """

    def __init__(self, config: Optional[OCRPipelineConfig] = None):
        """Initialize the OCR pipeline with configuration options.

        Args:
            config: Configuration for the pipeline. If None, default
                configuration will be used.

        """
        self.config = config or OCRPipelineConfig()
        logger.info('Initializing OCR pipeline')

        # Create PDF converter config from pipeline config
        pdf_config = PDFImageConverterConfig(
            dpi=self.config.pdf_dpi,
            deskew_enabled=self.config.deskew_enabled,
            threshold_enabled=self.config.threshold_enabled,
            threshold_method=self.config.threshold_method,
            contrast_adjust=self.config.contrast_adjust,
            sharpen_enabled=self.config.sharpen_enabled,
            denoise_enabled=self.config.denoise_enabled,
        )

        # Only set preprocessing_enabled if it differs from the default
        if self.config.preprocessing_enabled:
            pdf_config.preprocessing_enabled = self.config.preprocessing_enabled

        # Initialize the PDF converter
        self.pdf_converter = PDFImageConverter(config=pdf_config)

        # Initialize the OCR engine
        self.ocr_engine = TesseractOCREngine(
            tesseract_path=self.config.tesseract_path,
            languages=self.config.languages,
            psm=self.config.psm,
            oem=self.config.oem,
            config_params=self.config.config_params,
        )

        # Initialize LLM cleaner if enabled
        self.llm_cleaner = None
        if self.config.llm_cleaning_enabled:
            llm_config = LLMCleanerConfig(
                model_name=self.config.llm_model_name,
                model_backend=self.config.llm_model_backend,
                confidence_threshold=self.config.llm_confidence_threshold,
                min_error_rate=self.config.llm_min_error_rate,
                preserve_layout=self.config.llm_preserve_layout,
            )
            self.llm_cleaner = LLMCleaner(config=llm_config)
            logger.info(
                f'LLM cleaner initialized with model: {self.config.llm_model_name} '
                f'(backend: {self.config.llm_model_backend})'
            )

        # Initialize formatters
        self.markdown_formatter = MarkdownFormatter(
            detect_headings=self.config.detect_headings,
            detect_lists=self.config.detect_lists,
            detect_tables=self.config.detect_tables,
        )
        self.json_formatter = JSONFormatter()

        logger.info('OCR pipeline initialized successfully')

    def process_pdf(self, pdf_path: Union[str, Path]) -> str:
        """Process an entire PDF document and return the OCR text.

        Args:
            pdf_path: Path to the PDF file to process.

        Returns:
            Extracted text from all processed pages combined into a single string.
            The format depends on the config.output_format setting.

        Raises:
            DocumentProcessingError: If processing fails at any stage.

        """
        try:
            pdf_path = Path(pdf_path)
            logger.info(f'Processing PDF: {pdf_path}')

            # Convert PDF pages to images
            if self.config.process_pages is not None:
                # Process specific pages
                images = []
                for page_num in self.config.process_pages:
                    image = self.pdf_converter.get_page_image(pdf_path, page_num)
                    if image:
                        images.append(image)
            else:
                # Process all pages
                images = list(self.pdf_converter.pdf_to_images(pdf_path))

            if not images:
                error_msg = f'No valid images extracted from PDF: {pdf_path}'
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg)

            logger.info(f'Extracted {len(images)} images from PDF')

            # Process images with OCR
            text_pages = self.ocr_engine.process_multiple_images(images)

            # Apply LLM cleaning if enabled
            if self.llm_cleaner:
                logger.info('Applying LLM cleaning to OCR text')
                cleaned_pages = []
                for i, page_text in enumerate(text_pages):
                    # Get confidence score if available
                    confidence_score = getattr(page_text, 'confidence', None)

                    # Get metadata for context
                    metadata = {
                        'language': self.config.languages,
                        'document_type': 'PDF',
                        'page_number': i + 1,
                    }

                    # Update language metadata if it's a list
                    if isinstance(self.config.languages, list):
                        metadata['language'] = ','.join(self.config.languages)

                    # Clean the text
                    cleaned_text = self.llm_cleaner.clean_text(page_text, confidence_score, metadata)
                    cleaned_pages.append(cleaned_text)

                text_pages = cleaned_pages
                logger.info('LLM cleaning completed')

            # Format output based on configuration
            if self.config.output_format.lower() == 'markdown':
                result = self.markdown_formatter.format_document(text_pages)
            elif self.config.output_format.lower() == 'json':
                result = self.json_formatter.format_document(text_pages)
            else:  # raw text
                # Simple joining of pages with double newlines for raw text output
                result = '\n\n'.join(text_pages)

            logger.info(f'OCR completed for {pdf_path}, extracted {len(result)} characters')

            return result

        except Exception as e:
            error_msg = f'OCR pipeline failed for PDF {pdf_path}: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def process_pdf_pages(self, pdf_path: Union[str, Path], page_numbers: List[int]) -> Dict[int, str]:
        """Process specific pages of a PDF document.

        Args:
            pdf_path: Path to the PDF file to process.
            page_numbers: List of page numbers to process (0-indexed).

        Returns:
            Dictionary mapping page numbers to their extracted text.

        Raises:
            DocumentProcessingError: If processing fails at any stage.

        """
        try:
            pdf_path = Path(pdf_path)
            logger.info(f'Processing specific pages from PDF: {pdf_path}')

            results = {}
            for page_num in page_numbers:
                logger.info(f'Processing page {page_num}')

                # Convert PDF page to image
                image = self.pdf_converter.get_page_image(pdf_path, page_num)

                if image:
                    # Process image with OCR
                    text = self.ocr_engine.process_image(image)

                    # Apply LLM cleaning if enabled
                    if self.llm_cleaner:
                        # Get confidence score if available
                        confidence_score = getattr(text, 'confidence', None)

                        # Get metadata for context
                        metadata = {
                            'language': self.config.languages,
                            'document_type': 'PDF',
                            'page_number': page_num,
                        }

                        # Update language metadata if it's a list
                        if isinstance(self.config.languages, list):
                            metadata['language'] = ','.join(self.config.languages)

                        # Clean the text
                        text = self.llm_cleaner.clean_text(text, confidence_score, metadata)

                    results[page_num] = text
                    logger.debug(f'Completed OCR for page {page_num}')
                else:
                    logger.warning(f'Could not extract image for page {page_num}')

            if not results:
                error_msg = f'No text extracted from specified pages in PDF: {pdf_path}'
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg)

            logger.info(f'OCR completed for {len(results)} pages from {pdf_path}')
            return results

        except Exception as e:
            error_msg = f'OCR pipeline failed for PDF {pdf_path}: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def save_to_markdown(self, pdf_path: Union[str, Path], output_dir: Union[str, Path, None] = None) -> Path:
        """Process a PDF document and save the results to a Markdown file.

        Args:
            pdf_path: Path to the PDF file to process.
            output_dir: Directory where to save the output file. If None,
                uses the same directory as the input file.

        Returns:
            Path to the created Markdown file.

        Raises:
            DocumentProcessingError: If processing fails at any stage.

        """
        pdf_path = Path(pdf_path)

        # Save the original output format
        original_format = self.config.output_format

        try:
            # Force markdown format for this operation
            self.config.output_format = 'markdown'

            # Process the PDF
            markdown_content = self.process_pdf(pdf_path)

            # Determine output path
            if output_dir is None:
                output_dir = pdf_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)

            output_file = output_dir / f'{pdf_path.stem}.md'

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f'Saved OCR output to {output_file}')

            return output_file

        except Exception as e:
            error_msg = f'Failed to save OCR results to Markdown: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

        finally:
            # Restore original output format
            self.config.output_format = original_format

    def process_and_save(
        self, pdf_path: Union[str, Path], output_dir: Union[str, Path, None] = None, format: str = 'markdown'
    ) -> Path:
        """Process a PDF document and save the results to a file.

        Args:
            pdf_path: Path to the PDF file to process.
            output_dir: Directory where to save the output file. If None,
                uses the same directory as the input file.
            format: Output format - "markdown", "json", or "txt" for raw text.

        Returns:
            Path to the created output file.

        Raises:
            DocumentProcessingError: If processing fails at any stage.
            ValueError: If an unsupported output format is specified.

        """
        pdf_path = Path(pdf_path)

        # Save the original output format
        original_format = self.config.output_format

        try:
            # Set output format
            format = format.lower()
            if format not in ['markdown', 'json', 'txt']:
                raise ValueError(f'Unsupported output format: {format}')

            self.config.output_format = format if format != 'txt' else 'raw'

            # Process the PDF
            content = self.process_pdf(pdf_path)

            # Determine output path
            if output_dir is None:
                output_dir = pdf_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)

            # Set file extension based on format
            if format == 'markdown':
                extension = '.md'
            elif format == 'json':
                extension = '.json'
            else:  # txt
                extension = '.txt'

            output_file = output_dir / f'{pdf_path.stem}{extension}'

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f'Saved OCR output to {output_file}')

            return output_file

        except Exception as e:
            error_msg = f'Failed to save OCR results: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

        finally:
            # Restore original output format
            self.config.output_format = original_format
