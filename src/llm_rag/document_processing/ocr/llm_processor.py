"""LLM-based OCR text cleaning and enhancement.

This module provides functionality to improve OCR text quality using Language Models.
It applies sophisticated text correction, formatting enhancement, and can operate
asynchronously to improve processing time.
"""

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

from llm_rag.models.factory import ModelFactory
from llm_rag.utils.errors import ModelError
from llm_rag.utils.logging import get_logger

# Import for language detection
try:
    import langdetect

    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

logger = get_logger(__name__)


@dataclass
class LLMCleanerConfig:
    """Configuration for LLM-based OCR text cleaning.

    Attributes:
        model_name: Name of the model to use (default: "gemma-2b" for speed)
        model_backend: Backend to use for the model (default: "ollama")
        confidence_threshold: Only apply LLM cleaning when OCR confidence is below
            this threshold (default: 0.8)
        min_error_rate: Only apply LLM cleaning when character error rate estimation
            is above this threshold (default: 0.05 = 5%)
        async_processing: Whether to process in background threads (default: False)
        max_workers: Maximum number of worker threads for async processing (default: 4)
        max_chunk_size: Maximum characters to process in a single LLM call (default: 2000)
        max_retry_attempts: Maximum number of retries for failed LLM calls (default: 2)
        preserve_layout: Whether to preserve document layout like paragraphs and lists
            (default: True)
        timeout: Timeout in seconds for LLM calls (default: 30)
        detect_language: Whether to automatically detect the document language (default: True)
        preserve_language: Whether to preserve the original document language (default: True)
        translate_to_language: Optional target language for translation (default: None)
        language_models: Mapping of language codes to preferred models for that language
            (default: empty dict, will use default model)

    """

    model_name: str = 'gemma-2b'
    model_backend: str = 'ollama'
    confidence_threshold: float = 0.8
    min_error_rate: float = 0.05
    async_processing: bool = False
    max_workers: int = 4
    max_chunk_size: int = 2000
    max_retry_attempts: int = 2
    preserve_layout: bool = True
    timeout: int = 30
    detect_language: bool = True
    preserve_language: bool = True
    translate_to_language: Optional[str] = None
    language_models: Dict[str, str] = None

    def __post_init__(self):
        """Initialize any default values that can't be set directly."""
        if self.language_models is None:
            self.language_models = {}

        # If translation is requested, disable language preservation
        if self.translate_to_language:
            self.preserve_language = False


class LLMCleaner:
    """LLM-based OCR text cleaner.

    This class uses Language Models to improve OCR text quality by correcting errors,
    fixing formatting issues, and enhancing readability.
    """

    def __init__(self, config: Optional[LLMCleanerConfig] = None):
        """Initialize the LLM cleaner.

        Args:
            config: Configuration for the LLM cleaner. If None, default configuration
                will be used.

        """
        self.config = config or LLMCleanerConfig()
        logger.info(f'Initializing LLMCleaner with model: {self.config.model_name}')

        # Set up thread pool for async processing if enabled
        self._executor = None
        if self.config.async_processing:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            logger.info(f'Async processing enabled with {self.config.max_workers} workers')

        # Lazy initialization of model - only load when needed
        self._model = None
        self._language_specific_models = {}

    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                # Create the model using ModelFactory.create_model
                backend = self.config.model_backend
                self._model = ModelFactory.create_model(
                    model_path_or_name=self.config.model_name,
                    backend=backend,
                    # Add any additional parameters as needed
                    max_tokens=512,
                    temperature=0.7,
                )
                logger.info(f'Loaded model: {self.config.model_name} using backend: {backend}')
            except Exception as e:
                logger.error(f'Failed to load model {self.config.model_name}: {e}')
                raise ModelError(f'Failed to load model: {str(e)}') from e
        return self._model

    def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the input text.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code (e.g., 'en', 'de', 'fr') or None if detection fails

        """
        if not self.config.detect_language:
            return None

        if not HAS_LANGDETECT:
            logger.warning(
                "Language detection requested but langdetect not installed. Install with 'pip install langdetect'"
            )
            return None

        try:
            # Use a larger sample for better accuracy
            sample_text = text[: min(5000, len(text))]
            lang_code = langdetect.detect(sample_text)
            logger.info(f'Detected language: {lang_code}')
            return lang_code
        except Exception as e:
            logger.warning(f'Language detection failed: {e}')
            return None

    def get_model_for_language(self, language_code: Optional[str]) -> Any:
        """Get appropriate model for the specified language.

        Args:
            language_code: ISO 639-1 language code

        Returns:
            Language model appropriate for the language

        """
        if not language_code or language_code not in self.config.language_models:
            return self.model

        model_name = self.config.language_models.get(language_code)
        if not model_name:
            return self.model

        # Check if we've already created this model
        if model_name in self._language_specific_models:
            return self._language_specific_models[model_name]

        # Create a new model for this language
        try:
            model = ModelFactory.create_model(
                model_path_or_name=model_name,
                backend=self.config.model_backend,
                max_tokens=512,
                temperature=0.7,
            )
            self._language_specific_models[model_name] = model
            logger.info(f'Created language-specific model for {language_code}: {model_name}')
            return model
        except Exception as e:
            logger.error(f'Failed to load language model {model_name}: {e}')
            # Fall back to default model
            return self.model

    def _generate_text(self, prompt: str, max_tokens: int = None, timeout: int = None) -> str:
        """Generate text from the model, handling different model API styles.

        This wrapper method handles the different ways models might expect prompts
        (single string vs list of strings) and adapts accordingly.

        Args:
            prompt: The prompt text
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout in seconds

        Returns:
            Generated text

        """
        max_tokens = max_tokens or int(len(prompt) * 1.2)
        timeout = timeout or self.config.timeout

        try:
            # Special handling for Ollama models
            if self.config.model_backend == 'ollama':
                # For LangChain's Ollama implementation, use __call__ directly
                # This avoids issues with the implementation details of different LangChain versions
                try:
                    # Most direct way to call the model
                    if callable(self.model):
                        return str(self.model(prompt))
                except Exception as e:
                    logger.warning(f'Direct model call failed: {e}. Using fallback method.')
                    # If all else fails, let's try a couple of options
                    try:
                        # Try the simple invoke method if available
                        if hasattr(self.model, 'invoke'):
                            return str(self.model.invoke(prompt))
                    except Exception as e:
                        logger.warning(f'Model invoke failed: {e}')
                        # Return original text to avoid pipeline failure
                        logger.warning('All Ollama API attempts failed, returning original text')
                        return prompt

            # Standard LLM approach for non-Ollama models
            if hasattr(self.model, 'generate'):
                try:
                    return str(self.model.generate(prompt, max_tokens=max_tokens, timeout=timeout))
                except Exception as e:
                    logger.warning(f'Model generate failed: {e}. Trying alternative methods.')

            # Try invoke method
            if hasattr(self.model, 'invoke'):
                try:
                    return str(self.model.invoke(prompt))
                except Exception as e:
                    logger.warning(f'Model invoke failed: {e}. Trying alternative methods.')

            # Last resort, try direct callable
            if callable(self.model):
                return str(self.model(prompt))

            # If nothing worked, at least return something
            logger.warning('All model API approaches failed, returning original prompt')
            return prompt

        except Exception as e:
            logger.error(f'Error generating text: {e}')
            raise

    def clean_text(
        self, ocr_text: str, confidence_score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Clean OCR text using the LLM.

        Args:
            ocr_text: The raw OCR text to clean
            confidence_score: Optional confidence score from OCR (0-1)
            metadata: Optional metadata about the document

        Returns:
            Cleaned text

        """
        # Skip cleaning if confidence is high enough
        if confidence_score is not None and confidence_score > self.config.confidence_threshold:
            logger.info(f'Skipping LLM cleaning - high confidence score: {confidence_score:.2f}')
            return ocr_text

        # Check for estimated error rate and skip if too low
        error_rate = self._estimate_error_rate(ocr_text)
        if error_rate < self.config.min_error_rate:
            logger.info(f'Skipping LLM cleaning - low estimated error rate: {error_rate:.2f}')
            return ocr_text

        logger.info(f'Cleaning OCR text with LLM (estimated error rate: {error_rate:.2f})')

        # Detect language if configured to do so and not provided in metadata
        detected_language = None
        if self.config.detect_language and (not metadata or 'language' not in metadata):
            detected_language = self.detect_language(ocr_text)
            if metadata is None:
                metadata = {}
            if detected_language:
                metadata['language'] = detected_language
                logger.info(f'Detected document language: {detected_language}')

        # If text is too long, split into chunks
        if len(ocr_text) > self.config.max_chunk_size:
            return self._process_long_text(ocr_text, metadata)

        # Process the text with LLM
        return self._process_with_llm(ocr_text, metadata)

    async def clean_text_async(
        self, ocr_text: str, confidence_score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Clean OCR text asynchronously.

        Args:
            ocr_text: The raw OCR text to clean
            confidence_score: Optional confidence score from OCR (0-1)
            metadata: Optional metadata about the document

        Returns:
            Cleaned text

        """
        if not self.config.async_processing:
            logger.warning('Async processing not enabled, processing synchronously')
            return self.clean_text(ocr_text, confidence_score, metadata)

        # Create a partial function with all arguments
        func = partial(self.clean_text, ocr_text, confidence_score, metadata)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func)

    def _estimate_error_rate(self, text: str) -> float:
        """Estimate error rate in OCR text without ground truth.

        Uses heuristics to detect likely OCR errors.

        Args:
            text: OCR text to analyze

        Returns:
            Estimated error rate (0-1)

        """
        # Count words and check what percentage might be misspelled
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        # Simple heuristics to detect likely OCR errors - these are common patterns
        error_patterns = [
            r'\bI[^a-zA-Z\s]',  # I followed by punctuation (likely 1)
            r'[a-z][A-Z]',  # Mid-word capitals (likely OCR error)
            r'[0-9][a-zA-Z]|[a-zA-Z][0-9]',  # Letters adjacent to numbers
            r'vv|rn',  # common substitutions
            r'l1|11',  # l → 1 confusion
            r'([^\w\s])\1{2,}',  # Repeated punctuation
        ]

        potential_errors = 0
        for pattern in error_patterns:
            potential_errors += len(re.findall(pattern, text))

        # Check for words with unusual character combos that might be OCR errors
        unusual_character_patterns = [
            r'\b\w*[^a-zA-Z0-9\s.,;:!?-]\w*\b',  # Words with unusual characters
        ]

        for pattern in unusual_character_patterns:
            potential_errors += len(re.findall(pattern, text))

        # Normalize by text length
        error_rate = min(1.0, potential_errors / max(1, len(words)))

        return error_rate

    def _create_cleaning_prompt(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a prompt for the LLM to clean OCR text.

        Args:
            text: OCR text to clean
            metadata: Optional metadata about the document

        Returns:
            Prompt for the LLM

        """
        layout_instruction = (
            'Preserve the original document layout including paragraphs, lists, and headings.'
            if self.config.preserve_layout
            else ''
        )

        # Add document-specific context if available
        context = ''
        language_instruction = ''

        # Determine language handling
        document_language = None
        if metadata and metadata.get('language'):
            document_language = metadata.get('language')
            context += f'This document is in {document_language}. '

        # Language preservation or translation instruction
        if self.config.translate_to_language:
            target_lang = self.config.translate_to_language
            language_instruction = f'Translate the text to {target_lang} while fixing OCR errors. '
            if document_language:
                language_instruction = (
                    f'Translate the text from {document_language} to {target_lang} while fixing OCR errors. '
                )
        elif self.config.preserve_language and document_language:
            language_instruction = (
                f'Keep the text in the original {document_language} language - DO NOT translate to English. '
            )
        else:
            # Default behavior - preserve language but don't specify it explicitly
            language_instruction = 'Preserve the original language of the document. DO NOT translate the content. '

        # Add document type if available
        if metadata and metadata.get('document_type'):
            context += f'This is a {metadata["document_type"]}. '

        # Format prompt with proper line breaks to stay within line limits
        prompt = (
            f'Fix OCR errors in the following text. {language_instruction}'
            f'Correct spelling, punctuation, and formatting issues while preserving the '
            f'original meaning and technical terms. '
            f'{layout_instruction} {context}\n\n'
            f'OCR TEXT:\n{text}\n\n'
            f'CORRECTED TEXT:\n'
        )
        return prompt

    def _process_with_llm(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process text with the LLM.

        Args:
            text: Text to process
            metadata: Optional metadata

        Returns:
            Processed text

        """
        prompt = self._create_cleaning_prompt(text, metadata)

        # Select the appropriate model based on language
        current_model = self.model
        if metadata and 'language' in metadata and self.config.language_models:
            language = metadata['language']
            current_model = self.get_model_for_language(language)

        # Try multiple times in case of model errors
        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                start_time = time.time()

                # Generate text using the selected model or our default model
                if current_model != self.model:
                    # Use the language-specific model
                    if hasattr(current_model, 'generate'):
                        response = str(
                            current_model.generate(prompt, max_tokens=int(len(text) * 1.2), timeout=self.config.timeout)
                        )
                    elif hasattr(current_model, 'invoke'):
                        response = str(current_model.invoke(prompt))
                    elif callable(current_model):
                        response = str(current_model(prompt))
                    else:
                        # Fallback to default model
                        logger.warning('Language-specific model API not supported, falling back to default model')
                        response = self._generate_text(
                            prompt, max_tokens=int(len(text) * 1.2), timeout=self.config.timeout
                        )
                else:
                    # Use our wrapper method that handles different model APIs
                    response = self._generate_text(prompt, max_tokens=int(len(text) * 1.2), timeout=self.config.timeout)

                duration = time.time() - start_time
                logger.debug(f'LLM processing took {duration:.2f} seconds')

                # Extract the corrected text from the response
                # The model should follow the instruction and output the corrected text
                if 'CORRECTED TEXT:' in response:
                    # Extract only the part after "CORRECTED TEXT:"
                    return response.split('CORRECTED TEXT:', 1)[1].strip()
                return response

            except Exception as e:
                if attempt < self.config.max_retry_attempts:
                    logger.warning(f'LLM processing failed (attempt {attempt + 1}): {e}. Retrying...')
                    time.sleep(1)  # Small delay before retry
                else:
                    logger.error(f'LLM processing failed after {self.config.max_retry_attempts + 1} attempts: {e}')
                    # Return original text if all attempts fail
                    return text

    def _process_long_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process long text by splitting into chunks.

        Args:
            text: Long text to process
            metadata: Optional metadata

        Returns:
            Processed text

        """
        logger.info(f'Splitting text of length {len(text)} into chunks')

        # Split text into paragraph chunks for more natural boundaries
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_length = 0

        # Group paragraphs into chunks
        for para in paragraphs:
            if current_length + len(para) > self.config.max_chunk_size and current_chunk:
                # Add the current chunk to chunks and start a new one
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        logger.info(f'Processing {len(chunks)} text chunks')

        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            logger.debug(f'Processing chunk {i + 1}/{len(chunks)}')
            processed_chunk = self._process_with_llm(chunk, metadata)
            processed_chunks.append(processed_chunk)

        # Combine the processed chunks
        return '\n\n'.join(processed_chunks)


class AsyncLLMProcessor:
    """Asynchronous processor for OCR text using LLMs.

    This class manages a background queue of OCR texts to process, allowing
    immediate return of raw OCR results while LLM processing happens in
    the background.
    """

    def __init__(self, config: Optional[LLMCleanerConfig] = None):
        """Initialize the async processor.

        Args:
            config: Configuration for the LLM cleaner

        """
        # Override any config to ensure async mode
        if config:
            config.async_processing = True
        else:
            config = LLMCleanerConfig(async_processing=True)

        self.cleaner = LLMCleaner(config)
        self.processing_queue = asyncio.Queue()
        self._processing_task = None
        self._results = {}

    async def start_processing(self):
        """Start the background processing task."""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_queue())
            logger.info('Started background LLM processing task')

    async def stop_processing(self):
        """Stop the background processing task."""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
            logger.info('Stopped background LLM processing task')

    async def submit_text(
        self,
        document_id: str,
        ocr_text: str,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit text for background processing.

        Args:
            document_id: Unique ID for the document
            ocr_text: OCR text to clean
            confidence_score: Optional confidence score
            metadata: Optional metadata

        Returns:
            The original OCR text (processing happens in background)

        """
        # Start processing if not already started
        if self._processing_task is None:
            await self.start_processing()

        # Add to queue
        await self.processing_queue.put((document_id, ocr_text, confidence_score, metadata))
        logger.info(f'Submitted document {document_id} for background processing')

        # Return original text immediately
        return ocr_text

    async def get_result(self, document_id: str, wait: bool = False) -> Optional[str]:
        """Get the processed result for a document.

        Args:
            document_id: ID of the document
            wait: Whether to wait for processing to complete

        Returns:
            Processed text if available, None otherwise

        """
        if document_id in self._results:
            return self._results[document_id]

        if wait:
            while document_id not in self._results:
                await asyncio.sleep(0.5)
            return self._results[document_id]

        return None

    async def _process_queue(self):
        """Process the queue in the background."""
        while True:
            try:
                # Get the next item from the queue
                document_id, ocr_text, confidence_score, metadata = await self.processing_queue.get()

                # Process the text
                logger.info(f'Processing document {document_id} in background')
                start_time = time.time()

                cleaned_text = await self.cleaner.clean_text_async(ocr_text, confidence_score, metadata)

                duration = time.time() - start_time
                logger.info(f'Background processing of document {document_id} completed in {duration:.2f}s')

                # Store the result
                self._results[document_id] = cleaned_text

                # Mark the task as done
                self.processing_queue.task_done()

            except asyncio.CancelledError:
                logger.info('Background processing task cancelled')
                break
            except Exception as e:
                logger.error(f'Error in background processing: {e}')
                # Continue with the next item
