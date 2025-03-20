"""Model factory for creating LLM instances.

This module provides a factory for creating LLM instances
from different backends such as llama-cpp-python and Hugging Face.
"""

import logging
import os
from enum import Enum
from typing import Any, Union

# Import LLM class here to avoid circular imports later
from langchain_core.language_models.llms import LLM

logger = logging.getLogger(__name__)


class ModelBackend(str, Enum):
    """Available model backends."""

    LLAMA_CPP = "llama_cpp"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class ModelFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_model(
        model_path_or_name: str,
        backend: Union[str, ModelBackend] = ModelBackend.LLAMA_CPP,
        device: str = "cpu",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs: Any,
    ) -> LLM:
        """Create an LLM instance.

        Args:
            model_path_or_name: Path to the model file or model name to load
            backend: Backend to use for the model
            device: Device to run the model on
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top p for nucleus sampling
            repetition_penalty: Repetition penalty for generation
            **kwargs: Additional kwargs to pass to the LLM

        Returns:
            An LLM instance

        """
        backend_enum = backend if isinstance(backend, ModelBackend) else ModelBackend(backend)

        if backend_enum == ModelBackend.LLAMA_CPP:
            return ModelFactory._create_llama_cpp_model(
                model_path=model_path_or_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        elif backend_enum == ModelBackend.HUGGINGFACE:
            return ModelFactory._create_huggingface_model(
                model_name=model_path_or_name,
                device=device,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        elif backend_enum == ModelBackend.OLLAMA:
            return ModelFactory._create_ollama_model(
                model_name=model_path_or_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def _create_llama_cpp_model(
        model_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs: Any,
    ) -> LLM:
        """Create a llama-cpp-python model.

        Args:
            model_path: Path to the model file
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top p for nucleus sampling
            repetition_penalty: Repetition penalty for generation
            **kwargs: Additional kwargs to pass to the LLM

        Returns:
            A llama-cpp-python LLM instance

        """
        # Import here to avoid circular imports
        from src.llm_rag.main import CustomLlamaCpp

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create the model
        logger.info(f"Creating llama-cpp model: {model_path}")
        return CustomLlamaCpp(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            **kwargs,
        )

    @staticmethod
    def _create_huggingface_model(
        model_name: str,
        device: str = "cpu",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs: Any,
    ) -> LLM:
        """Create a Hugging Face model.

        Args:
            model_name: Name of the model to load from Hugging Face
            device: Device to run the model on
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top p for nucleus sampling
            repetition_penalty: Repetition penalty for generation
            **kwargs: Additional kwargs to pass to the LLM

        Returns:
            A Hugging Face LLM instance

        """
        # Import here to avoid circular imports
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.llm_rag.models.huggingface import HuggingFaceLLM

        # Create the model
        logger.info(f"Creating Hugging Face model: {model_name}")
        logger.info(f"Parameters: device={device}, max_tokens={max_tokens}")
        logger.info(f"Additional kwargs: {kwargs}")

        try:
            # Load the model and tokenizer
            logger.info(f"Loading model {model_name} on {device}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
            )

            return HuggingFaceLLM(
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error creating HuggingFaceLLM: {e}")
            # Print all parameters for debugging
            logger.error(f"model_name: {model_name}")
            logger.error(f"device: {device}")
            logger.error(f"max_tokens: {max_tokens}")
            logger.error(f"temperature: {temperature}")
            logger.error(f"top_p: {top_p}")
            logger.error(f"repetition_penalty: {repetition_penalty}")
            logger.error(f"kwargs: {kwargs}")
            raise

    @staticmethod
    def _create_ollama_model(
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs: Any,
    ) -> LLM:
        """Create an Ollama model.

        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3')
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top p for nucleus sampling
            repetition_penalty: Repetition penalty for generation
            **kwargs: Additional kwargs to pass to the LLM

        Returns:
            An Ollama LLM instance

        """
        logger.info(f"Creating Ollama model: {model_name}")
        logger.info(f"Parameters: max_tokens={max_tokens}, temperature={temperature}")
        logger.info(f"Additional kwargs: {kwargs}")

        try:
            # Try importing from langchain_ollama first (preferred)
            try:
                from langchain_ollama import OllamaLLM

                logger.info("Using langchain_ollama.OllamaLLM")
                return OllamaLLM(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty,
                    **kwargs,
                )
            except ImportError:
                # Fallback to langchain_community
                try:
                    from langchain_community.llms.ollama import OllamaLLM

                    logger.info("Using langchain_community.llms.ollama.OllamaLLM")
                    return OllamaLLM(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repetition_penalty,
                        **kwargs,
                    )
                except ImportError:
                    # Final fallback to older API
                    from langchain_community.llms import Ollama

                    logger.info("Using langchain_community.llms.Ollama")
                    return Ollama(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repetition_penalty,
                        **kwargs,
                    )
        except Exception as e:
            logger.error(f"Error creating Ollama model: {e}")
            raise
