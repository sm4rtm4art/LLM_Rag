"""Hugging Face models implementation for the LLM RAG system.

This module provides a wrapper around transformers models for use with the RAG system.
It supports various model architectures from Hugging Face, including Llama-3.
"""

import logging
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLM):
    """Hugging Face LLM implementation for the RAG system."""

    model_name: str = Field(..., description="The name of the model to load from Hugging Face")
    device: str = Field(default="cpu", description="Device to run the model on")
    max_tokens: int = Field(default=512, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    top_p: float = Field(default=0.95, description="Top p for nucleus sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty for generation")

    # These aren't Pydantic fields, just regular instance attributes
    model: Any = None
    tokenizer: Any = None
    pipeline: Any = None

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs: Any,
    ):
        """Initialize the Hugging Face LLM.

        Args:
            model_name: Name of the model to load from Hugging Face.
            device: Device to run the model on.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for sampling.
            top_p: Top p for nucleus sampling.
            repetition_penalty: Repetition penalty for generation.
            **kwargs: Additional kwargs to pass to the LLM.

        """
        # Debug logging
        logger.info(f"Initializing HuggingFaceLLM with model_name={model_name}")

        # We need to explicitly set all the fields that are defined in the class
        super().__init__(
            model_name=model_name,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

        # Load the model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """Load the model and tokenizer from Hugging Face."""
        logger.info(f"Loading model {self.model_name} on {self.device}")

        try:
            # For Llama models, we need to use specific parameters
            if "llama" in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            else:
                # For other models, use standard parameters
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                )

            # Create the pipeline - NEVER pass device parameter to avoid accelerate issues
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                # Explicitly NOT passing device parameter
            )

            logger.info(f"Successfully loaded model {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "huggingface"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the model with the prompt and return the output.

        Args:
            prompt: String prompt to pass to the model.
            stop: List of strings to stop generation when encountered.
            run_manager: Callback manager for the LLM run.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            String output from the model.

        """
        # Format the prompt
        formatted_prompt = self._format_prompt(prompt)

        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True,
            "return_full_text": False,
            **kwargs,
        }

        # Generate text
        logger.info(f"Generating text with prompt: {formatted_prompt[:50]}...")

        # Stream tokens if a run_manager is provided
        generated_text = ""

        try:
            outputs = self.pipeline(formatted_prompt, **gen_kwargs)

            # Extract the generated text
            generated_text = outputs[0]["generated_text"]

            # Remove the prompt from the output if it's included
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt) :].strip()

            # Call callback for entire text if provided
            if run_manager:
                run_manager.on_llm_new_token(generated_text)

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"

        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in generated_text:
                    generated_text = generated_text[: generated_text.find(stop_seq) + len(stop_seq)]

        return generated_text.strip()

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for the model.

        Args:
            prompt: The prompt to format.

        Returns:
            The formatted prompt.

        """
        # For Llama models, we need to use a specific format
        if "llama" in self.model_name.lower():
            # Llama-3 instruction format
            return f"<|begin_of_text|><|prompt|>{prompt}<|answer|>"

        # For other models, just return the prompt as is
        return prompt
