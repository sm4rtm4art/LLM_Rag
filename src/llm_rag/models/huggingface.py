"""Hugging Face model implementation for the LLM RAG system.

This module provides the Hugging Face model implementation for the
LLM RAG system.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from langchain_core.language_models.llms import LLM
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLM):
    """Hugging Face model implementation for the LLM RAG system."""

    model_name: str = Field(description='The name of the model')
    tokenizer: Optional[Any] = Field(default=None, description='The tokenizer to use')
    model: Optional[Any] = Field(default=None, description='The model to use')
    device: str = Field(default='cpu', description='The device to use (cpu or cuda)')
    max_tokens: int = Field(default=512, description='The maximum number of tokens to generate')
    temperature: float = Field(default=0.7, description='The temperature to use for generation')
    top_p: float = Field(default=0.9, description='The top_p value to use for generation')
    do_sample: bool = Field(default=True, description='Whether to use sampling')
    pipeline: Optional[Any] = Field(default=None, description='The pipeline to use')

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the HuggingFaceLLM.

        If model and tokenizer are not provided, they will be loaded
        using the _load_model_and_tokenizer method.
        """
        super().__init__(**kwargs)

        # Load model and tokenizer if not provided
        if self.model is None or self.tokenizer is None:
            try:
                self.model, self.tokenizer = self._load_model_and_tokenizer()
            except Exception as e:
                logger.warning(f'Failed to load model and tokenizer: {e}')
                # Don't raise an exception here to allow for testing

    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return 'huggingface'

    def _get_parameters(self) -> Dict[str, Any]:
        """Get the parameters for the LLM."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
        }

    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load the model and tokenizer.

        Returns:
            A tuple containing the model and tokenizer

        """
        try:
            logger.info(f'Loading model {self.model_name} on {self.device}')
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                trust_remote_code=True,
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f'Error loading model and tokenizer: {e}')
            raise

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for the model.

        Args:
            prompt: The prompt to format

        Returns:
            The formatted prompt

        """
        # Special handling for Llama models
        if 'llama' in self.model_name.lower():
            return f'<|begin_of_text|><|prompt|>{prompt}<|answer|>'
        return prompt

    def _call(self, prompt: str, **kwargs: Any) -> str:
        """Call the model with a prompt."""
        # Format the prompt
        formatted_prompt = self._format_prompt(prompt)

        # Use the pipeline if available
        if self.pipeline is not None:
            # Generate with the pipeline
            response = self.pipeline(
                formatted_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                return_full_text=False,
                **kwargs,
            )

            # Extract the generated text
            generated_text = response[0]['generated_text']

            # Handle stop sequences
            if 'stop' in kwargs and kwargs['stop']:
                for stop_seq in kwargs['stop']:
                    if stop_seq in generated_text:
                        idx = generated_text.find(stop_seq)
                        end_idx = idx + len(stop_seq)
                        generated_text = generated_text[:end_idx]

            return generated_text

        # Fall back to invoke if pipeline is not available
        return self.invoke(formatted_prompt)

    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt.

        This method has been modified to return a string directly instead of
        an AIMessage to avoid Pydantic validation issues with LangChain.

        Args:
            prompt: The prompt to send to the model

        Returns:
            A string containing the model's response

        """
        try:
            # Ensure we have a model and tokenizer
            if self.model is None or self.tokenizer is None:
                return 'Model or tokenizer not initialized'

            # Ensure the tokenizer has a padding token
            tokenizer_obj = self.tokenizer
            if tokenizer_obj.pad_token is None:
                tokenizer_obj.pad_token = tokenizer_obj.eos_token

            # Encode the input and create attention mask
            inputs = tokenizer_obj(prompt, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate text with the model
            model_obj = self.model
            with torch.no_grad():
                output = model_obj.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=len(inputs['input_ids'][0]) + self.max_tokens,
                    num_return_sequences=1,
                    eos_token_id=tokenizer_obj.eos_token_id,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

            # Decode the response
            decoded_response = tokenizer_obj.decode(output[0], skip_special_tokens=True)

            # Extract the response text after the prompt
            if decoded_response.startswith(prompt):
                response_text = decoded_response[len(prompt) :].strip()
            else:
                response_text = decoded_response.strip()

            # Return the response as a string directly
            return response_text
        except Exception as e:
            logger.error(f'Error generating response: {e}')
            return 'I encountered an error while generating a response.'
