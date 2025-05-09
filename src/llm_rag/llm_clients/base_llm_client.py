from typing import Any, Dict, Optional, Protocol

from pydantic import BaseModel, Field


class LLMRequestConfig(BaseModel):
    """Configuration for making a request to a generative LLM."""

    model_name: str = Field(description='The name or identifier of the LLM to use.')
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description='Controls randomness. Lower is more deterministic.'
    )
    max_tokens: int = Field(default=512, gt=0, description='Maximum number of tokens to generate.')
    # Example of other common parameters, can be extended
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description='Nucleus sampling parameter.')
    # stop_sequences: Optional[List[str]] = Field(default=None, description="Sequences to stop generation.")
    # presence_penalty: Optional[float] = Field(default=None, description="Penalty for new tokens (presence).")
    # frequency_penalty: Optional[float] = Field(default=None, description="Penalty for new tokens (frequency).")

    class Config:
        """Pydantic model configuration."""

        extra = 'allow'  # Allow other Ollama/LLM specific params not explicitly defined


class GenerativeLLMResponse(BaseModel):
    """Standardized response from a generative LLM client."""

    text: str = Field(description='The generated text content from the LLM.')
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None, description='The full, raw response from the LLM API for debugging or extended info.'
    )
    model_name_used: Optional[str] = Field(default=None, description='Actual model name that processed the request.')
    # usage_metadata: Optional[Dict[str, int]] = Field(default=None, description="Token usage (prompt, compl, total).")
    finish_reason: Optional[str] = Field(
        default=None, description="Reason why the generation finished (e.g., 'stop', 'length')."
    )

    class Config:
        """Pydantic model configuration."""

        frozen = True
        extra = 'forbid'


class GenerativeLLMClient(Protocol):
    """Protocol defining the interface for a client that interacts with a generative LLM."""

    async def agenerate_response(self, prompt: str, config: LLMRequestConfig) -> GenerativeLLMResponse:
        """Asynchronously generates a response from the LLM based on the given prompt and configuration.

        Args:
            prompt: The input prompt string for the LLM.
            config: Configuration for the LLM request.

        Returns:
            A GenerativeLLMResponse object containing the LLM's output.

        Raises:
            Various exceptions on communication failure or API errors.

        """
        ...
