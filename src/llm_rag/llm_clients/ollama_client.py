# If this OllamaClient is a core feature, httpx should be moved to main dependencies.
import json
from typing import Optional

import httpx  # httpx is in pyproject.toml's [dev] dependencies.

from llm_rag.llm_clients.base_llm_client import (
    GenerativeLLMClient,
    GenerativeLLMResponse,
    LLMRequestConfig,
)
from llm_rag.utils.logging import get_logger  # Assuming standard logger setup

logger = get_logger(__name__)

DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434'


class OllamaError(Exception):
    """Custom exception for Ollama client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        """Initialize OllamaError.

        Args:
            message: The error message.
            status_code: Optional HTTP status code from the error.
            response_text: Optional raw response text from the error.

        """
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class OllamaClient(GenerativeLLMClient):
    """Client for interacting with an Ollama generative LLM service (v0.1.30+ for JSON mode).

    See: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        api_endpoint: str = '/api/generate',
        timeout: float = 60.0,
    ):
        """Initialize the OllamaClient.

        Args:
            base_url: Base URL of the Ollama service.
            api_endpoint: API endpoint for Ollama service.
            timeout: Default timeout for requests to Ollama.

        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f'{self.base_url}{api_endpoint}'
        self.timeout = timeout
        logger.info(f'OllamaClient initialized with API URL: {self.api_url}')

    async def agenerate_response(self, prompt: str, config: LLMRequestConfig) -> GenerativeLLMResponse:
        """Asynchronously generate a response from the Ollama LLM.

        For structured JSON output, the prompt must explicitly request JSON,
        and the model must be capable of adhering to it.
        The `format: "json"` parameter can be used with compatible Ollama versions/models.
        """
        payload = {
            'model': config.model_name,
            'prompt': prompt,
            'stream': False,  # For this use case, non-streaming is simpler
            'options': {
                'temperature': config.temperature,
                'num_predict': config.max_tokens,  # Ollama uses num_predict for max_tokens
                'top_p': config.top_p,
                # Add other Ollama-specific option mappings from LLMRequestConfig if needed
            },
            # "format": "json", # Enable if Ollama version and model support it reliably.
            # This can help ensure the output is a valid JSON string.
            # The prompt should still ask for JSON as a primary instruction.
        }
        # Conditionally add format if model is known to support it or if we want to try by default
        if getattr(config, 'force_json_format', True):  # Add a flag to LLMRequestConfig if needed
            payload['format'] = 'json'

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.debug(
                    f'Sending request to Ollama: model={payload["model"]}, '
                    f'prompt length={len(prompt)}, format={payload.get("format")}'
                )
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()  # Raise HTTPStatusError for 4xx/5xx

                response_data = response.json()
                # logger.debug(f"Raw Ollama response data: {response_data}")

                generated_text = response_data.get('response', '')
                if not generated_text and response_data.get('done') is True:
                    logger.warning("Ollama returned an empty response string but indicated 'done'.")

                # If 'format: json' was used, response_data['response'] should be a string
                # that is valid JSON. If not, it's raw text.
                # The LLMComparer will be responsible for parsing this string if it's JSON.

                return GenerativeLLMResponse(
                    text=generated_text.strip(),
                    raw_response=response_data,
                    model_name_used=response_data.get('model'),
                    finish_reason=response_data.get('done_reason'),
                )
            except httpx.HTTPStatusError as e:
                logger.error(f'Ollama API request failed with status {e.response.status_code}: {e.response.text}')
                raise OllamaError(
                    message=f'Ollama API error: {e.response.status_code} - {e.response.text}',
                    status_code=e.response.status_code,
                    response_text=e.response.text,
                ) from e
            except httpx.RequestError as e:
                logger.error(f'Ollama API request failed due to network/connection issue: {e}')
                raise OllamaError(f'Ollama connection error: {e}') from e
            except json.JSONDecodeError as e:  # Should not happen if Ollama API is correct
                logger.error(f"Failed to decode Ollama's own JSON response structure: {e}")
                raise OllamaError(f'Ollama API returned invalid JSON structure: {e}') from e
            except Exception as e:
                logger.error(f'An unexpected error occurred while querying Ollama: {e}')
                raise OllamaError(f'Unexpected error during Ollama request: {e}') from e
