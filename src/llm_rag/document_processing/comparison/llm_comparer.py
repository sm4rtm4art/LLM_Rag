import json
from typing import Optional, Type

from pydantic import ValidationError

# Assuming LLMAnalysisResult will be in domain_models.py
from llm_rag.document_processing.comparison.domain_models import (
    LLMAnalysisResult,
)
from llm_rag.llm_clients.base_llm_client import (
    GenerativeLLMClient,
    LLMRequestConfig,
)
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


NEW_LEGAL_PROMPT_TEMPLATE = (
    'You are a legal assistant specialized in analyzing changes in legal or contractual texts.\\n\\n'
    'Below are two text sections — Section A and Section B — extracted from legal documents.\\n'
    'Your task is to analyze whether they express the same legal meaning or introduce any semantic, ontological, '
    'or legal changes.\\n\\n'
    'Respond ONLY in **strict JSON** using the following keys:\\n'
    '- "comparison_category": (string) — One of:\\n'
    '    - "SEMANTIC_REWRITE": Section B rephrases A with no change in legal meaning.\\n'
    '    - "ONTOLOGICAL_SUBSUMPTION": One generalizes or specializes the concept in the other.\\n'
    '    - "LEGAL_EFFECT_CHANGE": There is a change in rights, duties, or enforceability.\\n'
    '    - "STRUCTURAL_REORDERING": Reordering or formatting differences only.\\n'
    '    - "DIFFERENT_CONCEPTS": Unrelated legal topics.\\n'
    '    - "NO_MEANINGFUL_CONTENT": One or both sections are too short or garbled to analyze.\\n'
    '    - "UNCERTAIN": You cannot determine the relationship due to ambiguity or lack of context.\\n'
    '- "explanation": (string) — Briefly explain why you chose that category. Mention key terms or changes.\\n'
    '- "confidence": (float) — A number between 0.0 and 1.0 indicating your confidence.\\n\\n'
    '---\\n'
    'Section A:\\n'
    "'''{text_section_a}'''\\n\\n"
    'Section B:\\n'
    "'''{text_section_b}'''\\n\\n"
    '---\\n'
    'Output:\\n'
    'Respond with JSON only. Do not include any prose or commentary.\\n'
)


class LLMComparer:
    """Compares two text sections using a generative LLM for nuanced analysis."""

    DEFAULT_PROMPT_TEMPLATE = NEW_LEGAL_PROMPT_TEMPLATE

    def __init__(
        self,
        llm_client: GenerativeLLMClient,
        default_llm_config: Optional[LLMRequestConfig] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        llm_analysis_model: Type[LLMAnalysisResult] = LLMAnalysisResult,
    ):
        """Initialize the LLMComparer.

        Args:
            llm_client: An instance of a GenerativeLLMClient.
            default_llm_config: Default configuration for LLM requests.
            prompt_template: The prompt template to use for comparison.
            llm_analysis_model: The Pydantic model to parse LLM responses into.

        """
        self.llm_client = llm_client
        self.default_llm_config = default_llm_config or LLMRequestConfig(
            model_name='phi3:mini',  # Default model, ensure availability in Ollama
            temperature=0.05,  # Low temp for factual, deterministic output
            max_tokens=768,  # Accommodates JSON and explanation
        )
        self.prompt_template = prompt_template
        self.llm_analysis_model = llm_analysis_model
        logger.info(
            f'LLMComparer initialized with client: {type(llm_client).__name__}, '
            f'model: {self.default_llm_config.model_name}'
        )

    async def analyze_sections(
        self,
        text_section_a: str,
        text_section_b: str,
        llm_config_override: Optional[LLMRequestConfig] = None,
    ) -> LLMAnalysisResult:
        """Analyzes two text sections using an LLM and parses the response.

        Args:
            text_section_a: The first text section.
            text_section_b: The second text section.
            llm_config_override: Optional LLM config to override defaults.

        Returns:
            An LLMAnalysisResult object.

        """
        current_config = llm_config_override or self.default_llm_config
        prompt = self.prompt_template.format(text_section_a=text_section_a, text_section_b=text_section_b)

        raw_llm_output_text = None
        try:
            logger.info(
                f'Requesting LLM analysis. Model: {current_config.model_name}. '
                f'Sec A len: {len(text_section_a)}, Sec B len: {len(text_section_b)}'
            )
            llm_api_response = await self.llm_client.agenerate_response(prompt=prompt, config=current_config)
            raw_llm_output_text = llm_api_response.text
            logger.debug(f'Raw LLM output: {raw_llm_output_text[:200]}...')  # Shorter log

            # The response text is expected to be a JSON string from the LLM
            # Attempt to parse the JSON string, removing potential markdown code fences
            cleaned_json_string = raw_llm_output_text.strip()
            if cleaned_json_string.startswith('```json'):
                cleaned_json_string = cleaned_json_string[len('```json') :]
            if cleaned_json_string.startswith('```'):  # Generic code fence
                cleaned_json_string = cleaned_json_string[len('```') :]
            if cleaned_json_string.endswith('```'):
                cleaned_json_string = cleaned_json_string[: -len('```')]
            cleaned_json_string = cleaned_json_string.strip()

            parsed_data = json.loads(cleaned_json_string)
            analysis_result = self.llm_analysis_model(
                **parsed_data,
                raw_llm_response=raw_llm_output_text,  # Store original full raw response
            )
            logger.info(f'Successfully parsed LLM analysis: {analysis_result.comparison_category}')
            return analysis_result

        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode JSON from LLM: {e}. Response snippet: '{str(raw_llm_output_text)[:100]}...'"
            logger.error(error_msg)
            return self.llm_analysis_model(
                comparison_category='llm_error',
                explanation=error_msg,
                raw_llm_response=raw_llm_output_text,
            )
        except ValidationError as e:
            error_msg = (
                f"LLM JSON didn't match Pydantic model {self.llm_analysis_model.__name__}: {e}. "
                f"Response snippet: '{str(raw_llm_output_text)[:100]}...'"
            )
            logger.error(error_msg)
            # Log problematic data if possible (e.g., parsed_data before validation)
            # logger.error(f"Problematic parsed data: {parsed_data if 'parsed_data' in locals() else 'N/A'}")
            return self.llm_analysis_model(
                comparison_category='llm_error',
                explanation=error_msg,
                raw_llm_response=raw_llm_output_text,
            )
        except Exception as e:
            error_msg = f'Error in LLM section analysis: {type(e).__name__} - {e}'
            logger.error(error_msg, exc_info=True)
            return self.llm_analysis_model(
                comparison_category='llm_error',
                explanation=error_msg,
                raw_llm_response=raw_llm_output_text,  # Might be None if error before LLM call
            )
