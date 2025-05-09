import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rag.document_processing.comparison.domain_models import (
    LLMAnalysisResult,
)
from llm_rag.document_processing.comparison.llm_comparer import LLMComparer
from llm_rag.llm_clients.base_llm_client import (
    GenerativeLLMClient,
    GenerativeLLMResponse,
)

# A sample valid JSON structure that LLMComparer expects
VALID_LLM_RESPONSE_DATA = {
    'comparison_category': 'SEMANTIC_REWRITE',
    'explanation': ('The sections are essentially saying the same thing using different words.'),
    'confidence': 0.9,
}

MALFORMED_JSON_STRING = '{"comparison_category": "SEMANTIC_REWRITE", "explanation": "Oops, no closing brace'

# Valid JSON, but does not match LLMAnalysisResult structure
# (e.g., missing required fields)
MISMATCHED_JSON_DATA = {
    'category': 'SEMANTIC_REWRITE',  # Wrong key name
    'desc': 'Some description.',
}


@pytest.fixture
def mock_llm_client():
    """Provides a MagicMock for GenerativeLLMClient."""
    client = MagicMock(spec=GenerativeLLMClient)  # Can use OllamaClient if more specific spec is needed
    # Ensure agenerate_response is an AsyncMock if it's called with await
    client.agenerate_response = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_analyze_sections_successful_parse(mock_llm_client):
    """Test successful analysis and parsing of LLM response."""
    mock_llm_client.agenerate_response.return_value = GenerativeLLMResponse(
        text=json.dumps(VALID_LLM_RESPONSE_DATA),  # Simulate LLM returning JSON
        model_name_used='test-model',
        finish_reason='stop',
    )
    comparer = LLMComparer(llm_client=mock_llm_client)
    result = await comparer.analyze_sections('text a', 'text b')

    assert isinstance(result, LLMAnalysisResult)
    assert result.comparison_category == VALID_LLM_RESPONSE_DATA['comparison_category']
    assert result.explanation == VALID_LLM_RESPONSE_DATA['explanation']
    assert result.confidence == VALID_LLM_RESPONSE_DATA['confidence']
    assert result.raw_llm_response == json.dumps(VALID_LLM_RESPONSE_DATA)


@pytest.mark.asyncio
async def test_analyze_sections_json_decode_error(mock_llm_client):
    """Test handling of JSONDecodeError when LLM output is malformed."""
    mock_llm_client.agenerate_response.return_value = GenerativeLLMResponse(
        text=MALFORMED_JSON_STRING,
        model_name_used='test-model',
        finish_reason='stop',
    )
    comparer = LLMComparer(llm_client=mock_llm_client)
    result = await comparer.analyze_sections('text a', 'text b')

    assert isinstance(result, LLMAnalysisResult)
    assert result.comparison_category == 'llm_error'
    assert 'Failed to decode JSON from LLM' in result.explanation
    assert result.raw_llm_response == MALFORMED_JSON_STRING


@pytest.mark.asyncio
async def test_analyze_sections_pydantic_validation_error(mock_llm_client):
    """
    Test handling of Pydantic ValidationError when LLM JSON has wrong structure.
    """
    mismatched_json_string = json.dumps(MISMATCHED_JSON_DATA)
    mock_llm_client.agenerate_response.return_value = GenerativeLLMResponse(
        text=mismatched_json_string,
        model_name_used='test-model',
        finish_reason='stop',
    )
    comparer = LLMComparer(llm_client=mock_llm_client)
    result = await comparer.analyze_sections('text a', 'text b')

    assert isinstance(result, LLMAnalysisResult)
    assert result.comparison_category == 'llm_error'
    assert f"LLM JSON didn't match Pydantic model {LLMAnalysisResult.__name__}" in result.explanation
    assert result.raw_llm_response == mismatched_json_string


@pytest.mark.asyncio
async def test_analyze_sections_llm_client_exception(mock_llm_client):
    """Test handling of exceptions raised by the LLM client."""
    mock_llm_client.agenerate_response.side_effect = Exception('Simulated LLM client error')

    comparer = LLMComparer(llm_client=mock_llm_client)
    result = await comparer.analyze_sections('text a', 'text b')

    assert isinstance(result, LLMAnalysisResult)
    assert result.comparison_category == 'llm_error'
    assert ('Error in LLM section analysis: Exception - Simulated LLM client error') in result.explanation
    # Because the error happened before/during LLM call
    assert result.raw_llm_response is None


@pytest.mark.asyncio
async def test_analyze_sections_handles_code_fences(mock_llm_client):
    """
    Test that LLM responses with JSON within markdown code fences are handled.
    """
    fenced_response_data = f'```json\n{json.dumps(VALID_LLM_RESPONSE_DATA)}\n```'
    mock_llm_client.agenerate_response.return_value = GenerativeLLMResponse(
        text=fenced_response_data,
        model_name_used='test-model',
        finish_reason='stop',
    )
    comparer = LLMComparer(llm_client=mock_llm_client)
    result = await comparer.analyze_sections('text a', 'text b')

    assert isinstance(result, LLMAnalysisResult)
    assert result.comparison_category == VALID_LLM_RESPONSE_DATA['comparison_category']
    assert result.explanation == VALID_LLM_RESPONSE_DATA['explanation']
    assert result.raw_llm_response == fenced_response_data
