"""Tests for generation components in the RAG pipeline.

This module contains comprehensive tests for the response generation
components of the RAG pipeline system.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import PromptTemplate

from llm_rag.rag.pipeline.generation import (
    DEFAULT_PROMPT_TEMPLATE,
    BaseGenerator,
    LLMGenerator,
    TemplatedGenerator,
    create_generator,
)
from llm_rag.utils.errors import ModelError, PipelineError


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response="This is a mocked response"):
        self.response = response
        self.invoke_called = False
        self.last_prompt = None
        self.last_kwargs = None

    def invoke(self, prompt, **kwargs):
        """Mock invoke method."""
        self.invoke_called = True
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return self.response


class TestBaseGenerator(unittest.TestCase):
    """Tests for the BaseGenerator abstract base class."""

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""

        # Create a concrete subclass for testing
        class TestGenerator(BaseGenerator):
            def generate(self, query: str, context: str, history: str = "", **kwargs) -> str:
                return "test"

        generator = TestGenerator()
        # This should not raise an exception
        generator._validate_inputs("test query", "test context")

    def test_validate_inputs_invalid_query(self):
        """Test input validation with invalid query."""

        class TestGenerator(BaseGenerator):
            def generate(self, query: str, context: str, history: str = "", **kwargs) -> str:
                return "test"

        generator = TestGenerator()

        # Test with empty query
        with self.assertRaises(PipelineError) as context:
            generator._validate_inputs("", "test context")
        self.assertIn("Query must be a non-empty string", str(context.exception))

        # Test with non-string query
        with self.assertRaises(PipelineError) as context:
            generator._validate_inputs(123, "test context")  # type: ignore
        self.assertIn("Query must be a non-empty string", str(context.exception))

    def test_validate_inputs_invalid_context(self):
        """Test input validation with invalid context."""

        class TestGenerator(BaseGenerator):
            def generate(self, query: str, context: str, history: str = "", **kwargs) -> str:
                return "test"

        generator = TestGenerator()

        # Test with non-string context
        with self.assertRaises(PipelineError) as context:
            generator._validate_inputs("test query", 123)  # type: ignore
        self.assertIn("Context must be a string", str(context.exception))


class TestLLMGenerator(unittest.TestCase):
    """Tests for the LLMGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLM()
        self.generator = LLMGenerator(llm=self.mock_llm)

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default initialization
        generator = LLMGenerator(llm=self.mock_llm)
        self.assertEqual(generator.llm, self.mock_llm)
        self.assertTrue(generator.apply_anti_hallucination)
        self.assertEqual(generator.prompt_template.template, DEFAULT_PROMPT_TEMPLATE)

        # Custom initialization with string template
        custom_template = "Custom template with {context} and {query}"
        generator = LLMGenerator(
            llm=self.mock_llm,
            prompt_template=custom_template,
            apply_anti_hallucination=False,
        )
        self.assertEqual(generator.llm, self.mock_llm)
        self.assertFalse(generator.apply_anti_hallucination)
        self.assertEqual(generator.prompt_template.template, custom_template)

        # Custom initialization with PromptTemplate
        template = PromptTemplate(
            template="Template object with {context} and {query}",
            input_variables=["context", "query"],
        )
        generator = LLMGenerator(llm=self.mock_llm, prompt_template=template)
        self.assertEqual(generator.prompt_template, template)

    @patch("llm_rag.rag.pipeline.generation.post_process_response")
    def test_generate_with_anti_hallucination(self, mock_post_process):
        """Test generation with anti-hallucination post-processing."""
        mock_post_process.return_value = "Post-processed response"

        result = self.generator.generate(
            query="test query",
            context="test context",
            history="test history",
        )

        # Check that the LLM was called with appropriate prompt
        self.assertTrue(self.mock_llm.invoke_called)
        self.assertIn("test query", self.mock_llm.last_prompt)
        self.assertIn("test context", self.mock_llm.last_prompt)
        self.assertIn("test history", self.mock_llm.last_prompt)

        # Check that post-processing was applied
        mock_post_process.assert_called_once()
        self.assertEqual(result, "Post-processed response")

    @patch("llm_rag.rag.pipeline.generation.post_process_response")
    def test_generate_without_anti_hallucination(self, mock_post_process):
        """Test generation without anti-hallucination post-processing."""
        generator = LLMGenerator(llm=self.mock_llm, apply_anti_hallucination=False)

        result = generator.generate(
            query="test query",
            context="test context",
        )

        # Check that the LLM was called
        self.assertTrue(self.mock_llm.invoke_called)

        # Check that post-processing was not applied
        mock_post_process.assert_not_called()
        self.assertEqual(result, "This is a mocked response")

    def test_generate_with_generation_params(self):
        """Test generation with additional parameters."""
        self.generator.generate(
            query="test query",
            context="test context",
            temperature=0.7,
            max_tokens=500,
        )

        # Check that parameters were passed to the LLM
        self.assertEqual(self.mock_llm.last_kwargs.get("temperature"), 0.7)
        self.assertEqual(self.mock_llm.last_kwargs.get("max_tokens"), 500)

    def test_generate_with_llm_error(self):
        """Test handling of LLM errors."""
        # Create a mock LLM that raises an exception
        error_llm = MagicMock()
        error_llm.invoke.side_effect = Exception("LLM error")

        generator = LLMGenerator(llm=error_llm)

        with self.assertRaises(ModelError) as context:
            generator.generate(query="test query", context="test context")

        self.assertIn("Language model failed to generate response", str(context.exception))

    def test_generate_with_object_response(self):
        """Test handling of object responses from LLM."""

        # Create a mock LLM that returns an object with content attribute
        class ObjectResponse:
            def __init__(self, content):
                self.content = content

        object_llm = MagicMock()
        object_llm.invoke.return_value = ObjectResponse("Response content")

        generator = LLMGenerator(llm=object_llm, apply_anti_hallucination=False)

        result = generator.generate(query="test query", context="test context")
        self.assertEqual(result, "Response content")


class TestTemplatedGenerator(unittest.TestCase):
    """Tests for the TemplatedGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLM()
        self.templates = {
            "default": "Default template with {context} and {query}",
            "qa": "Q&A template with {query} using {context}",
        }
        self.generator = TemplatedGenerator(
            llm=self.mock_llm,
            templates=self.templates,
        )

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Check default initialization
        self.assertEqual(self.generator.llm, self.mock_llm)
        self.assertTrue(self.generator.apply_anti_hallucination)
        self.assertEqual(self.generator.default_template, "default")

        # Custom initialization with different default
        generator = TemplatedGenerator(
            llm=self.mock_llm,
            templates=self.templates,
            default_template="qa",
            apply_anti_hallucination=False,
        )
        self.assertEqual(generator.default_template, "qa")
        self.assertFalse(generator.apply_anti_hallucination)

    def test_generate_with_template_selection(self):
        """Test generation with template selection."""
        # Generate with default template
        self.generator.generate(query="test query", context="test context")
        self.assertIn("Default template", self.mock_llm.last_prompt)

        # Generate with selected template
        self.generator.generate(
            query="test query",
            context="test context",
            template="qa",
        )
        self.assertIn("Q&A template", self.mock_llm.last_prompt)

    def test_generate_with_invalid_template(self):
        """Test generation with invalid template selection."""
        with self.assertRaises(PipelineError) as context:
            self.generator.generate(
                query="test query",
                context="test context",
                template="nonexistent",
            )
        self.assertIn("Template 'nonexistent' not found", str(context.exception))


class TestCreateGenerator(unittest.TestCase):
    """Tests for the create_generator factory function."""

    def test_create_with_llm(self):
        """Test creating a generator with a valid LLM."""
        mock_llm = MockLLM()
        generator = create_generator(llm=mock_llm)

        self.assertIsInstance(generator, LLMGenerator)
        self.assertEqual(generator.llm, mock_llm)

    def test_create_with_prompt_template(self):
        """Test creating a generator with a custom prompt template."""
        mock_llm = MockLLM()
        custom_template = "Custom template for {query} and {context}"

        generator = create_generator(llm=mock_llm, prompt_template=custom_template)

        self.assertIsInstance(generator, LLMGenerator)
        self.assertEqual(generator.prompt_template.template, custom_template)


@pytest.mark.parametrize(
    "query,context,expected_error",
    [
        (None, "context", "Query must be a non-empty string"),
        ("", "context", "Query must be a non-empty string"),
        ("query", None, "Context must be a string"),
        ("query", 123, "Context must be a string"),
    ],
)
def test_generator_validation_errors(query, context, expected_error):
    """Test validation errors with various invalid inputs."""
    mock_llm = MockLLM()
    generator = LLMGenerator(llm=mock_llm)

    with pytest.raises(PipelineError) as excinfo:
        generator.generate(query=query, context=context)

    assert expected_error in str(excinfo.value)


if __name__ == "__main__":
    unittest.main()
