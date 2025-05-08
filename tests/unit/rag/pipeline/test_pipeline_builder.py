"""Unit tests for the pipeline_builder.py module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

from llm_rag.rag.pipeline.context import BaseContextFormatter
from llm_rag.rag.pipeline.generation import BaseGenerator
from llm_rag.rag.pipeline.retrieval import BaseRetriever
from llm_rag.utils.errors import PipelineError
from src.llm_rag.rag.pipeline.pipeline_builder import RAGPipelineBuilder, create_rag_pipeline


class TestPipelineBuilder:
    """Test cases for the pipeline_builder module."""

    def test_create_rag_pipeline(self):
        """Test the create_rag_pipeline factory function."""
        builder = create_rag_pipeline()
        assert isinstance(builder, RAGPipelineBuilder)

    def test_builder_initialization(self):
        """Test the builder is correctly initialized with default values."""
        builder = RAGPipelineBuilder()

        # Check default pipeline type
        assert builder._pipeline_type == 'standard'

        # Check default component values
        assert builder._retriever is None
        assert builder._formatter is None
        assert builder._generator is None

        # Check default configuration dictionaries
        assert builder._retriever_config == {}
        assert builder._formatter_config == {}
        assert builder._generator_config == {}
        assert builder._pipeline_config == {}
        assert builder._template is None
        assert builder._llm is None

    def test_with_standard_pipeline(self):
        """Test setting the pipeline type to standard."""
        builder = RAGPipelineBuilder()
        result = builder.with_standard_pipeline()

        assert builder._pipeline_type == 'standard'
        assert result is builder  # Check method chaining

    def test_with_conversational_pipeline(self):
        """Test setting the pipeline type to conversational."""
        builder = RAGPipelineBuilder()
        result = builder.with_conversational_pipeline()

        assert builder._pipeline_type == 'conversational'
        assert result is builder  # Check method chaining

    def test_with_vector_retriever(self):
        """Test configuring a vector retriever."""
        mock_vectorstore = MagicMock(spec=VectorStore)

        builder = RAGPipelineBuilder()
        result = builder.with_vector_retriever(vectorstore=mock_vectorstore, top_k=10)

        assert builder._retriever_config == {'type': 'vector', 'source': mock_vectorstore, 'top_k': 10}
        assert result is builder  # Check method chaining

    def test_with_hybrid_retriever(self):
        """Test configuring a hybrid retriever."""
        mock_retrievers = [MagicMock(spec=BaseRetriever), MagicMock(spec=BaseRetriever)]
        weights = [0.7, 0.3]

        builder = RAGPipelineBuilder()
        result = builder.with_hybrid_retriever(retrievers=mock_retrievers, weights=weights)

        assert builder._retriever_config == {'type': 'hybrid', 'source': mock_retrievers, 'weights': weights}
        assert result is builder  # Check method chaining

    def test_with_custom_retriever(self):
        """Test configuring a custom retriever."""
        mock_retriever = MagicMock(spec=BaseRetriever)

        builder = RAGPipelineBuilder()
        result = builder.with_custom_retriever(retriever=mock_retriever)

        assert builder._retriever is mock_retriever
        assert result is builder  # Check method chaining

    def test_with_simple_formatter(self):
        """Test configuring a simple formatter."""
        builder = RAGPipelineBuilder()
        result = builder.with_simple_formatter(include_metadata=False, max_length=1000)

        assert builder._formatter_config == {'type': 'simple', 'include_metadata': False, 'max_length': 1000}
        assert result is builder  # Check method chaining

    def test_with_markdown_formatter(self):
        """Test configuring a markdown formatter."""
        builder = RAGPipelineBuilder()
        result = builder.with_markdown_formatter(include_metadata=True, max_length=2000)

        assert builder._formatter_config == {'type': 'markdown', 'include_metadata': True, 'max_length': 2000}
        assert result is builder  # Check method chaining

    def test_with_custom_formatter(self):
        """Test configuring a custom formatter."""
        mock_formatter = MagicMock(spec=BaseContextFormatter)

        builder = RAGPipelineBuilder()
        result = builder.with_custom_formatter(formatter=mock_formatter)

        assert builder._formatter is mock_formatter
        assert result is builder  # Check method chaining

    def test_with_llm_generator(self):
        """Test configuring an LLM generator."""
        mock_llm = MagicMock(spec=BaseLanguageModel)
        mock_prompt = PromptTemplate.from_template('Test template {context}')

        builder = RAGPipelineBuilder()
        result = builder.with_llm_generator(llm=mock_llm, prompt_template=mock_prompt, apply_anti_hallucination=False)

        assert builder._llm is mock_llm
        assert builder._generator_config == {
            'type': 'llm',
            'prompt_template': mock_prompt,
            'apply_anti_hallucination': False,
        }
        assert result is builder  # Check method chaining

    def test_with_templated_generator(self):
        """Test configuring a templated generator."""
        mock_llm = MagicMock(spec=BaseLanguageModel)
        templates = {
            'default': 'Default template {context}',
            'custom': PromptTemplate.from_template('Custom template {context}'),
        }

        builder = RAGPipelineBuilder()
        result = builder.with_templated_generator(
            llm=mock_llm, templates=templates, default_template='custom', apply_anti_hallucination=True
        )

        assert builder._llm is mock_llm
        assert builder._generator_config == {
            'type': 'templated',
            'templates': templates,
            'default_template': 'custom',
            'apply_anti_hallucination': True,
        }
        assert result is builder  # Check method chaining

    def test_with_custom_generator(self):
        """Test configuring a custom generator."""
        mock_generator = MagicMock(spec=BaseGenerator)

        builder = RAGPipelineBuilder()
        result = builder.with_custom_generator(generator=mock_generator)

        assert builder._generator is mock_generator
        assert result is builder  # Check method chaining

    def test_with_config(self):
        """Test setting additional pipeline configuration."""
        builder = RAGPipelineBuilder()
        result = builder.with_config(max_tokens=1000, temperature=0.7, custom_param='value')

        assert builder._pipeline_config == {'max_tokens': 1000, 'temperature': 0.7, 'custom_param': 'value'}

        # Test updating existing config
        result = builder.with_config(temperature=0.5, new_param='new_value')
        assert builder._pipeline_config == {
            'max_tokens': 1000,
            'temperature': 0.5,
            'custom_param': 'value',
            'new_param': 'new_value',
        }

        assert result is builder  # Check method chaining

    @patch('src.llm_rag.rag.pipeline.pipeline_builder.rag_factory')
    @patch('src.llm_rag.rag.pipeline.pipeline_builder.RAGPipeline')
    def test_build_standard_pipeline(self, mock_pipeline_cls, mock_factory):
        """Test building a standard pipeline."""
        # Setup mocks
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_llm = MagicMock(spec=BaseLanguageModel)
        mock_retriever = MagicMock(spec=BaseRetriever)
        mock_formatter = MagicMock(spec=BaseContextFormatter)
        mock_generator = MagicMock(spec=BaseGenerator)
        mock_pipeline = MagicMock()

        mock_factory.create_retriever.return_value = mock_retriever
        mock_factory.create_formatter.return_value = mock_formatter
        mock_factory.create_generator.return_value = mock_generator
        mock_pipeline_cls.return_value = mock_pipeline

        # Configure and build the pipeline
        builder = (
            RAGPipelineBuilder()
            .with_standard_pipeline()
            .with_vector_retriever(vectorstore=mock_vectorstore, top_k=5)
            .with_simple_formatter()
            .with_llm_generator(llm=mock_llm)
            .with_config(max_tokens=1000)
        )

        result = builder.build()

        # Verify the components were created correctly
        mock_factory.create_retriever.assert_called_once()
        mock_factory.create_formatter.assert_called_once()
        mock_factory.create_generator.assert_called_once()

        # Verify the pipeline was created with the right components
        mock_pipeline_cls.assert_called_once_with(
            retriever=mock_retriever, formatter=mock_formatter, generator=mock_generator, max_tokens=1000
        )

        assert result == mock_pipeline

    @patch('src.llm_rag.rag.pipeline.pipeline_builder.rag_factory')
    @patch('src.llm_rag.rag.pipeline.pipeline_builder.ConversationalRAGPipeline')
    def test_build_conversational_pipeline(self, mock_pipeline_cls, mock_factory):
        """Test building a conversational pipeline."""
        # Setup mocks
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_llm = MagicMock(spec=BaseLanguageModel)
        mock_retriever = MagicMock(spec=BaseRetriever)
        mock_formatter = MagicMock(spec=BaseContextFormatter)
        mock_generator = MagicMock(spec=BaseGenerator)
        mock_pipeline = MagicMock()

        mock_factory.create_retriever.return_value = mock_retriever
        mock_factory.create_formatter.return_value = mock_formatter
        mock_factory.create_generator.return_value = mock_generator
        mock_pipeline_cls.return_value = mock_pipeline

        # Configure and build the pipeline
        builder = (
            RAGPipelineBuilder()
            .with_conversational_pipeline()
            .with_vector_retriever(vectorstore=mock_vectorstore, top_k=5)
            .with_markdown_formatter()
            .with_llm_generator(llm=mock_llm)
            .with_config(max_history_length=5)
        )

        result = builder.build()

        # Verify the components were created correctly
        mock_factory.create_retriever.assert_called_once()
        mock_factory.create_formatter.assert_called_once()
        mock_factory.create_generator.assert_called_once()

        # Verify the pipeline was created with the right components
        mock_pipeline_cls.assert_called_once_with(
            retriever=mock_retriever, formatter=mock_formatter, generator=mock_generator, max_history_length=5
        )

        assert result == mock_pipeline

    def test_build_with_custom_components(self):
        """Test building a pipeline with custom components."""
        # Setup mocks
        mock_retriever = MagicMock(spec=BaseRetriever)
        mock_formatter = MagicMock(spec=BaseContextFormatter)
        mock_generator = MagicMock(spec=BaseGenerator)

        # Use patch as context manager for the pipeline class
        with patch('src.llm_rag.rag.pipeline.pipeline_builder.RAGPipeline') as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline

            # Configure and build the pipeline
            builder = (
                RAGPipelineBuilder()
                .with_custom_retriever(mock_retriever)
                .with_custom_formatter(mock_formatter)
                .with_custom_generator(mock_generator)
            )

            result = builder.build()

            # Verify the pipeline was created with the right components
            mock_pipeline_cls.assert_called_once_with(
                retriever=mock_retriever, formatter=mock_formatter, generator=mock_generator
            )

            assert result == mock_pipeline

    def test_build_with_missing_retriever(self):
        """Test building a pipeline with missing retriever configuration."""
        builder = RAGPipelineBuilder()

        with pytest.raises(ValueError) as excinfo:
            builder.build()

        assert 'Retriever configuration is required' in str(excinfo.value)

    def test_build_with_missing_llm(self):
        """Test building a pipeline with missing language model."""
        builder = (
            RAGPipelineBuilder().with_vector_retriever(vectorstore=MagicMock(spec=VectorStore)).with_simple_formatter()
        )

        with pytest.raises(ValueError) as excinfo:
            builder.build()

        assert 'Language model (llm) is required' in str(excinfo.value)

    def test_build_with_missing_generator_config(self):
        """Test building a pipeline with missing generator configuration."""
        builder = (
            RAGPipelineBuilder().with_vector_retriever(vectorstore=MagicMock(spec=VectorStore)).with_simple_formatter()
        )

        # Set LLM but no generator config
        builder._llm = MagicMock(spec=BaseLanguageModel)

        with pytest.raises(ValueError) as excinfo:
            builder.build()

        assert 'Generator configuration is required' in str(excinfo.value)

    @patch('src.llm_rag.rag.pipeline.pipeline_builder.rag_factory')
    def test_build_with_exception(self, mock_factory):
        """Test exception handling during pipeline building."""
        # Setup to raise an exception
        mock_factory.create_retriever.side_effect = Exception('Component creation failed')

        builder = (
            RAGPipelineBuilder()
            .with_vector_retriever(vectorstore=MagicMock(spec=VectorStore))
            .with_simple_formatter()
            .with_llm_generator(llm=MagicMock(spec=BaseLanguageModel))
        )

        with pytest.raises(PipelineError) as excinfo:
            builder.build()

        assert 'Failed to build RAG pipeline' in str(excinfo.value)
        assert isinstance(excinfo.value.original_exception, Exception)
        assert 'Component creation failed' in str(excinfo.value.original_exception)
