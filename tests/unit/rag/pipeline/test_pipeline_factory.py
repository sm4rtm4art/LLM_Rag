"""Unit tests for the pipeline_factory.py module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from llm_rag.utils.errors import PipelineError
from src.llm_rag.rag.pipeline.pipeline_factory import PipelineType, RagPipelineFactory, create_pipeline


class TestPipelineFactory:
    """Test cases for the pipeline_factory.py module."""

    def test_module_exports(self):
        """Test that the module exports all expected classes and functions."""
        # Essential exports from directly imported module
        assert hasattr(PipelineType, "STANDARD")
        assert hasattr(PipelineType, "CONVERSATIONAL")
        assert callable(create_pipeline)

    @patch("src.llm_rag.rag.pipeline.pipeline_factory.RagPipelineFactory")
    def test_create_pipeline_calls_factory(self, mock_factory_cls):
        """Test that create_pipeline delegates to the RagPipelineFactory."""
        # Setup
        mock_factory = MagicMock()
        mock_factory_cls.return_value = mock_factory
        mock_factory.create.return_value = "mock_pipeline"

        # Call the function
        result = create_pipeline(pipeline_type="standard", config={"key": "value"})

        # Verify factory was created and called correctly
        mock_factory_cls.assert_called_once_with({"key": "value"})
        mock_factory.create.assert_called_once_with("standard")
        assert result == "mock_pipeline"

    def test_pipeline_type_enum(self):
        """Test the PipelineType enum."""
        # Check that enum values are strings
        assert isinstance(PipelineType.STANDARD.value, str)
        assert isinstance(PipelineType.CONVERSATIONAL.value, str)

        # Check enum string values
        assert PipelineType.STANDARD.value == "standard"
        assert PipelineType.CONVERSATIONAL.value == "conversational"

    def test_factory_init_with_empty_config(self):
        """Test factory initialization with empty config."""
        factory = RagPipelineFactory()
        assert factory.config == {}

    def test_factory_init_with_config(self):
        """Test factory initialization with a config dictionary."""
        config = {"param1": "value1", "param2": "value2"}
        factory = RagPipelineFactory(config)
        assert factory.config == config

    @patch("src.llm_rag.rag.pipeline.pipeline_factory.RAGPipeline")
    def test_create_standard_pipeline(self, mock_pipeline_cls):
        """Test creating a standard pipeline."""
        # Setup mocks
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_llm = MagicMock(spec=BaseLanguageModel)

        # Create factory and pipeline
        factory = RagPipelineFactory({"vectorstore": mock_vectorstore, "llm": mock_llm, "top_k": 5})
        result = factory.create(PipelineType.STANDARD)

        # Verify the pipeline was created correctly
        mock_pipeline_cls.assert_called_once_with(vectorstore=mock_vectorstore, llm=mock_llm, top_k=5)
        assert result == mock_pipeline

    @patch("src.llm_rag.rag.pipeline.pipeline_factory.ConversationalRAGPipeline")
    def test_create_conversational_pipeline(self, mock_pipeline_cls):
        """Test creating a conversational pipeline."""
        # Setup mocks
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_llm = MagicMock(spec=BaseLanguageModel)

        # Create factory and pipeline
        factory = RagPipelineFactory({"vectorstore": mock_vectorstore, "llm": mock_llm, "max_history_length": 3})
        result = factory.create(PipelineType.CONVERSATIONAL)

        # Verify the pipeline was created correctly
        mock_pipeline_cls.assert_called_once_with(vectorstore=mock_vectorstore, llm=mock_llm, max_history_length=3)
        assert result == mock_pipeline

    def test_create_with_string_pipeline_type(self):
        """Test creating a pipeline with string pipeline type."""
        # Setup
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_llm = MagicMock(spec=BaseLanguageModel)
        config = {"vectorstore": mock_vectorstore, "llm": mock_llm}

        # Use patch as context manager
        with patch("src.llm_rag.rag.pipeline.pipeline_factory.RAGPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline

            # Create factory and pipeline
            factory = RagPipelineFactory(config)
            result = factory.create("standard")

            # Verify the pipeline was created correctly
            mock_pipeline_cls.assert_called_once_with(vectorstore=mock_vectorstore, llm=mock_llm)
            assert result == mock_pipeline

    def test_create_with_invalid_pipeline_type(self):
        """Test creating a pipeline with invalid pipeline type."""
        factory = RagPipelineFactory({"vectorstore": MagicMock(), "llm": MagicMock()})

        # Verify that an invalid type raises ValueError
        with pytest.raises(ValueError) as excinfo:
            factory.create("invalid_type")

        # Check error message
        assert "Unknown pipeline type" in str(excinfo.value)

    def test_create_without_vectorstore(self):
        """Test creating a pipeline without vectorstore."""
        factory = RagPipelineFactory({"llm": MagicMock()})

        # Verify that missing vectorstore raises ValueError
        with pytest.raises(ValueError) as excinfo:
            factory.create(PipelineType.STANDARD)

        # Check error message
        assert "vectorstore is required" in str(excinfo.value)

    def test_create_without_llm(self):
        """Test creating a pipeline without llm."""
        factory = RagPipelineFactory({"vectorstore": MagicMock()})

        # Verify that missing llm raises ValueError
        with pytest.raises(ValueError) as excinfo:
            factory.create(PipelineType.STANDARD)

        # Check error message
        assert "language model (llm) is required" in str(excinfo.value)

    @patch("src.llm_rag.rag.pipeline.pipeline_factory.RAGPipeline")
    def test_create_pipeline_exception_handling(self, mock_pipeline_cls):
        """Test exception handling during pipeline creation."""
        # Setup to raise an exception
        mock_pipeline_cls.side_effect = Exception("Pipeline creation failed")
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_llm = MagicMock(spec=BaseLanguageModel)

        factory = RagPipelineFactory({"vectorstore": mock_vectorstore, "llm": mock_llm})

        # Verify that the exception is wrapped in PipelineError
        with pytest.raises(PipelineError) as excinfo:
            factory.create(PipelineType.STANDARD)

        # Check error details
        assert "Failed to create standard pipeline" in str(excinfo.value)
        assert isinstance(excinfo.value.original_exception, Exception)
        assert "Pipeline creation failed" in str(excinfo.value.original_exception)

    @patch("src.llm_rag.rag.pipeline.pipeline_factory.RagPipelineFactory")
    def test_create_pipeline_with_enum_type(self, mock_factory_cls):
        """Test create_pipeline with PipelineType enum."""
        # Setup
        mock_factory = MagicMock()
        mock_factory_cls.return_value = mock_factory
        mock_factory.create.return_value = "mock_pipeline"

        # Call the function with enum value
        result = create_pipeline(pipeline_type=PipelineType.CONVERSATIONAL, config={"key": "value"})

        # Verify factory was created and called correctly
        mock_factory_cls.assert_called_once_with({"key": "value"})
        mock_factory.create.assert_called_once_with(PipelineType.CONVERSATIONAL)
        assert result == "mock_pipeline"

    @patch("src.llm_rag.rag.pipeline.pipeline_factory.RagPipelineFactory")
    def test_create_pipeline_defaults(self, mock_factory_cls):
        """Test create_pipeline default arguments."""
        # Setup
        mock_factory = MagicMock()
        mock_factory_cls.return_value = mock_factory
        mock_factory.create.return_value = "mock_pipeline"

        # Call the function with no arguments
        result = create_pipeline()

        # Verify default values are used
        mock_factory_cls.assert_called_once_with(None)
        mock_factory.create.assert_called_once_with(PipelineType.STANDARD)
        assert result == "mock_pipeline"
