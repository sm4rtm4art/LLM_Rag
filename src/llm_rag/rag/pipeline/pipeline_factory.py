"""Factory module for creating complete RAG pipelines.

This module provides a factory for creating different types of RAG pipelines
(standard, conversational) with appropriate components and configuration.
"""

import enum
from typing import Any, Dict, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from llm_rag.utils.errors import PipelineError
from llm_rag.utils.logging import get_logger

from .base import RAGPipeline
from .conversational import ConversationalRAGPipeline

logger = get_logger(__name__)


class PipelineType(enum.Enum):
    """Enum for pipeline types."""

    STANDARD = 'standard'
    CONVERSATIONAL = 'conversational'


class RagPipelineFactory:
    """Factory for creating complete RAG pipelines.

    This factory handles the creation of appropriate pipeline types with
    all necessary components configured.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline factory with configuration.

        Args:
            config: Configuration dictionary for pipeline components

        """
        self.config = config or {}
        logger.info(f'RagPipelineFactory initialized with {len(self.config)} configuration items')

    def create(self, pipeline_type: Union[str, PipelineType]) -> Union[RAGPipeline, ConversationalRAGPipeline]:
        """Create a pipeline of the specified type.

        Args:
            pipeline_type: Type of pipeline to create ("standard" or
                "conversational")

        Returns:
            Configured pipeline instance

        Raises:
            ValueError: If pipeline_type is not recognized
            PipelineError: If pipeline creation fails

        """
        # Convert enum to string if needed
        if isinstance(pipeline_type, PipelineType):
            pipeline_type = pipeline_type.value

        # Validate pipeline type
        if pipeline_type not in [
            PipelineType.STANDARD.value,
            PipelineType.CONVERSATIONAL.value,
        ]:
            raise ValueError(
                f"Unknown pipeline type: '{pipeline_type}'. "
                f'Available types: {PipelineType.STANDARD.value}, '
                f'{PipelineType.CONVERSATIONAL.value}'
            )

        try:
            # Extract key components from config
            vectorstore = self.config.get('vectorstore')
            if not vectorstore or not isinstance(vectorstore, VectorStore):
                raise ValueError('A vectorstore is required for pipeline creation')

            llm = self.config.get('llm')
            if not llm or not isinstance(llm, BaseLanguageModel):
                raise ValueError('A language model (llm) is required for pipeline creation')

            # Create the appropriate pipeline type
            if pipeline_type == PipelineType.STANDARD.value:
                logger.info('Creating standard RAG pipeline')
                return RAGPipeline(vectorstore=vectorstore, llm=llm, **self.config)
            else:  # pipeline_type == PipelineType.CONVERSATIONAL.value
                logger.info('Creating conversational RAG pipeline')
                return ConversationalRAGPipeline(vectorstore=vectorstore, llm=llm, **self.config)

        except Exception as e:
            logger.error(f'Error creating {pipeline_type} pipeline: {str(e)}')
            raise PipelineError(
                f'Failed to create {pipeline_type} pipeline: {str(e)}',
                original_exception=e,
            ) from e


def create_pipeline(
    pipeline_type: Union[str, PipelineType] = PipelineType.STANDARD,
    config: Optional[Dict[str, Any]] = None,
) -> Union[RAGPipeline, ConversationalRAGPipeline]:
    """Create a RAG pipeline of the specified type.

    This is a convenience function that creates a pipeline factory
    and delegates to it.

    Args:
        pipeline_type: Type of pipeline to create
        config: Configuration dictionary for pipeline components

    Returns:
        Configured pipeline instance

    Raises:
        ValueError: If pipeline_type is not recognized or configuration invalid
        PipelineError: If pipeline creation fails

    """
    factory = RagPipelineFactory(config)
    return factory.create(pipeline_type)
