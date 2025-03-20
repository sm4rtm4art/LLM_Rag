"""Builder pattern implementation for RAG pipelines.

This module provides a flexible builder for constructing RAG pipelines
with various components and configurations.
"""

from typing import Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

from llm_rag.utils.errors import PipelineError
from llm_rag.utils.logging import get_logger

from .base import RAGPipeline
from .component_factory import rag_factory
from .context import BaseContextFormatter
from .conversational import ConversationalRAGPipeline
from .generation import BaseGenerator
from .retrieval import BaseRetriever

logger = get_logger(__name__)


class RAGPipelineBuilder:
    """Builder for constructing RAG pipelines with fluent interface.

    This builder simplifies the process of creating and configuring
    RAG pipelines with different components.
    """

    def __init__(self):
        """Initialize an empty pipeline builder."""
        # Pipeline type and components
        self._pipeline_type = "standard"  # or "conversational"
        self._retriever = None
        self._formatter = None
        self._generator = None

        # Component configurations
        self._retriever_config = {}
        self._formatter_config = {}
        self._generator_config = {}

        # Pipeline configuration
        self._pipeline_config = {}
        self._template = None
        self._llm = None

        logger.debug("Initialized RAGPipelineBuilder")

    # Pipeline type setters
    def with_standard_pipeline(self) -> "RAGPipelineBuilder":
        """Configure as a standard RAG pipeline.

        Returns:
            Self for method chaining

        """
        self._pipeline_type = "standard"
        return self

    def with_conversational_pipeline(self) -> "RAGPipelineBuilder":
        """Configure as a conversational RAG pipeline.

        Returns:
            Self for method chaining

        """
        self._pipeline_type = "conversational"
        return self

    # Component configuration methods
    def with_vector_retriever(self, vectorstore: VectorStore, top_k: int = 5) -> "RAGPipelineBuilder":
        """Configure a vector store retriever.

        Args:
            vectorstore: Vector store for document retrieval
            top_k: Number of documents to retrieve

        Returns:
            Self for method chaining

        """
        self._retriever_config = {
            "type": "vector",
            "source": vectorstore,
            "top_k": top_k,
        }
        return self

    def with_hybrid_retriever(
        self, retrievers: List[BaseRetriever], weights: Optional[List[float]] = None
    ) -> "RAGPipelineBuilder":
        """Configure a hybrid retriever.

        Args:
            retrievers: List of retrievers to combine
            weights: Optional weights for each retriever

        Returns:
            Self for method chaining

        """
        self._retriever_config = {
            "type": "hybrid",
            "source": retrievers,
            "weights": weights,
        }
        return self

    def with_custom_retriever(self, retriever: BaseRetriever) -> "RAGPipelineBuilder":
        """Configure a custom retriever.

        Args:
            retriever: Pre-configured retriever instance

        Returns:
            Self for method chaining

        """
        self._retriever = retriever
        return self

    def with_simple_formatter(
        self, include_metadata: bool = True, max_length: Optional[int] = None
    ) -> "RAGPipelineBuilder":
        """Configure a simple context formatter.

        Args:
            include_metadata: Whether to include document metadata
            max_length: Maximum length of formatted context

        Returns:
            Self for method chaining

        """
        self._formatter_config = {
            "type": "simple",
            "include_metadata": include_metadata,
            "max_length": max_length,
        }
        return self

    def with_markdown_formatter(
        self, include_metadata: bool = True, max_length: Optional[int] = None
    ) -> "RAGPipelineBuilder":
        """Configure a markdown context formatter.

        Args:
            include_metadata: Whether to include document metadata
            max_length: Maximum length of formatted context

        Returns:
            Self for method chaining

        """
        self._formatter_config = {
            "type": "markdown",
            "include_metadata": include_metadata,
            "max_length": max_length,
        }
        return self

    def with_custom_formatter(self, formatter: BaseContextFormatter) -> "RAGPipelineBuilder":
        """Configure a custom formatter.

        Args:
            formatter: Pre-configured formatter instance

        Returns:
            Self for method chaining

        """
        self._formatter = formatter
        return self

    def with_llm_generator(
        self,
        llm: BaseLanguageModel,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        apply_anti_hallucination: bool = True,
    ) -> "RAGPipelineBuilder":
        """Configure an LLM generator.

        Args:
            llm: Language model for generation
            prompt_template: Optional custom prompt template
            apply_anti_hallucination: Whether to apply anti-hallucination techniques

        Returns:
            Self for method chaining

        """
        self._llm = llm
        self._generator_config = {
            "type": "llm",
            "prompt_template": prompt_template,
            "apply_anti_hallucination": apply_anti_hallucination,
        }
        return self

    def with_templated_generator(
        self,
        llm: BaseLanguageModel,
        templates: Dict[str, Union[str, PromptTemplate]],
        default_template: str = "default",
        apply_anti_hallucination: bool = True,
    ) -> "RAGPipelineBuilder":
        """Configure a templated generator with multiple templates.

        Args:
            llm: Language model for generation
            templates: Dictionary of named templates
            default_template: Name of the default template
            apply_anti_hallucination: Whether to apply anti-hallucination techniques

        Returns:
            Self for method chaining

        """
        self._llm = llm
        self._generator_config = {
            "type": "templated",
            "templates": templates,
            "default_template": default_template,
            "apply_anti_hallucination": apply_anti_hallucination,
        }
        return self

    def with_custom_generator(self, generator: BaseGenerator) -> "RAGPipelineBuilder":
        """Configure a custom generator.

        Args:
            generator: Pre-configured generator instance

        Returns:
            Self for method chaining

        """
        self._generator = generator
        return self

    # Pipeline configuration methods
    def with_config(self, **kwargs) -> "RAGPipelineBuilder":
        """Set additional pipeline configuration.

        Args:
            **kwargs: Configuration parameters

        Returns:
            Self for method chaining

        """
        self._pipeline_config.update(kwargs)
        return self

    # Build method to create the pipeline
    def build(self) -> Union[RAGPipeline, ConversationalRAGPipeline]:
        """Build the configured RAG pipeline.

        Returns:
            Configured RAG pipeline instance

        Raises:
            PipelineError: If the pipeline cannot be built with the current configuration

        """
        try:
            # Create components if not already provided
            if not self._retriever:
                if not self._retriever_config:
                    raise ValueError("Retriever configuration is required")

                retriever_type = self._retriever_config.pop("type")
                source = self._retriever_config.pop("source", None)
                self._retriever = rag_factory.create_retriever(
                    retriever_type=retriever_type, source=source, **self._retriever_config
                )

            if not self._formatter:
                formatter_type = self._formatter_config.pop("type", "simple")
                self._formatter = rag_factory.create_formatter(formatter_type=formatter_type, **self._formatter_config)

            if not self._generator:
                if not self._llm:
                    raise ValueError("Language model (llm) is required")
                elif not self._generator_config:
                    raise ValueError("Generator configuration is required")

                generator_type = self._generator_config.pop("type", "llm")
                self._generator = rag_factory.create_generator(
                    generator_type=generator_type, llm=self._llm, **self._generator_config
                )

            # Create appropriate pipeline type
            if self._pipeline_type == "conversational":
                pipeline = ConversationalRAGPipeline(
                    retriever=self._retriever,
                    formatter=self._formatter,
                    generator=self._generator,
                    **self._pipeline_config,
                )
                logger.info("Built ConversationalRAGPipeline")
            else:
                pipeline = RAGPipeline(
                    retriever=self._retriever,
                    formatter=self._formatter,
                    generator=self._generator,
                    **self._pipeline_config,
                )
                logger.info("Built RAGPipeline")

            return pipeline

        except ValueError as e:
            # Re-raise ValueError for missing components or configuration
            logger.error(f"Error building RAG pipeline: {str(e)}")
            raise
        except Exception as e:
            # Wrap other errors in PipelineError
            logger.error(f"Error building RAG pipeline: {str(e)}")
            raise PipelineError(f"Failed to build RAG pipeline: {str(e)}", original_exception=e) from e


# Create a builder function for easy access
def create_rag_pipeline() -> RAGPipelineBuilder:
    """Create a new RAG pipeline builder.

    Returns:
        Configured RAG pipeline builder

    """
    return RAGPipelineBuilder()
