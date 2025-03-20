"""Factory pattern implementation for RAG pipeline components.

This module provides a structured Factory pattern implementation for creating
and registering pipeline components like retrievers, formatters, and generators.
"""

from typing import Dict, List, Type, TypeVar, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from llm_rag.utils.errors import PipelineError
from llm_rag.utils.logging import get_logger

from .context import BaseContextFormatter, MarkdownContextFormatter, SimpleContextFormatter
from .generation import BaseGenerator, LLMGenerator, TemplatedGenerator
from .retrieval import BaseRetriever, HybridRetriever, VectorStoreRetriever

logger = get_logger(__name__)

# Type variables for component types
R = TypeVar("R", bound=BaseRetriever)
F = TypeVar("F", bound=BaseContextFormatter)
G = TypeVar("G", bound=BaseGenerator)


class ComponentFactory:
    """Factory for creating RAG pipeline components.

    This factory allows registering custom component implementations
    and creating components with the appropriate configuration.
    """

    def __init__(self):
        """Initialize the component factory with default implementations."""
        # Initialize registries for each component type
        self._retriever_registry: Dict[str, Type[BaseRetriever]] = {
            "vector": VectorStoreRetriever,
            "hybrid": HybridRetriever,
        }

        self._formatter_registry: Dict[str, Type[BaseContextFormatter]] = {
            "simple": SimpleContextFormatter,
            "markdown": MarkdownContextFormatter,
            "html": MarkdownContextFormatter,  # Alias for markdown
        }

        self._generator_registry: Dict[str, Type[BaseGenerator]] = {
            "llm": LLMGenerator,
            "templated": TemplatedGenerator,
        }

        logger.info(
            f"ComponentFactory initialized with "
            f"{len(self._retriever_registry)} retrievers, "
            f"{len(self._formatter_registry)} formatters, "
            f"{len(self._generator_registry)} generators"
        )

    # Registration methods
    def register_retriever(self, name: str, retriever_class: Type[BaseRetriever]) -> None:
        """Register a custom retriever implementation.

        Args:
            name: Unique name for the retriever type
            retriever_class: Retriever class to register

        Raises:
            ValueError: If the name is already registered or class doesn't inherit from BaseRetriever

        """
        if name in self._retriever_registry:
            raise ValueError(f"Retriever type '{name}' is already registered")

        if not issubclass(retriever_class, BaseRetriever):
            raise ValueError(f"Class {retriever_class.__name__} must inherit from BaseRetriever")

        self._retriever_registry[name] = retriever_class
        logger.info(f"Registered retriever type '{name}': {retriever_class.__name__}")

    def register_formatter(self, name: str, formatter_class: Type[BaseContextFormatter]) -> None:
        """Register a custom formatter implementation.

        Args:
            name: Unique name for the formatter type
            formatter_class: Formatter class to register

        Raises:
            ValueError: If the name is already registered or class doesn't inherit from BaseContextFormatter

        """
        if name in self._formatter_registry:
            raise ValueError(f"Formatter type '{name}' is already registered")

        if not issubclass(formatter_class, BaseContextFormatter):
            raise ValueError(f"Class {formatter_class.__name__} must inherit from BaseContextFormatter")

        self._formatter_registry[name] = formatter_class
        logger.info(f"Registered formatter type '{name}': {formatter_class.__name__}")

    def register_generator(self, name: str, generator_class: Type[BaseGenerator]) -> None:
        """Register a custom generator implementation.

        Args:
            name: Unique name for the generator type
            generator_class: Generator class to register

        Raises:
            ValueError: If the name is already registered or class doesn't inherit from BaseGenerator

        """
        if name in self._generator_registry:
            raise ValueError(f"Generator type '{name}' is already registered")

        if not issubclass(generator_class, BaseGenerator):
            raise ValueError(f"Class {generator_class.__name__} must inherit from BaseGenerator")

        self._generator_registry[name] = generator_class
        logger.info(f"Registered generator type '{name}': {generator_class.__name__}")

    # Factory methods
    def create_retriever(
        self,
        retriever_type: str = "vector",
        source: Union[VectorStore, BaseRetriever, List[BaseRetriever]] = None,
        **kwargs,
    ) -> BaseRetriever:
        """Create a retriever instance.

        Args:
            retriever_type: Type of retriever to create
            source: Source for retrieval (e.g., vectorstore)
            **kwargs: Additional configuration parameters

        Returns:
            Configured retriever instance

        Raises:
            ValueError: If retriever_type is not registered or required parameters are missing

        """
        if retriever_type not in self._retriever_registry:
            raise ValueError(
                f"Unknown retriever type: '{retriever_type}'. "
                f"Available types: {', '.join(self._retriever_registry.keys())}"
            )

        retriever_class = self._retriever_registry[retriever_type]

        try:
            if retriever_type == "vector":
                if not source or not isinstance(source, VectorStore):
                    raise ValueError("Vector retriever requires a VectorStore source")

                top_k = kwargs.get("top_k", 5)
                return retriever_class(vectorstore=source, top_k=top_k)

            elif retriever_type == "hybrid":
                if not source or not isinstance(source, list):
                    raise ValueError("Hybrid retriever requires a list of retrievers")

                weights = kwargs.get("weights", None)
                return retriever_class(retrievers=source, weights=weights)

            else:
                # For custom registered retrievers, pass all kwargs
                return retriever_class(**kwargs)

        except Exception as e:
            logger.error(f"Error creating {retriever_type} retriever: {str(e)}")
            raise PipelineError(f"Failed to create retriever: {str(e)}", original_exception=e) from e

    def create_formatter(self, formatter_type: str = "simple", **kwargs) -> BaseContextFormatter:
        """Create a context formatter instance.

        Args:
            formatter_type: Type of formatter to create
            **kwargs: Configuration parameters

        Returns:
            Configured formatter instance

        Raises:
            ValueError: If formatter_type is not registered

        """
        if formatter_type not in self._formatter_registry:
            raise ValueError(
                f"Unknown formatter type: '{formatter_type}'. "
                f"Available types: {', '.join(self._formatter_registry.keys())}"
            )

        formatter_class = self._formatter_registry[formatter_type]

        try:
            include_metadata = kwargs.get("include_metadata", True)
            max_length = kwargs.get("max_length", None)

            if formatter_type == "simple":
                separator = kwargs.get("separator", "\n\n")
                return formatter_class(
                    include_metadata=include_metadata,
                    max_length=max_length,
                    separator=separator,
                )
            else:
                return formatter_class(
                    include_metadata=include_metadata,
                    max_length=max_length,
                )

        except Exception as e:
            logger.error(f"Error creating {formatter_type} formatter: {str(e)}")
            raise PipelineError(f"Failed to create formatter: {str(e)}", original_exception=e) from e

    def create_generator(self, generator_type: str = "llm", llm: BaseLanguageModel = None, **kwargs) -> BaseGenerator:
        """Create a response generator instance.

        Args:
            generator_type: Type of generator to create
            llm: Language model to use for generation
            **kwargs: Configuration parameters

        Returns:
            Configured generator instance

        Raises:
            ValueError: If generator_type is not registered or required parameters are missing

        """
        if generator_type not in self._generator_registry:
            raise ValueError(
                f"Unknown generator type: '{generator_type}'. "
                f"Available types: {', '.join(self._generator_registry.keys())}"
            )

        if not llm:
            raise ValueError("Language model (llm) is required for generator creation")

        generator_class = self._generator_registry[generator_type]

        try:
            prompt_template = kwargs.get("prompt_template", None)
            apply_anti_hallucination = kwargs.get("apply_anti_hallucination", True)

            if generator_type == "llm":
                return generator_class(
                    llm=llm,
                    prompt_template=prompt_template,
                    apply_anti_hallucination=apply_anti_hallucination,
                )

            elif generator_type == "templated":
                templates = kwargs.get("templates")
                if not templates:
                    raise ValueError("Templates dictionary is required for templated generator")

                default_template = kwargs.get("default_template", "default")
                return generator_class(
                    llm=llm,
                    templates=templates,
                    default_template=default_template,
                    apply_anti_hallucination=apply_anti_hallucination,
                )

            else:
                # For custom registered generators, pass all kwargs
                return generator_class(llm=llm, **kwargs)

        except Exception as e:
            logger.error(f"Error creating {generator_type} generator: {str(e)}")
            raise PipelineError(f"Failed to create generator: {str(e)}", original_exception=e) from e


# Create a global factory instance for easy access
factory = ComponentFactory()
