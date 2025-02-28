"""Embeddings module for the llm-rag package.

This module provides utilities for generating text embeddings using various models.
"""
from typing import Any, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """A wrapper for sentence-transformers embedding models.

    This class provides a simple interface for generating embeddings from text
    using pretrained sentence-transformer models.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        """Initialize the embedding model.

        Args:
        ----
            model_name: Name of the sentence-transformers model to use.
                Defaults to "all-MiniLM-L6-v2".
            device: Device to use for the model (cpu, cuda).
                Defaults to None (auto-detect).

        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: List[str]) -> List[NDArray[Union[np.float32, np.int32]]]:
        """Generate embeddings for input texts (compatibility with EmbeddingFunction protocol).

        Args:
        ----
            input: List of texts to generate embeddings for.

        Returns:
        -------
            List of numpy arrays containing embeddings.

        """
        embeddings = self.model.encode(input, normalize_embeddings=True, batch_size=32)
        # Convert to list of numpy arrays
        if len(input) == 1:
            return [embeddings]
        return [embedding for embedding in embeddings]

    def embed_with_retries(self, input: List[str], **retry_kwargs: Any) -> List[NDArray[Union[np.float32, np.int32]]]:
        """Embed with retries for robustness (compatibility with EmbeddingFunction protocol).

        Args:
        ----
            input: List of texts to generate embeddings for.
            retry_kwargs: Additional retry parameters.

        Returns:
        -------
            List of numpy arrays containing embeddings.

        """
        # Simple implementation without actual retries for now
        return self.__call__(input)

    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a single text query.

        Args:
        ----
            text: The text to embed.

        Returns:
        -------
            A list of floating point numbers representing the text embedding.

        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        # Explicitly convert to list of float to satisfy mypy
        return [float(x) for x in embedding.tolist()]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
        ----
            documents: List of text documents to embed.

        Returns:
        -------
            A list of embeddings, where each embedding is a list of float
            values.

        """
        embeddings = self.model.encode(documents, normalize_embeddings=True, batch_size=32)
        return [emb.tolist() for emb in embeddings]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings produced by this model.

        Returns
        -------
            The dimension of the embeddings as an integer.

        """
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            return 0
        # Explicitly convert to int to satisfy mypy
        return int(dim)
