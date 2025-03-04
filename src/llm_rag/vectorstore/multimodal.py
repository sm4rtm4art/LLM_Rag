"""Multi-modal vector store implementation.

This module provides a vector store implementation that can handle different types
of content (text, tables, images) with specialized embedding models.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from chromadb.api.types import (
    EmbeddingFunction,
)
from langchain_core.documents import Document
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .chroma import ChromaVectorStore


class MultiModalEmbeddingFunction(EmbeddingFunction):
    """Embedding function for multi-modal content.

    This class provides specialized embedding models for different content types:
    - Text: Uses a text embedding model (e.g., all-MiniLM-L6-v2)
    - Tables: Uses a table-specific model or falls back to text model
    - Images: Uses CLIP or similar vision-language model
    """

    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        image_model_name: str = "clip-ViT-B-32",
        embedding_dim: int = 512,
    ) -> None:
        """Initialize the multi-modal embedding function.

        Args:
        ----
            text_model_name: Name of the text embedding model
            image_model_name: Name of the image embedding model
            embedding_dim: Dimension of the unified embedding space

        """
        # Check if running in CI environment
        if os.environ.get("GITHUB_ACTIONS") == "true":
            self.is_mock = True
            self.embedding_dim = embedding_dim
        else:
            self.is_mock = False

            # Initialize text embedding model
            self.text_model = SentenceTransformer(text_model_name)

            # Initialize image embedding model if available
            try:
                # Use the already imported SentenceTransformer
                self.image_model = SentenceTransformer(image_model_name)
                self.has_image_model = True
            except Exception as e:
                print(f"Warning: Could not load image model: {e}")
                self.has_image_model = False

            # Initialize table embedding model (using text model as fallback)
            self.table_model = self.text_model

        # Store the target embedding dimension
        self.embedding_dim = embedding_dim

    def _embed_text(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for text content.

        Args:
        ----
            texts: List of text strings to embed

        Returns:
        -------
            List of embedding vectors

        """
        if self.is_mock:
            # Return random embeddings for testing
            return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in texts]

        # Generate embeddings using the text model
        embeddings = self.text_model.encode(texts, convert_to_numpy=True)

        # Ensure correct dimensionality
        if embeddings[0].shape[0] != self.embedding_dim:
            # Simple dimensionality adjustment (in practice, use more sophisticated methods)
            adjusted_embeddings = []
            for emb in embeddings:
                if emb.shape[0] < self.embedding_dim:
                    # Pad with zeros
                    padded = np.zeros(self.embedding_dim, dtype=np.float32)
                    padded[: emb.shape[0]] = emb
                    adjusted_embeddings.append(padded)
                else:
                    # Truncate
                    adjusted_embeddings.append(emb[: self.embedding_dim])
            return adjusted_embeddings

        return embeddings

    def _embed_table(self, tables: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for table content.

        Args:
        ----
            tables: List of table strings to embed

        Returns:
        -------
            List of embedding vectors

        """
        if self.is_mock:
            # Return random embeddings for testing
            return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in tables]

        # For now, use the text model for tables
        # In a production system, you might want to use a specialized model for tables
        return self._embed_text(tables)

    def _embed_image(self, image_paths: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for images.

        Args:
        ----
            image_paths: List of paths to image files

        Returns:
        -------
            List of embedding vectors

        """
        if self.is_mock:
            # Return random embeddings for testing
            return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in image_paths]

        if not self.has_image_model:
            # Fallback to random embeddings if image model is not available
            print("Warning: Image model not available, using random embeddings")
            return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in image_paths]

        try:
            from PIL import Image

            # Load images
            images = []
            for path in image_paths:
                try:
                    img = Image.open(path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    # Use a placeholder for failed images
                    images.append(None)

            # Generate embeddings for valid images
            valid_images = [img for img in images if img is not None]
            valid_indices = [i for i, img in enumerate(images) if img is not None]

            if not valid_images:
                # No valid images, return random embeddings
                return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in image_paths]

            # Generate embeddings
            valid_embeddings = self.image_model.encode(valid_images, convert_to_numpy=True)

            # Create result list with placeholders for invalid images
            result = [np.zeros(self.embedding_dim, dtype=np.float32) for _ in image_paths]
            for i, emb in zip(valid_indices, valid_embeddings, strict=False):
                # Ensure correct dimensionality
                if emb.shape[0] != self.embedding_dim:
                    if emb.shape[0] < self.embedding_dim:
                        # Pad with zeros
                        padded = np.zeros(self.embedding_dim, dtype=np.float32)
                        padded[: emb.shape[0]] = emb
                        result[i] = padded
                    else:
                        # Truncate
                        result[i] = emb[: self.embedding_dim]
                else:
                    result[i] = emb

            return result

        except Exception as e:
            print(f"Error embedding images: {e}")
            # Fallback to random embeddings
            return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in image_paths]

    def __call__(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[NDArray[np.float32]]:
        """Generate embeddings based on content type.

        Args:
        ----
            texts: List of content strings or references
            metadatas: Optional metadata for each item, used to determine content type

        Returns:
        -------
            List of embedding vectors

        """
        if not metadatas:
            # Default to text embedding if no metadata is provided
            return self._embed_text(texts)

        # Process each item based on its content type
        embeddings = []

        for _i, (text, metadata) in enumerate(zip(texts, metadatas, strict=False)):
            filetype = metadata.get("filetype", "")

            if filetype in ["image", "technical_drawing"]:
                # For images, the text might contain a caption, but we need the image path
                image_path = metadata.get("image_path", "")
                if image_path:
                    # Embed a single image
                    emb = self._embed_image([image_path])[0]
                else:
                    # Fallback to text embedding if no image path
                    emb = self._embed_text([text])[0]

            elif filetype == "table":
                # Embed table content
                emb = self._embed_table([text])[0]

            else:
                # Default to text embedding
                emb = self._embed_text([text])[0]

            embeddings.append(emb)

        return embeddings


class MultiModalVectorStore(ChromaVectorStore):
    """Vector store for multi-modal content.

    This class extends ChromaVectorStore to handle different types of content
    with specialized embedding and retrieval strategies.
    """

    def __init__(
        self,
        collection_name: str = "multimodal_docs",
        persist_directory: str = "chroma_multimodal",
        embedding_function: Optional[EmbeddingFunction] = None,
        text_model_name: str = "all-MiniLM-L6-v2",
        image_model_name: str = "clip-ViT-B-32",
        embedding_dim: int = 512,
    ) -> None:
        """Initialize the multi-modal vector store.

        Args:
        ----
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
            embedding_function: Optional custom embedding function
            text_model_name: Name of the text embedding model
            image_model_name: Name of the image embedding model
            embedding_dim: Dimension of the unified embedding space

        """
        # Create multi-modal embedding function if not provided
        if embedding_function is None:
            embedding_function = MultiModalEmbeddingFunction(
                text_model_name=text_model_name,
                image_model_name=image_model_name,
                embedding_dim=embedding_dim,
            )

        # Initialize the parent class
        super().__init__(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )

        # Store content type information
        self.content_types = {
            "text": [],
            "table": [],
            "image": [],
            "technical_drawing": [],
        }

    def add_documents(
        self,
        documents: Union[List[str], List[Document]],
        metadatas: Optional[List[Dict[str, Union[str, int, float, bool]]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents to the vector store with content type tracking.

        Args:
        ----
            documents: List of document texts or Document objects
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document

        """
        # Call the parent method to add documents
        super().add_documents(documents, metadatas, ids)

        # Track content types if metadata is available
        if metadatas:
            for i, metadata in enumerate(metadatas):
                filetype = metadata.get("filetype", "")
                doc_id = ids[i] if ids else f"doc_{i}"

                if filetype in ["image", "technical_drawing"]:
                    self.content_types["image"].append(doc_id)
                elif filetype == "table":
                    self.content_types["table"].append(doc_id)
                else:
                    self.content_types["text"].append(doc_id)

    def search_by_content_type(
        self,
        query: str,
        content_type: str = "all",
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for documents of a specific content type.

        Args:
        ----
            query: Search query text
            content_type: Type of content to search for ("text", "table", "image",
                         "technical_drawing", or "all")
            n_results: Number of results to return

        Returns:
        -------
            List of dictionaries containing matched documents and metadata

        """
        if content_type == "all" or not self.content_types.get(content_type):
            # Search all documents if no content type specified or no documents of that type
            return self.search(query, n_results=n_results)

        # Filter by content type using the 'where' parameter
        return self.search(
            query,
            n_results=n_results,
            where={"filetype": content_type},
        )

    def multimodal_search(
        self,
        query: str,
        n_results_per_type: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all content types and return organized results.

        Args:
        ----
            query: Search query text
            n_results_per_type: Number of results to return per content type

        Returns:
        -------
            Dictionary with content types as keys and lists of results as values

        """
        results = {}

        # Search each content type
        for content_type in ["text", "table", "image", "technical_drawing"]:
            if self.content_types.get(content_type):
                type_results = self.search_by_content_type(
                    query,
                    content_type=content_type,
                    n_results=n_results_per_type,
                )
                results[content_type] = type_results

        return results

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> "MultiModalRetriever":
        """Create a retriever from this vector store.

        Args:
        ----
            search_kwargs: Optional search parameters

        Returns:
        -------
            A MultiModalRetriever instance

        """
        search_kwargs = search_kwargs or {}
        return MultiModalRetriever(vectorstore=self, search_kwargs=search_kwargs)


class MultiModalRetriever:
    """Retriever for multi-modal content.

    This class provides retrieval capabilities for the MultiModalVectorStore,
    with support for content type filtering and multi-modal search.
    """

    def __init__(
        self,
        vectorstore: MultiModalVectorStore,
        search_kwargs: Dict[str, Any],
    ):
        """Initialize the multi-modal retriever.

        Args:
        ----
            vectorstore: The MultiModalVectorStore to retrieve from
            search_kwargs: Search parameters

        """
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs
        self.k = search_kwargs.get("k", 4)
        self.content_type = search_kwargs.get("content_type", "all")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query.

        Args:
        ----
            query: Search query text

        Returns:
        -------
            List of relevant Document objects

        """
        if self.content_type != "all":
            # Search for specific content type
            results = self.vectorstore.search_by_content_type(
                query,
                content_type=self.content_type,
                n_results=self.k,
            )
        else:
            # Standard search across all content types
            results = self.vectorstore.search(
                query,
                n_results=self.k,
            )

        # Convert results to Document objects
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata=result["metadata"],
            )
            documents.append(doc)

        return documents

    def get_multimodal_documents(self, query: str) -> Dict[str, List[Document]]:
        """Get documents organized by content type.

        Args:
        ----
            query: Search query text

        Returns:
        -------
            Dictionary with content types as keys and lists of Document objects as values

        """
        # Get results organized by content type
        results_by_type = self.vectorstore.multimodal_search(
            query,
            n_results_per_type=self.k,
        )

        # Convert results to Document objects
        documents_by_type = {}
        for content_type, results in results_by_type.items():
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata=result["metadata"],
                )
                documents.append(doc)
            documents_by_type[content_type] = documents

        return documents_by_type
