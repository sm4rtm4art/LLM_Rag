"""Base vector store interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """Base class for vector store implementations."""

    @abstractmethod
    def add_documents(
        self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the vector store.

        Args:
        ----
            documents: List of document texts to add
            metadatas: Optional metadata for each document

        """
        pass

    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
        ----
            query: Search query text
            n_results: Number of results to return

        Returns:
        -------
            List of dictionaries containing matched documents and metadata

        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass
