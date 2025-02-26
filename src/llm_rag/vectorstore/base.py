"""Vector store abstractions."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store."""
        pass 