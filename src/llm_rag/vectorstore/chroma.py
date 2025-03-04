import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import chromadb
import numpy as np
from chromadb.api import ClientAPI, Collection
from chromadb.api.types import (
    EmbeddingFunction,
    Metadata,
)
from chromadb.config import Settings
from langchain_core.documents import Document
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .base import VectorStore


class EmbeddingFunctionWrapper(EmbeddingFunction):
    """Wrapper for SentenceTransformer to match ChromaDB's interface."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding function wrapper.

        Args:
        ----
            model_name: Name of the sentence transformer model to use.
                      Defaults to 'all-MiniLM-L6-v2'.

        """
        # Check if running in CI environment (GitHub Actions)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            self.is_mock = True
            self.embedding_dim = 384  # Standard dimension for all-MiniLM-L6-v2
        else:
            self.is_mock = False
            self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[NDArray[Union[np.float32, np.int32]]]:
        """Generate embeddings for input texts.

        Args:
        ----
            input: List of texts to generate embeddings for.

        Returns:
        -------
            List of embedding vectors as numpy arrays.

        """
        if self.is_mock:
            # Return mock embeddings with the correct dimensions
            return [np.zeros(self.embedding_dim, dtype=np.float32) + 0.1 for _ in input]

        # Get embeddings as numpy array
        embeddings = self.model.encode(input, convert_to_tensor=False, normalize_embeddings=True)

        # Ensure correct format and type
        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            return [embeddings]
        return [embedding for embedding in embeddings]


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(
        self,
        collection_name: str = "llm_rag_docs",
        persist_directory: str = "chroma_data",
        embedding_function: Optional[EmbeddingFunction] = None,
    ) -> None:
        """Initialize ChromaDB client and collection.

        Args:
        ----
            collection_name: Name of the collection to use
            persist_directory: Directory to persist vector store data
            embedding_function: Optional custom embedding function

        """
        self._persist_directory = persist_directory
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client: ClientAPI = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

        # Create default embedding function if none provided
        if embedding_function is None:
            embedding_function = EmbeddingFunctionWrapper()

        # Get or create collection
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: Union[List[str], List[Document]],
        metadatas: Optional[List[Dict[str, Union[str, int, float, bool]]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents to ChromaDB.

        Args:
        ----
            documents: List of document texts or Document objects
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document

        """
        # Handle LangChain Document objects
        if documents and isinstance(documents[0], Document):
            doc_strings = [doc.page_content for doc in documents]

            # If metadatas is not provided, use the metadata from Document objects
            if metadatas is None:
                metadatas = [doc.metadata for doc in documents]
        else:
            # Handle string documents
            doc_strings = cast(List[str], documents)

        if ids is None:
            ids = [str(i) for i in range(len(doc_strings))]

        self.collection.add(
            documents=doc_strings,
            metadatas=cast(Optional[List[Metadata]], metadatas),
            ids=ids,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        search_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
        ----
            query: Search query text
            n_results: Number of results to return
            where: Optional filter conditions on metadata
            where_document: Optional filter conditions on document content
            search_type: Optional search type (similarity, mmr, etc.) - not used in this implementation

        Returns:
        -------
            List of dictionaries containing matched documents and metadata

        """
        # Ignoring search_type as our implementation doesn't need it
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"] and results["metadatas"][0]
                        else None,
                        "distance": results["distances"][0][i]
                        if "distances" in results and results["distances"]
                        else None,
                    }
                )

        return formatted_results

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection.name)
        except ValueError as e:
            if "does not exist" not in str(e):
                raise

    def get_all_documents(self) -> List[Document]:
        """Get all documents from the collection.

        Returns
        -------
            List of all documents as LangChain Document objects

        """
        # Get all documents from the collection
        results = self.collection.get()

        # Convert to LangChain Document format
        documents = []
        if results["documents"]:
            for i in range(len(results["documents"])):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                documents.append(
                    Document(
                        page_content=results["documents"][i],
                        metadata=metadata,
                    )
                )

        return documents

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Document]:
        """Search for similar documents and return as LangChain Documents.

        Args:
        ----
            query: Search query text
            k: Number of results to return
            where: Optional filter conditions on metadata
            where_document: Optional filter conditions on document content

        Returns:
        -------
            List of LangChain Document objects

        """
        # Use the existing search method to get results
        results = self.search(
            query=query,
            n_results=k,
            where=where,
            where_document=where_document,
        )

        # Convert to LangChain Document format
        documents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result["document"],
                    metadata=result["metadata"] or {},
                )
            )

        return documents

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> "ChromaRetriever":
        """Create a retriever from this vector store.

        Args:
        ----
            search_kwargs: Optional search parameters

        Returns:
        -------
            A retriever object

        """
        return ChromaRetriever(vectorstore=self, search_kwargs=search_kwargs or {})


class ChromaRetriever:
    """Retriever for ChromaVectorStore."""

    def __init__(
        self,
        vectorstore: ChromaVectorStore,
        search_kwargs: Dict[str, Any],
    ):
        """Initialize the retriever.

        Args:
        ----
            vectorstore: The vector store to retrieve from
            search_kwargs: Search parameters

        """
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query.

        Args:
        ----
            query: The query to search for

        Returns:
        -------
            List of relevant documents

        """
        k = self.search_kwargs.get("k", 5)
        where = self.search_kwargs.get("where")
        where_document = self.search_kwargs.get("where_document")

        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            where=where,
            where_document=where_document,
        )
