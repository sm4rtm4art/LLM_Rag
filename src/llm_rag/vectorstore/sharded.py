"""
Sharded vector store implementation for improved scalability.

This module provides a sharded vector store implementation that distributes documents
across multiple ChromaVectorStore instances (shards) to improve vertical scalability.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union

from llm_rag.vectorstore.chroma import ChromaVectorStore
from llm_rag.vectorstore.base import VectorStore

# Set up a basic logger
logger = logging.getLogger(__name__)


class ShardedChromaVectorStore(VectorStore):
    """
    A sharded vector store that distributes documents across multiple
    ChromaVectorStore instances (shards) to improve vertical scalability.
    
    Each shard is limited to a maximum number of documents. When a shard's
    capacity is reached, a new shard is created automatically.

    Attributes:
        shard_capacity (int): Maximum number of documents per shard.
        base_persist_directory (str): Base directory for storing shard data.
        max_workers (int): Number of threads used for concurrent search operations.
        chroma_kwargs (dict): Additional keyword arguments for initializing ChromaVectorStore.
        shards (List[ChromaVectorStore]): List of ChromaVectorStore shard instances.
    """

    def __init__(
        self,
        shard_capacity: int = 10000,
        base_persist_directory: str = "sharded_chroma_db",
        max_workers: int = 4,
        **chroma_kwargs: Any
    ) -> None:
        """
        Initializes the ShardedChromaVectorStore.

        Args:
            shard_capacity: Maximum number of documents per shard.
            base_persist_directory: Base directory where all shard data will be stored.
            max_workers: Maximum number of worker threads for concurrent search.
            chroma_kwargs: Additional parameters for initializing ChromaVectorStore.
        """
        self.shard_capacity = shard_capacity
        self.base_persist_directory = base_persist_directory
        self.chroma_kwargs = chroma_kwargs
        self.max_workers = max_workers
        self.shards: List[ChromaVectorStore] = []
        self._create_new_shard()

    def _create_new_shard(self) -> None:
        """
        Creates a new shard (a ChromaVectorStore instance) and appends it to the shards list.
        """
        shard_index = len(self.shards)
        shard_dir = os.path.join(self.base_persist_directory, f"shard_{shard_index}")
        os.makedirs(shard_dir, exist_ok=True)
        new_shard = ChromaVectorStore(persist_directory=shard_dir, **self.chroma_kwargs)
        self.shards.append(new_shard)
        logger.info(f"Created new shard at {shard_dir}")

    def add_documents(
        self, 
        documents: List[Union[str, Dict[str, Any]]], 
        metadatas: Optional[List[Dict[str, Any]]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Adds documents to the current shard. If the addition would exceed the current shard's capacity,
        a new shard is created first.

        Args:
            documents: A list of document contents or dictionaries with content and metadata.
            metadatas: A list of metadata dictionaries corresponding to each document.
            ids: Optional list of document identifiers.
        """
        # Process documents if they are in dict format with content/metadata
        processed_docs = []
        processed_metadatas = [] if metadatas is None else metadatas
        
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and "content" in doc:
                # Extract content and metadata from document dict
                processed_docs.append(doc["content"])
                if "metadata" in doc and processed_metadatas is not None:
                    if i < len(processed_metadatas):
                        # Update existing metadata
                        processed_metadatas[i].update(doc["metadata"])
                    else:
                        # Append new metadata
                        processed_metadatas.append(doc["metadata"])
            else:
                # Document is a string
                processed_docs.append(doc)
        
        # If metadatas is None but we added some from documents, use our processed list
        if metadatas is None and processed_metadatas:
            metadatas = processed_metadatas
        
        # Get the current shard (the last one in the list)
        current_shard = self.shards[-1]
        processed_docs = documents
        
        # Get the current count of documents in the shard
        current_count = (
            current_shard.get_collection_size() 
            if hasattr(current_shard, "get_collection_size") 
            else 0
        )
        
        # Check if we need a new shard
        if current_count + len(processed_docs) > self.shard_capacity:
            self._create_new_shard()
            current_shard = self.shards[-1]
        
        # Add documents to the current shard
        current_shard.add_documents(processed_docs, metadatas, ids)
        
        shard_count = (
            current_shard.get_collection_size() 
            if hasattr(current_shard, "get_collection_size") 
            else 0
        )
        
        logger.info(
            f"Added {len(processed_docs)} documents to shard {len(self.shards) - 1}. "
            f"Total in shard: {shard_count}"
        )

    def _search_shard(
        self,
        shard: ChromaVectorStore,
        query: str,
        n_results: int,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Executes a search on a single shard.

        Args:
            shard: The shard instance to search.
            query: The search query string.
            n_results: Maximum number of results to retrieve from the shard.
            kwargs: Additional search parameters.

        Returns:
            A list of search result dictionaries from the shard.
        """
        return shard.search(query, n_results, **kwargs)

    def search(
        self, 
        query: str, 
        n_results: int = 10, 
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Searches across all shards concurrently and aggregates the results.

        Args:
            query: The search query string.
            n_results: Total number of top results to return.
            kwargs: Additional search parameters.

        Returns:
            A list of the top search result dictionaries aggregated from all shards.
        """
        all_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_shard = {
                executor.submit(
                    self._search_shard, shard, query, n_results, **kwargs
                ): shard
                for shard in self.shards
            }
            for future in as_completed(future_to_shard):
                shard = future_to_shard[future]
                try:
                    results = future.result()
                    logger.info(f"Shard search returned {len(results)} results")
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error searching shard: {e}")
        # Sort results if they have a 'score' key.
        if all_results and "score" in all_results[0]:
            all_results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Total aggregated results: {len(all_results)}")
        return all_results[:n_results]

    def count(self) -> int:
        """
        Computes the total number of documents stored across all shards.

        Returns:
            Total document count as an integer.
        """
        total = sum(
            shard.get_collection_size() 
            for shard in self.shards 
            if hasattr(shard, "get_collection_size")
        )
        logger.info(f"Total document count across all shards: {total}")
        return total
    
    def as_retriever(self, **kwargs: Any) -> Any:
        """
        Creates a retriever from this vector store.

        Args:
            kwargs: Additional parameters for initializing the retriever.

        Returns:
            A retriever instance.
        """
        # Import within the method to avoid circular imports
        from llm_rag.vectorstore.chroma import ChromaRetriever as Retriever
        
        return Retriever(vectorstore=self, **kwargs)

    def delete_collection(self) -> None:
        """
        Delete all collections across all shards.
        """
        for shard in self.shards:
            shard.delete_collection()
        self.shards = []
        logger.info("Deleted all shard collections")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Any]:
        """Search for similar documents and return as LangChain Documents.

        Args:
            query: Search query text
            k: Number of results to return
            where: Optional filter conditions on metadata
            where_document: Optional filter conditions on document content

        Returns:
            List of LangChain Document objects
        """
        from langchain_core.documents import Document
        
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