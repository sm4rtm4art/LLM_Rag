# Sharded Vector Store

The `ShardedChromaVectorStore` provides a horizontally scalable vector database solution built on top of ChromaDB. It automatically distributes documents across multiple ChromaDB instances (shards) to improve performance when working with large document collections.

## Key Features

- **Automatic Sharding**: Distributes documents across multiple ChromaDB shards
- **Capacity Management**: Automatically creates new shards when capacity limits are reached
- **Concurrent Search**: Searches across all shards in parallel for improved performance
- **Seamless API**: Compatible with the standard VectorStore interface

## When to Use

Consider using the ShardedChromaVectorStore when:

- Your document collection contains 100,000+ documents
- Individual ChromaDB instances are becoming slow or running out of memory
- You need to scale horizontally for better performance
- You want to manage very large datasets without compromising search speed

## Usage Examples

### Basic Initialization

```python
from llm_rag.vectorstore import ShardedChromaVectorStore
from llm_rag.models.embeddings import SentenceTransformerEmbeddings

# Initialize the embedding function
embeddings = SentenceTransformerEmbeddings()

# Create a sharded vector store with custom settings
vector_store = ShardedChromaVectorStore(
    shard_capacity=50000,  # Maximum documents per shard
    base_persist_directory="data/sharded_db",  # Base directory for all shards
    max_workers=4,  # Number of threads for concurrent search
    embedding_function=embeddings  # Pass to each ChromaVectorStore shard
)
```

### Adding Documents

```python
# Add a batch of documents
documents = [
    "Document content 1",
    "Document content 2",
    # ... more documents
]

metadatas = [
    {"source": "file1.txt", "category": "technical"},
    {"source": "file2.txt", "category": "financial"},
    # ... metadata for each document
]

# The documents will be automatically distributed across shards
vector_store.add_documents(documents, metadatas)

# You can also add documents with embedded metadata
documents_with_metadata = [
    {"content": "Document content 3", "metadata": {"source": "file3.txt"}},
    {"content": "Document content 4", "metadata": {"source": "file4.txt"}},
    # ... more documents
]

vector_store.add_documents(documents_with_metadata)
```

### Searching

```python
# Search across all shards concurrently
results = vector_store.search(
    query="What is machine learning?",
    n_results=5
)

# Each result includes content, metadata, and relevance score
for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Source: {result['metadata']['source']}")
    print("---")
```

### Using as a Retriever

```python
# Create a retriever from the sharded vector store
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# Use the retriever in your RAG pipeline
documents = retriever.get_relevant_documents("What is machine learning?")
```

### Managing the Vector Store

```python
# Get the total document count across all shards
total_docs = vector_store.count()
print(f"Total documents: {total_docs}")

# Delete the entire collection (all shards)
vector_store.delete_collection()
```

## Performance Considerations

- **Shard Capacity**: Choose a shard capacity that balances performance and management complexity. The default of 10,000 documents per shard is a good starting point, but you may need to adjust based on document size and available RAM.

- **Workers**: Set `max_workers` to match your available CPU cores for optimal search performance.

- **Memory Usage**: Each shard runs a separate ChromaDB instance, which requires memory. Monitor system resources when scaling to many shards.

- **Persistence**: Each shard has its own persistence directory, making it easy to back up or migrate individual shards.

## Internal Architecture

The `ShardedChromaVectorStore` manages a list of `ChromaVectorStore` instances internally. When you add documents:

1. It checks if adding to the current shard would exceed capacity
2. If needed, it creates a new shard automatically
3. Documents are added to the appropriate shard

During search operations:

1. The query is sent to all shards concurrently using a thread pool
2. Results from all shards are aggregated
3. Results are sorted by relevance score
4. The top `n_results` are returned

This architecture allows for efficient scaling with large document collections while maintaining high search performance.
