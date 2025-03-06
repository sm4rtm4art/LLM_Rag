# Anti-Hallucination Features in the RAG System

This document describes the anti-hallucination features implemented in the RAG system to reduce the likelihood of generating incorrect or fabricated information.

## Overview

Hallucinations in RAG systems occur when the language model generates information that is not supported by the retrieved documents. This can happen for several reasons:

1. The retrieval system fails to find relevant documents
2. The language model ignores the retrieved context
3. The language model "fills in the gaps" with plausible but incorrect information

Our anti-hallucination system addresses these issues through multiple complementary approaches.

## Implemented Features

### 1. Enhanced Prompt Engineering

We've updated the prompt template to explicitly instruct the model to:

- Only use information from the provided context
- Acknowledge when it doesn't have enough information
- Never make up facts or hallucinate information

The new prompt template includes clear instructions and guardrails for the model.

### 2. Improved Document Retrieval and Formatting

- Increased the default number of retrieved documents from 3 to 5
- Enhanced document formatting with clear separation between documents
- Added source attribution to help trace information back to its origin
- Improved metadata display for better context understanding

### 3. Confidence Scoring

The system now calculates a confidence score for each query based on:

- Number of documents retrieved
- Presence of query terms in the retrieved documents
- Document similarity scores

When confidence is low, a warning is added to the context to encourage the model to be more cautious in its response.

### 4. Entity Verification

After generating a response, the system:

1. Extracts key entities from both the response and the context
2. Identifies entities in the response that are not present in the context
3. Calculates an entity coverage ratio
4. Adds a warning to the response if too many entities are not grounded in the context

### 5. Post-Processing

The system post-processes responses to:

- Add warnings about potential hallucinations
- Highlight specific terms that may be hallucinated
- Suggest verification from other sources when confidence is low

## Usage

### Basic Usage

The anti-hallucination features are enabled by default in the RAG pipeline. No additional configuration is needed.

```python
from src.llm_rag.models.factory import ModelBackend, ModelFactory
from src.llm_rag.rag.pipeline import RAGPipeline
from src.llm_rag.vectorstore.chroma import ChromaVectorStore

# Create the LLM
factory = ModelFactory()
llm = factory.create_model(
    model_path_or_name="microsoft/phi-2",
    backend=ModelBackend.HUGGINGFACE,
)

# Load the vector store
vector_store = ChromaVectorStore(
    collection_name="documents",
    persist_directory="chroma_db",
)

# Create the RAG pipeline
rag_pipeline = RAGPipeline(
    vectorstore=vector_store,
    llm=llm,
    top_k=5,  # Number of documents to retrieve
)

# Run a query
result = rag_pipeline.query("What is the purpose of DIN_SPEC_31009?")

# The response will include warnings if hallucinations are detected
response = result["response"]
confidence = result["confidence"]
```

### Testing Anti-Hallucination Features

You can use the `test_anti_hallucination.py` script to test the anti-hallucination features:

```bash
# Run a single query
python scripts/test_anti_hallucination.py --query "What is the purpose of DIN_SPEC_31009?"

# Run a set of test queries
python scripts/test_anti_hallucination.py --test_queries
```

## Future Improvements

Planned improvements to the anti-hallucination system include:

1. **Semantic Similarity**: Using embeddings to measure semantic similarity between response and context
2. **Fact Checking**: Implementing a dedicated fact-checking module
3. **Adaptive Retrieval**: Dynamically adjusting the number of documents retrieved based on query complexity
4. **Multi-hop Reasoning**: Breaking complex queries into sub-queries for better context retrieval
5. **User Feedback Loop**: Incorporating user feedback to improve the system over time

## References

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Hallucination in Large Language Models](https://arxiv.org/abs/2309.01219)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
