#!/usr/bin/env python
"""Test RAG with specific document IDs."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.vectorstores import Chroma

from src.llm_rag.models.factory import ModelBackend, ModelFactory
from src.llm_rag.rag.pipeline import RAGPipeline


def run_rag_with_doc_id(
    query: str,
    doc_id: str,
    model_name: str,
    db_path: str,
    collection_name: str,
) -> Dict[str, Any]:
    """Run RAG with a specific document ID."""
    # Load the LLM model
    logger.info(f"Loading model: {model_name}")
    llm = ModelFactory.create_model(
        model_path_or_name=model_name,
        backend=ModelBackend.HUGGINGFACE,
        device="cpu",
        max_tokens=512,
        temperature=0.1,  # Lower temperature for more deterministic output
    )

    # Load the document directly from ChromaDB
    logger.info(f"Loading document with ID: {doc_id} from {db_path}")
    db = Chroma(collection_name=collection_name, persist_directory=db_path)
    results = db.get(ids=[doc_id])

    if not results["documents"]:
        logger.error(f"Document with ID {doc_id} not found")
        return {
            "response": f"Document with ID {doc_id} not found",
            "documents": [],
            "confidence": 0.0,
        }

    # Extract document content and metadata
    document_content = results["documents"][0]
    document_metadata = results["metadatas"][0]

    logger.info(f"Found document: {document_metadata}")
    logger.info(f"Document content preview: {document_content[:200]}...")

    # Create a document in the format expected by the RAG pipeline
    document = {
        "content": document_content,
        "metadata": document_metadata,
    }

    # Create the RAG pipeline with a custom prompt template
    logger.info("Creating RAG pipeline")
    custom_prompt = """
You are a precise assistant that answers questions based ONLY on the provided context.

Context:
{context}

Question: {query}

IMPORTANT INSTRUCTIONS:
1. ONLY use information explicitly stated in the context above.
2. If the context doesn't contain the answer, say "Based on the provided context, I cannot answer this question."
3. DO NOT make up or infer information not directly stated in the context.
4. DO NOT use any prior knowledge.
5. DIRECTLY QUOTE the relevant parts of the context in your answer.
6. Be concise and to the point.
7. If the question is in German, answer in German using the exact German text from the context.

Answer:
"""

    pipeline = RAGPipeline(
        vectorstore=None,  # We're not using the vectorstore for retrieval
        llm=llm,
        top_k=1,  # We're only using one document
        prompt_template=custom_prompt,
    )

    # Format the context manually
    context = pipeline.format_context([document])

    # Generate response
    logger.info(f"Generating response for query: {query}")
    response = pipeline.generate(query=query, context=context)

    return {
        "response": response,
        "documents": [document],
        "confidence": 1.0,  # We're manually providing the document, so confidence is high
    }


def main():
    """Execute the RAG test with specific document IDs."""
    parser = argparse.ArgumentParser(description="Test RAG with specific document ID")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to ask the RAG system",
    )
    parser.add_argument(
        "--doc_id",
        type=str,
        required=True,
        help="Document ID to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-2",
        help="Model name to use",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="test_subset",
        help="Collection name",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="chroma_db",
        help="Path to ChromaDB database",
    )

    args = parser.parse_args()

    # Run RAG with the specified document ID
    result = run_rag_with_doc_id(
        query=args.query,
        doc_id=args.doc_id,
        model_name=args.model_name,
        db_path=args.db_path,
        collection_name=args.collection,
    )

    # Print the response
    print("\nRAG System Response:")
    print("-------------------")
    print(result["response"])
    print("-------------------")

    # Print document information
    if result["documents"]:
        print("\nDocument Used:")
        print(f"ID: {args.doc_id}")
        metadata = result["documents"][0]["metadata"]
        print(f"Filename: {metadata.get('filename', 'Unknown')}")
        print(f"Source: {metadata.get('source', 'Unknown')}")

        # Print content preview
        content = result["documents"][0]["content"]
        print("\nContent Preview:")
        print(content[:500] + "..." if len(content) > 500 else content)


if __name__ == "__main__":
    main()
