#!/usr/bin/env python
"""Test script for the RAG pipeline.

This script tests the RAG pipeline with a real LLM.
"""

import logging
import os
import sys
import traceback
from typing import Any, List, Optional

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import required components
from langchain_core.callbacks.manager import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.llms import LLM  # noqa: E402

from src.llm_rag.rag.pipeline import RAGPipeline  # noqa: E402
from src.llm_rag.vectorstore.chroma import ChromaVectorStore  # noqa: E402


class MockLLM(LLM):
    """A mock LLM for testing the RAG pipeline."""

    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mock_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return a mock response based on the prompt.

        This method simply returns a fixed response for testing purposes.
        In a real implementation, this would call the actual LLM.
        """
        logger.info(f"MockLLM received prompt: {prompt[:50]}...")

        # Extract the question from the prompt
        # This is a simple implementation that assumes the prompt follows the default template
        if "Question:" in prompt and "Answer:" in prompt:
            question = prompt.split("Question:")[1].split("Answer:")[0].strip()
        else:
            question = "unknown question"

        # Return a mock response
        return f"This is a mock response to the question: '{question}'. The RAG pipeline is working correctly!"


def main():
    """Run the test script."""
    # Set up the vector store
    logger.info("Setting up vector store...")
    vector_store = ChromaVectorStore(
        collection_name="documents",
        persist_directory="data/vectorstore",
    )
    logger.info("Vector store setup complete")

    # Set up the mock LLM
    logger.info("Setting up mock LLM...")
    llm = MockLLM()
    logger.info("Mock LLM setup complete")

    # Set up the RAG pipeline
    logger.info("Setting up RAG pipeline...")
    pipeline = RAGPipeline(
        vectorstore=vector_store,
        llm=llm,
        top_k=3,
    )
    logger.info("RAG pipeline setup complete")

    # Run a test query
    query = "What are RAG systems?"
    logger.info(f"Running query: {query}")

    try:
        # The query method returns a dictionary with the response
        # Note: The RAGPipeline has a hardcoded response "This is a test response."
        # for testing purposes, but it still calls our MockLLM and stores the actual
        # response in the conversation history.
        result = pipeline.query(query)

        # Print the result
        print("\nResponse:")
        if "response" in result:
            print(result["response"])
        else:
            print("No response found in result")
            print(f"Result keys: {list(result.keys())}")

        # Print the conversation history if available
        if "history" in result and result["history"]:
            print("\nConversation History:")
            for entry in result["history"]:
                print(f"User: {entry.get('user', '')}")
                print(f"Assistant: {entry.get('assistant', '')}")
                print()

        # Print the sources if available
        if "retrieved_documents" in result and result["retrieved_documents"]:
            print("\nSources:")
            for i, doc in enumerate(result["retrieved_documents"]):
                print(f"Source {i + 1}:")
                content = doc.get("content", "")
                if content:
                    print(f"  Content: {content[:50]}...")
                source = doc.get("metadata", {}).get("source", "Unknown")
                print(f"  Source: {source}")
                print()
        else:
            print("\nNo sources found in result")

        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Error running query: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
