#!/usr/bin/env python
"""Interactive CLI for testing RAG with anti-hallucination features.

This script allows you to test the RAG system with various configurations
and see how the anti-hallucination module affects the responses.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add the src directory to the Python path if necessary
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from llm_rag.rag.anti_hallucination import HallucinationConfig, extract_key_entities, post_process_response
    from llm_rag.rag.pipeline import RAGPipeline
except ImportError:
    print("Error: Could not import from llm_rag. Make sure the package is installed.")
    sys.exit(1)

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI
except ImportError:
    print(
        "Error: Required dependencies not found. Try installing langchain, langchain-openai, and langchain-community."
    )
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_rag(
    vector_db_path: str, openai_api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"
) -> RAGPipeline:
    """Set up the RAG pipeline.

    Args:
        vector_db_path: Path to the vector database
        openai_api_key: OpenAI API key (uses env var if not provided)
        model_name: LLM model name to use

    Returns:
        Initialized RAG pipeline

    """
    # Set API key from environment if not provided
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load vector store
    vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)

    # Initialize LLM
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # Create a custom RAG pipeline class to override the search method
    class CustomRAGPipeline(RAGPipeline):
        def _fetch_documents_from_vectorstore(self, query: str) -> Optional[List[Any]]:
            """Override to use similarity search instead of hybrid search."""
            try:
                logger.info(f"Retrieving documents for query: {query}")
                # Use similarity search instead of hybrid search
                raw_docs = self.vectorstore.similarity_search(query, k=self.top_k)
                logger.info(f"Retrieved {len(raw_docs)} documents using similarity search")
                return raw_docs
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                return []

    # Initialize custom RAG pipeline
    rag = CustomRAGPipeline(vectorstore=vectorstore, llm=llm, top_k=5)

    return rag


def analyze_response(
    response: str, context: str, use_anti_hallucination: bool = True, config: Optional[HallucinationConfig] = None
) -> Tuple[str, Dict[str, Any]]:
    """Analyze a response using the anti-hallucination module.

    Args:
        response: The generated response
        context: The context used to generate the response
        use_anti_hallucination: Whether to use anti-hallucination features
        config: HallucinationConfig object (uses default if not provided)

    Returns:
        Tuple of (processed_response, metadata)

    """
    if not use_anti_hallucination:
        return response, {}

    if config is None:
        config = HallucinationConfig(
            flag_for_human_review=True, use_embeddings=True, entity_threshold=0.7, embedding_threshold=0.6
        )

    processed_response, metadata = post_process_response(
        response=response, context=context, config=config, return_metadata=True
    )

    return processed_response, metadata


def print_analysis(response: str, context: str, metadata: Dict[str, Any]) -> None:
    """Print a detailed analysis of the response.

    Args:
        response: The generated response
        context: The context used to generate the response
        metadata: Metadata from anti-hallucination processing

    """
    print("\n" + "=" * 80)
    print("RESPONSE ANALYSIS")
    print("=" * 80)

    # Print context and entities
    context_entities = extract_key_entities(context)
    response_entities = extract_key_entities(response)

    print(f"\nContext entities ({len(context_entities)}):")
    print(", ".join(sorted(list(context_entities))))

    print(f"\nResponse entities ({len(response_entities)}):")
    print(", ".join(sorted(list(response_entities))))

    # Print unique entities (potentially hallucinated)
    unique_entities = [e for e in response_entities if e not in context_entities]
    print(f"\nPotentially hallucinated entities ({len(unique_entities)}):")
    print(", ".join(sorted(unique_entities)) if unique_entities else "None")

    # Print verification results
    if metadata:
        print("\nVerification results:")
        print(f"  Verified: {metadata['verified']}")
        print(f"  Entity coverage: {metadata['entity_coverage']:.2f}")
        print(f"  Embedding similarity: {metadata['embedding_similarity']:.2f}")
        print(f"  Hallucination score: {metadata['hallucination_score']:.2f}")
        print(f"  Human review recommended: {metadata['human_review_recommended']}")

    print("\n" + "=" * 80)


def interactive_rag_session(rag: RAGPipeline, use_anti_hallucination: bool = True) -> None:
    """Run an interactive RAG session.

    Args:
        rag: Initialized RAG pipeline
        use_anti_hallucination: Whether to use anti-hallucination features

    """
    print("\nWelcome to the RAG interactive test session!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'config' to modify anti-hallucination settings.")
    print("Type 'analyze' to toggle response analysis.")

    show_analysis = True
    config = HallucinationConfig(
        flag_for_human_review=True, use_embeddings=True, entity_threshold=0.7, embedding_threshold=0.6
    )

    while True:
        query = input("\nEnter your query: ")

        if query.lower() in ("exit", "quit"):
            break

        if query.lower() == "config":
            print("\nCurrent configuration:")
            print(f"  Entity threshold: {config.entity_threshold}")
            print(f"  Embedding threshold: {config.embedding_threshold}")
            print(f"  Flag for human review: {config.flag_for_human_review}")
            print(f"  Use embeddings: {config.use_embeddings}")

            print("\nEnter new values (leave blank to keep current value):")

            val = input("  Entity threshold [0-1]: ")
            if val.strip():
                config.entity_threshold = float(val)

            val = input("  Embedding threshold [0-1]: ")
            if val.strip():
                config.embedding_threshold = float(val)

            val = input("  Flag for human review [y/n]: ")
            if val.strip():
                config.flag_for_human_review = val.lower() in ("y", "yes", "true")

            val = input("  Use embeddings [y/n]: ")
            if val.strip():
                config.use_embeddings = val.lower() in ("y", "yes", "true")

            print("\nConfiguration updated!")
            continue

        if query.lower() == "analyze":
            show_analysis = not show_analysis
            print(f"\nResponse analysis: {'ON' if show_analysis else 'OFF'}")
            continue

        try:
            print("\nRetrieving information...")
            result = rag.query(query)

            response = result["response"]
            context = result.get("context", "")

            if use_anti_hallucination:
                processed_response, metadata = analyze_response(response, context, True, config)

                print("\nResponse (processed with anti-hallucination):")
                print(processed_response)

                if show_analysis:
                    print_analysis(response, context, metadata)
            else:
                print("\nResponse (without anti-hallucination):")
                print(response)

        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test the RAG system with anti-hallucination features")
    parser.add_argument("--vector-db", type=str, required=True, help="Path to the vector database")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (uses env var if not provided)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model name")
    parser.add_argument("--no-anti-hallucination", action="store_true", help="Disable anti-hallucination")

    args = parser.parse_args()

    try:
        # Set up RAG pipeline
        rag = setup_rag(vector_db_path=args.vector_db, openai_api_key=args.api_key, model_name=args.model)

        # Run interactive session
        interactive_rag_session(rag, not args.no_anti_hallucination)

    except Exception as e:
        logger.error(f"Error during RAG session: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
