"""Test the RAG system with anti-hallucination features.

This script builds a test database from existing documents and then
tests the RAG system with anti-hallucination features using Ollama for local LLM.
"""

import argparse
import logging
import os
import re
import shutil
import sys
from typing import Any, Dict, List

# Add the src directory to the path so we can import the llm_rag module
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    # Updated import for Ollama
    from langchain_community.llms import Ollama

    # Import OllamaLLM if it exists in langchain_community
    try:
        from langchain_community.llms.ollama import OllamaLLM
    except ImportError:
        # If OllamaLLM doesn't exist, we'll use Ollama only
        OllamaLLM = None

    # Add deprecation warning for old imports
    import warnings

    from llm_rag.document_processing.loaders import DirectoryLoader
    from llm_rag.rag.anti_hallucination import HallucinationConfig, post_process_response

    # Import the new pipeline modules
    from llm_rag.rag.pipeline.base import RAGPipeline
    from llm_rag.vectorstore.chroma import ChromaVectorStore, EmbeddingFunctionWrapper

    warnings.warn(
        "Importing from 'llm_rag.rag.pipeline' is deprecated and will be removed in a future version. "
        "Please update your imports to use 'llm_rag.rag.pipeline.base' and related modules instead.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError:
    print('Error: Required modules not found. Please install the llm_rag package.')
    sys.exit(1)

try:
    from langchain_ollama import OllamaLLM

    use_new_ollama = True
    print('Using new langchain_ollama implementation.')
except ImportError:
    from langchain.llms import Ollama as OllamaLLM

    use_new_ollama = False
    print('Using legacy langchain.llms.Ollama implementation.')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_documents(docs_dir: str) -> List[Dict[str, Any]]:
    """Load documents from a directory.

    Args:
        docs_dir: Path to the directory containing documents

    Returns:
        List of documents with content and metadata

    """
    logger.info(f'Loading documents from {docs_dir}')

    # Use DirectoryLoader to load all documents in the directory
    loader = DirectoryLoader(
        directory_path=docs_dir,
        recursive=False,
    )

    documents = loader.load()
    logger.info(f'Loaded {len(documents)} documents')

    return documents


def build_vectorstore(
    documents: List[Dict[str, Any]], output_dir: str, collection_name: str = 'anti_hallucination_test'
) -> ChromaVectorStore:
    """Build a vector store from documents.

    Args:
        documents: List of documents with content and metadata
        output_dir: Directory to store the vector store
        collection_name: Name of the collection

    Returns:
        ChromaVectorStore instance

    """
    logger.info(f'Building vector store in {output_dir}')

    # Create a fresh directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize embedding function
    embedding_function = EmbeddingFunctionWrapper(model_name='all-MiniLM-L6-v2')

    # Create vector store
    vectorstore = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=output_dir,
        embedding_function=embedding_function,
        overwrite=True,
    )

    # Add documents to vector store
    texts = []
    metadatas = []

    for doc in documents:
        texts.append(doc['content'])
        # Ensure all metadata values are valid types
        metadata = {}
        for key, value in doc.get('metadata', {}).items():
            if value is not None and isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                # Convert None or invalid types to string
                metadata[key] = str(value) if value is not None else ''
        metadatas.append(metadata)  # Append the processed metadata to the list

    vectorstore.add_documents(texts, metadatas=metadatas)
    vectorstore.persist()

    logger.info(f'Added {len(texts)} documents to vector store')
    return vectorstore


def setup_rag_pipeline(vectorstore: ChromaVectorStore, model_name: str = 'llama3') -> RAGPipeline:
    """Set up a RAG pipeline with anti-hallucination features.

    Args:
        vectorstore: Vector store for document retrieval
        model_name: Name of the Ollama model to use

    Returns:
        Configured RAG pipeline

    """
    # Initialize LLM with proper Ollama implementation
    try:
        if use_new_ollama and OllamaLLM is not None:
            # Use the new recommended implementation
            llm = OllamaLLM(model=model_name)
            logger.info(f'Successfully initialized OllamaLLM with model: {model_name}')
        else:
            # Use the deprecated implementation
            llm = Ollama(model=model_name)
            logger.info(f'Successfully initialized Ollama with model: {model_name}')
    except Exception as e:
        logger.error(f'Error initializing Ollama: {e}')
        raise RuntimeError(f'Failed to initialize Ollama: {e}') from e

    # Define a wrapper for post_process_response to ensure it returns the expected format
    def post_processor_wrapper(response, context):
        try:
            # Process the response with anti-hallucination checks
            processed_response, metadata = post_process_response(
                response=response,
                context=context,
                return_metadata=True,
            )

            # Print analysis if available
            if metadata:
                print_analysis(response, context, metadata)

            return processed_response
        except Exception as e:
            logger.error(f'Error in post-processing: {e}')
            return response

    # Import the necessary modules
    from llm_rag.rag.pipeline.generation import DEFAULT_PROMPT_TEMPLATE

    # Create the pipeline with the vectorstore and LLM
    pipeline = RAGPipeline(
        vectorstore=vectorstore,
        llm=llm,
    )

    # Create a patched generator.generate method that can handle Ollama's string output
    def patched_generate(query, context, history='', **kwargs):
        """Patched generate method that handles Ollama's string output format."""
        logger.debug(f'Using patched generate method for query: {query}')

        # Format prompt directly to avoid the content attribute error
        prompt = DEFAULT_PROMPT_TEMPLATE.format(
            query=query,
            context=context,
            history=history,
        )

        try:
            # Generate response directly from the LLM
            response_raw = llm.invoke(prompt)

            # Handle the response format - could be string or an object with content attribute
            if hasattr(response_raw, 'content'):
                response = response_raw.content
            else:
                response = str(response_raw)

            # Apply anti-hallucination processing
            processed_response = post_processor_wrapper(response, context)
            return processed_response

        except Exception as e:
            logger.error(f'Error in patched_generate: {e}')
            return f"I apologize, but I couldn't generate a proper response due to a technical issue: {str(e)}"

    # Replace the generator's generate method with our patched version
    pipeline._generator.generate = patched_generate

    return pipeline


def print_analysis(response: str, context: str, metadata: Dict[str, Any]) -> None:
    """Print analysis of the response and hallucination detection.

    Args:
        response: The generated response
        context: The context used to generate the response
        metadata: Metadata from hallucination detection

    """
    print('\n' + '=' * 80)
    print('RESPONSE ANALYSIS')
    print('=' * 80)

    # Print the original response
    print('\nOriginal Response:')
    print('-' * 40)
    print(response)
    print('-' * 40)

    # Print hallucination detection results if available
    if 'entity_verification' in metadata:
        print('\nEntity Verification:')
        print(f'Score: {metadata["entity_verification"]["score"]:.2f}')

        if 'entities' in metadata['entity_verification']:
            print('\nEntities Found:')
            for entity, verified in metadata['entity_verification']['entities'].items():
                status = '✓' if verified else '✗'
                print(f'  {status} {entity}')

    if 'embedding_verification' in metadata:
        print('\nEmbedding Verification:')
        print(f'Score: {metadata["embedding_verification"]["score"]:.2f}')

        if 'similarity' in metadata['embedding_verification']:
            print(f'Similarity: {metadata["embedding_verification"]["similarity"]:.2f}')

    if 'combined_score' in metadata:
        print(f'\nCombined Score: {metadata["combined_score"]:.2f}')

    if 'needs_human_review' in metadata and metadata['needs_human_review']:
        print('\n⚠️ This response may need human review!')

    print('=' * 80 + '\n')


def interactive_rag_session(rag: RAGPipeline) -> None:
    """Run an interactive RAG session with anti-hallucination features.

    The user can ask questions and get responses from the RAG system.

    Args:
        rag: The RAG pipeline to use.

    """
    print('\nWelcome to the RAG interactive test session!')
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'config' to modify anti-hallucination settings.")
    print("Type 'toggle' to toggle anti-hallucination on/off.")
    print()

    use_anti_hallucination = True
    config = HallucinationConfig(
        flag_for_human_review=True, use_embeddings=True, entity_threshold=0.7, embedding_threshold=0.5
    )

    while True:
        query = input('\nEnter your query: ')

        if query.lower() in ('exit', 'quit'):
            break

        if query.lower() == 'toggle':
            use_anti_hallucination = not use_anti_hallucination
            print(f'\nAnti-hallucination features: {"ON" if use_anti_hallucination else "OFF"}')
            continue

        if query.lower() == 'config':
            print('\nCurrent configuration:')
            print(f'  Entity threshold: {config.entity_threshold}')
            print(f'  Embedding threshold: {config.embedding_threshold}')
            print(f'  Flag for human review: {config.flag_for_human_review}')
            print(f'  Use embeddings: {config.use_embeddings}')

            print('\nEnter new values (leave blank to keep current):')

            val = input('  Entity threshold [0-1]: ')
            if val.strip():
                config.entity_threshold = float(val)

            val = input('  Embedding threshold [0-1]: ')
            if val.strip():
                config.embedding_threshold = float(val)

            val = input('  Flag for human review [y/n]: ')
            if val.strip():
                config.flag_for_human_review = val.lower() in ('y', 'yes', 'true')

            val = input('  Use embeddings [y/n]: ')
            if val.strip():
                config.use_embeddings = val.lower() in ('y', 'yes', 'true')

            continue

        print('\nRetrieving information...')
        try:
            # Get response from RAG system
            result = rag.query(query)

            # Check if result is a dictionary or a string
            if isinstance(result, dict):
                response = result.get('response', 'No response generated')
            else:
                # If it's just a string, use it directly
                response = result

            # Print the response
            print(f'\nResponse {"(with anti-hallucination)" if use_anti_hallucination else ""}:')
            print(response)

        except Exception as e:
            print(f'\nError: {str(e)}')
            logger.error(f'Error during RAG query: {e}', exc_info=True)


def main():
    """Run RAG test with anti-hallucination."""
    parser = argparse.ArgumentParser(description='Test RAG with anti-hallucination using Ollama')
    parser.add_argument(
        '--docs-dir', type=str, default='data/documents/test_subset', help='Directory containing documents'
    )
    parser.add_argument(
        '--output-dir', type=str, default='test_anti_hallucination_db', help='Directory to store the vector database'
    )
    parser.add_argument('--model', type=str, default='llama3', help='Ollama model name')
    parser.add_argument('--skip-build', action='store_true', help='Skip building the vector store if it exists')
    parser.add_argument('--api-key', type=str, help='OpenAI API key')

    args = parser.parse_args()

    try:
        # Build vector store if needed
        if not args.skip_build or not os.path.exists(args.output_dir):
            # Load documents
            documents = load_documents(args.docs_dir)

            # Build vector store
            vectorstore = build_vectorstore(documents=documents, output_dir=args.output_dir)
        else:
            logger.info(f'Using existing vector store at {args.output_dir}')
            embedding_function = EmbeddingFunctionWrapper(model_name='all-MiniLM-L6-v2')
            vectorstore = ChromaVectorStore(persist_directory=args.output_dir, embedding_function=embedding_function)

        # Set up RAG pipeline
        rag = setup_rag_pipeline(vectorstore=vectorstore, model_name=args.model)

        # Run interactive session
        interactive_rag_session(rag)

    except Exception as e:
        logger.error(f'Error during RAG testing: {e}', exc_info=True)
        print(f'Error: {e}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())


def fix_pipeline_py():
    """Fix variable naming in pipeline.py."""
    with open('src/llm_rag/rag/pipeline.py', 'r') as f:
        content = f.read()

    # Fix document variables
    content = re.sub(r'\braw_docs\b', 'documents', content)
    content = re.sub(r'\bdocs\b', 'documents', content)
    content = re.sub(r'\bdoc\b(?!_)', 'document', content)
    content = re.sub(r'\bdoc_count\b', 'document_count', content)
    content = re.sub(r'\bdoc_index\b', 'document_index', content)
    content = re.sub(r'\bformatted_docs\b', 'formatted_documents', content)
    content = re.sub(r'\bprocessed_doc\b', 'processed_document', content)

    # Fix config variables
    content = re.sub(r'\bconfidence\b', 'config', content)
    content = re.sub(r'\bconfidence_warning\b', 'config_warning', content)

    with open('src/llm_rag/rag/pipeline.py', 'w') as f:
        f.write(content)


def fix_anti_hallucination_py():
    """Fix variable naming in anti_hallucination.py."""
    with open('src/llm_rag/rag/anti_hallucination.py', 'r') as f:
        content = f.read()

    # Fix embedding variables
    content = re.sub(r'\bresponse_embedding\b', 'response_embeddings', content)
    content = re.sub(r'\bcontext_embedding\b', 'context_embeddings', content)
    content = re.sub(r'\bembedding_similarity\b', 'embeddings_similarity', content)
    content = re.sub(r'\bembed_sim\b', 'embeddings_sim', content)

    # Fix directory variables
    content = re.sub(r'\bstopwords_path\b', 'stopwords_directory', content)

    # Fix config variables
    content = re.sub(r'\bconfidence_level\b', 'config', content)

    with open('src/llm_rag/rag/anti_hallucination.py', 'w') as f:
        f.write(content)


# Run the fixes
fix_pipeline_py()
fix_anti_hallucination_py()
print('Variable naming fixes applied to pipeline.py and anti_hallucination.py')
