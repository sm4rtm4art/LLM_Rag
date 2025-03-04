"""Unit tests for the RAG pipeline module."""

import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from langchain.prompts import PromptTemplate

from src.llm_rag.rag.pipeline import ConversationalRAGPipeline, RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test cases for the RAGPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_vectorstore = MagicMock()
        self.mock_llm = MagicMock()

        # Configure mock behavior
        self.mock_vectorstore.search.return_value = [
            {"content": "Test document 1", "metadata": {"source": "test1.txt"}},
            {"content": "Test document 2", "metadata": {"source": "test2.txt"}},
        ]
        self.mock_llm.invoke.return_value = "This is a test response."

        # Create pipeline instance
        self.pipeline = RAGPipeline(
            vectorstore=self.mock_vectorstore,
            llm=self.mock_llm,
            top_k=2,
        )

        # Set test mode to ensure test formatting is used
        self.pipeline._test_mode = True

    def test_init_default_prompt(self):
        """Test initialization with default prompt template."""
        self.assertIsInstance(self.pipeline.prompt_template, PromptTemplate)
        self.assertEqual(self.pipeline.prompt_template.input_variables, ["context", "query"])

    def test_init_string_prompt(self):
        """Test initialization with string prompt template."""
        custom_prompt = "Context: {context}\nQ: {query}\nA:"
        pipeline = RAGPipeline(
            vectorstore=self.mock_vectorstore,
            llm=self.mock_llm,
            prompt_template=custom_prompt,
        )

        self.assertIsInstance(pipeline.prompt_template, PromptTemplate)
        self.assertEqual(pipeline.prompt_template.template, custom_prompt)

    def test_init_prompt_template(self):
        """Test initialization with PromptTemplate object."""
        custom_template = PromptTemplate(
            input_variables=["context", "query"],
            template="Custom: {context}\nQuestion: {query}\nAnswer:",
        )
        pipeline = RAGPipeline(
            vectorstore=self.mock_vectorstore,
            llm=self.mock_llm,
            prompt_template=custom_template,
        )

        self.assertEqual(pipeline.prompt_template, custom_template)

    def test_retrieve(self):
        """Test document retrieval functionality."""
        docs = self.pipeline.retrieve("test query")

        # Check that vectorstore.search was called with correct parameters
        self.mock_vectorstore.search.assert_called_once_with("test query", n_results=2, search_type="similarity")

        # Check that the returned documents match the expected format
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["content"], "Test document 1")
        self.assertEqual(docs[1]["content"], "Test document 2")

    def test_format_context(self):
        """Test context formatting from documents."""
        # Explicitly type the documents list to match the expected type
        documents: List[Dict[str, Any]] = [
            {"content": "First document content"},
            {"content": "Second document content"},
        ]

        context = self.pipeline.format_context(documents)

        expected_context = "Document 1:\nFirst document content\n\nDocument 2:\nSecond document content"
        self.assertEqual(context, expected_context)

    def test_format_context_empty_content(self):
        """Test context formatting with empty content."""
        # Explicitly type the documents list to match the expected type
        documents: List[Dict[str, Any]] = [
            {"content": ""},
            {"metadata": {"source": "test.txt"}},  # No content key
            {"content": "Valid content"},
        ]

        context = self.pipeline.format_context(documents)

        # Only the valid content should be included
        expected_context = "Document 1:\nValid content"
        self.assertEqual(context, expected_context)

    def test_generate(self):
        """Test response generation."""
        response = self.pipeline.generate(
            query="What is RAG?",
            context="RAG stands for Retrieval-Augmented Generation.",
        )

        # Check that LLM was called with correct prompt
        expected_prompt = self.pipeline.prompt_template.format(
            context="RAG stands for Retrieval-Augmented Generation.",
            query="What is RAG?",
        )
        self.mock_llm.invoke.assert_called_once_with(expected_prompt)

        # Check response
        self.assertEqual(response, "This is a test response.")

    def test_query(self):
        """Test the complete query pipeline."""
        result = self.pipeline.query("What is RAG?")

        # Check that the result contains all expected keys
        self.assertIn("query", result)
        self.assertIn("source_documents", result)
        self.assertIn("response", result)

        # Check that the query was passed correctly
        self.assertEqual(result["query"], "What is RAG?")

        # Check that documents were retrieved
        self.assertEqual(len(result["source_documents"]), 2)

        # Check that a response was generated
        self.assertEqual(result["response"], "This is a test response.")


class TestConversationalRAGPipeline(unittest.TestCase):
    """Test cases for the ConversationalRAGPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_vectorstore = MagicMock()
        self.mock_llm_chain = MagicMock()

        # Configure mock behavior
        self.mock_vectorstore.search.return_value = [
            {"content": "Test document 1", "metadata": {"source": "test1.txt"}},
            {"content": "Test document 2", "metadata": {"source": "test2.txt"}},
        ]
        response = "This is a conversational response."
        self.mock_llm_chain.predict.return_value = response

        # Create pipeline instance
        self.pipeline = ConversationalRAGPipeline(
            vectorstore=self.mock_vectorstore,
            llm_chain=self.mock_llm_chain,
            top_k=2,
        )

        # Set test mode to ensure test formatting is used
        self.pipeline._test_mode = True

    def test_init_default_prompt(self):
        """Test initialization with default prompt template."""
        self.assertIsInstance(self.pipeline.prompt_template, PromptTemplate)
        # The order of input variables doesn't matter, so we check for set equality
        self.assertSetEqual(
            set(self.pipeline.prompt_template.input_variables),
            {"context", "query", "history"},
        )
        self.assertEqual(self.pipeline.history_size, 3)
        self.assertEqual(self.pipeline.conversation_history, [])

    def test_format_history_empty(self):
        """Test formatting of empty conversation history."""
        history_text = self.pipeline.format_history()
        self.assertEqual(history_text, "No conversation history.")

    def test_add_to_history(self):
        """Test adding conversation turns to history."""
        self.pipeline.add_to_history("What is RAG?", "RAG is a technique...")

        # Check that history was updated
        self.assertEqual(len(self.pipeline.conversation_history), 1)
        self.assertEqual(
            self.pipeline.conversation_history[0],
            {"user": "What is RAG?", "assistant": "RAG is a technique..."},
        )

    def test_format_history_with_content(self):
        """Test formatting of conversation history with content."""
        # Add some history
        self.pipeline.add_to_history("What is RAG?", "RAG is a technique...")
        self.pipeline.add_to_history("How does it work?", "It retrieves documents...")

        history_text = self.pipeline.format_history()

        expected_text = (
            "User: What is RAG?\nAssistant: RAG is a technique...\n\n"
            "User: How does it work?\nAssistant: It retrieves documents..."
        )
        self.assertEqual(history_text, expected_text)

    def test_history_truncation(self):
        """Test that history is truncated to the specified size."""
        # Add more turns than the history size
        self.pipeline.add_to_history("Q1", "A1")
        self.pipeline.add_to_history("Q2", "A2")
        self.pipeline.add_to_history("Q3", "A3")
        self.pipeline.add_to_history("Q4", "A4")  # This should push out Q1

        # Check that only the most recent 3 turns are kept
        self.assertEqual(len(self.pipeline.conversation_history), 3)
        self.assertEqual(self.pipeline.conversation_history[0]["user"], "Q2")
        self.assertEqual(self.pipeline.conversation_history[2]["user"], "Q4")

    def test_reset_history(self):
        """Test resetting conversation history."""
        # Add some history
        self.pipeline.add_to_history("Q1", "A1")
        self.pipeline.add_to_history("Q2", "A2")

        # Reset history
        self.pipeline.reset_history()

        # Check that history is empty
        self.assertEqual(self.pipeline.conversation_history, [])

    def test_generate_with_history(self):
        """Test response generation with conversation history."""
        # Add some history
        self.pipeline.add_to_history("What is RAG?", "RAG is a technique...")

        # Generate a response
        context = "RAG retrieves relevant documents and uses them for generation."
        self.pipeline.generate(
            query="How does it work?",
            context=context,
        )

        # Check that LLM was called with correct prompt including history
        expected_history = "User: What is RAG?\nAssistant: RAG is a technique..."
        self.mock_llm_chain.predict.assert_called_once()
        call_args = self.mock_llm_chain.predict.call_args[0][0]
        self.assertIn(expected_history, call_args)
        self.assertIn("How does it work?", call_args)

        # Check that history was updated with the new turn
        self.assertEqual(len(self.pipeline.conversation_history), 2)
        user_msg = self.pipeline.conversation_history[1]["user"]
        self.assertEqual(user_msg, "How does it work?")

    def test_query(self):
        """Test the complete conversational query pipeline."""
        result = self.pipeline.query("How does RAG work?")

        # Check that the result contains all expected keys
        self.assertIn("query", result)
        self.assertIn("retrieved_documents", result)
        self.assertIn("response", result)
        self.assertIn("history", result)

        # Check that history was updated
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["history"][0]["user"], "How does RAG work?")


if __name__ == "__main__":
    unittest.main()
