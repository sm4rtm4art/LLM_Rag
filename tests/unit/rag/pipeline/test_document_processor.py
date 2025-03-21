"""Unit tests for the document_processor module in the RAG pipeline."""

from llm_rag.rag.pipeline.document_processor import _process_document, _process_documents


class MockLangchainDocument:
    """Mock LangChain document for testing."""

    def __init__(self, page_content, metadata):
        """Initialize with page content and metadata."""
        self.page_content = page_content
        self.metadata = metadata


class TestDocumentProcessor:
    """Tests for document processor module."""

    def test_process_document_langchain(self):
        """Test processing a LangChain document."""
        # Create a mock LangChain document
        doc = MockLangchainDocument(page_content="Test content", metadata={"source": "test.txt"})

        # Process the document
        result = _process_document(doc)

        # Verify the result
        assert result is not None
        assert result["content"] == "Test content"
        assert result["metadata"]["source"] == "test.txt"

    def test_process_document_dict(self):
        """Test processing a dictionary document."""
        # Create a dictionary document
        doc = {"content": "Test content", "metadata": {"source": "test.txt"}}

        # Process the document
        result = _process_document(doc)

        # Verify the result
        assert result is not None
        assert result["content"] == "Test content"
        assert result["metadata"]["source"] == "test.txt"

    def test_process_document_dict_no_metadata(self):
        """Test processing a dictionary document without metadata."""
        # Create a dictionary document without metadata
        doc = {"content": "Test content"}

        # Process the document
        result = _process_document(doc)

        # Verify the result
        assert result is not None
        assert result["content"] == "Test content"
        assert isinstance(result["metadata"], dict)
        assert len(result["metadata"]) == 0

    def test_process_document_dict_non_dict_metadata(self):
        """Test processing a dictionary document with non-dict metadata."""
        # Create a dictionary document with string metadata
        doc = {"content": "Test content", "metadata": "source: test.txt"}

        # Process the document
        result = _process_document(doc)

        # Verify the result
        assert result is not None
        assert result["content"] == "Test content"
        assert result["metadata"]["raw_metadata"] == "source: test.txt"

    def test_process_document_langchain_style_dict(self):
        """Test processing a dictionary in LangChain style."""
        # Create a dictionary document in LangChain style
        doc = {"page_content": "Test content", "metadata": {"source": "test.txt"}}

        # Process the document
        result = _process_document(doc)

        # Verify the result
        assert result is not None
        assert result["content"] == "Test content"
        assert result["metadata"]["source"] == "test.txt"

    def test_process_document_unsupported(self):
        """Test processing an unsupported document type."""
        # The actual implementation treats strings as content,
        # so the result will be a document, not None
        result = _process_document("This is not a supported document")

        # Verify the result has expected format
        assert result is not None
        assert result["content"] == "This is not a supported document"
        assert isinstance(result["metadata"], dict)

    def test_process_document_error(self):
        """Test processing a document that raises an error."""

        # Create a mock that will raise an exception when processed
        class ErrorDocument:
            def __getattribute__(self, name):
                raise Exception("Test exception")

        # Process the document
        result = _process_document(ErrorDocument())

        # Verify the result is None
        assert result is None

    def test_process_documents_list(self):
        """Test processing a list of documents."""
        # Create a list of documents
        docs = [
            {"content": "Content 1", "metadata": {"source": "doc1.txt"}},
            {"content": "Content 2", "metadata": {"source": "doc2.txt"}},
        ]

        # Process the documents
        result = _process_documents(docs)

        # Verify the result
        assert len(result) == 2
        assert result[0]["content"] == "Content 1"
        assert result[0]["metadata"]["source"] == "doc1.txt"
        assert result[1]["content"] == "Content 2"
        assert result[1]["metadata"]["source"] == "doc2.txt"

    def test_process_documents_generator(self):
        """Test processing a generator of documents."""

        # Create a generator function
        def doc_generator():
            yield {"content": "Content 1", "metadata": {"source": "doc1.txt"}}
            yield {"content": "Content 2", "metadata": {"source": "doc2.txt"}}

        # Process the documents
        result = _process_documents(doc_generator())

        # Verify the result
        assert len(result) == 2
        assert result[0]["content"] == "Content 1"
        assert result[0]["metadata"]["source"] == "doc1.txt"
        assert result[1]["content"] == "Content 2"
        assert result[1]["metadata"]["source"] == "doc2.txt"

    def test_process_documents_single_doc(self):
        """Test processing a single document."""
        # Create a single document
        doc = {"content": "Test content", "metadata": {"source": "test.txt"}}

        # Process the document
        result = _process_documents(doc)

        # Verify the result
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["metadata"]["source"] == "test.txt"

    def test_process_documents_empty(self):
        """Test processing an empty list of documents."""
        # Process an empty list
        result = _process_documents([])

        # Verify the result is an empty list
        assert result == []
