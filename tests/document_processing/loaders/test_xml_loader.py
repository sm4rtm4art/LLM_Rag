import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.llm_rag.document_processing.loaders.file_loaders import XMLLoader


class TestXMLLoader(unittest.TestCase):
    """Test cases for the XMLLoader class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary XML file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.xml_file_path = Path(self.temp_dir.name) / 'test.xml'

        # Simple XML for testing
        self.xml_content = """
        <root>
            <metadata>
                <title attr1="value1">Test Document</title>
                <author>Test Author</author>
                <date>2023-05-01</date>
            </metadata>
            <content>
                <section id="section1">
                    <heading>Introduction</heading>
                    <paragraph>This is the first paragraph.</paragraph>
                    <paragraph>This is the second paragraph.</paragraph>
                </section>
                <section id="section2">
                    <heading>Chapter 1</heading>
                    <paragraph>Content of chapter 1.</paragraph>
                </section>
            </content>
        </root>
        """

        with open(self.xml_file_path, 'w') as f:
            f.write(self.xml_content)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with file path
        loader1 = XMLLoader(file_path=self.xml_file_path)
        self.assertEqual(loader1.file_path, self.xml_file_path)
        self.assertIsNone(loader1.content_tags)

        # Test with content tags
        loader2 = XMLLoader(file_path=self.xml_file_path, content_tags=['paragraph'])
        self.assertEqual(loader2.content_tags, ['paragraph'])

        # Test with metadata tags
        loader3 = XMLLoader(file_path=self.xml_file_path, metadata_tags=['title', 'author'])
        self.assertEqual(loader3.metadata_tags, ['title', 'author'])

    def test_load_basic(self):
        """Test basic loading of an XML file."""
        loader = XMLLoader(file_path=self.xml_file_path)
        documents = loader.load()

        # Should return a list with one document
        self.assertEqual(len(documents), 1)

        # Document should have content and metadata
        doc = documents[0]
        self.assertIn('content', doc)
        self.assertIn('metadata', doc)

        # Check metadata
        self.assertEqual(doc['metadata']['filename'], 'test.xml')
        self.assertEqual(doc['metadata']['filetype'], 'xml')

        # Check content includes text
        self.assertIn('Introduction', doc['content'])
        self.assertIn('This is the first paragraph', doc['content'])

    def test_load_from_file(self):
        """Test loading from a file path."""
        loader = XMLLoader()  # No file path provided during initialization
        documents = loader.load_from_file(self.xml_file_path)

        # Should return a list with one document
        self.assertEqual(len(documents), 1)
        self.assertIn('This is the first paragraph', documents[0]['content'])

    def test_load_with_content_tags(self):
        """Test loading with specific content tags."""
        loader = XMLLoader(file_path=self.xml_file_path, content_tags=['paragraph'])
        documents = loader.load()

        # Content should only include text from paragraph tags
        content = documents[0]['content']
        self.assertIn('paragraph: This is the first paragraph', content)
        self.assertIn('paragraph: This is the second paragraph', content)
        self.assertIn('paragraph: Content of chapter 1', content)

        # Should not include other content
        self.assertNotIn('Introduction', content)

    def test_load_with_metadata_tags(self):
        """Test loading with metadata tags."""
        loader = XMLLoader(file_path=self.xml_file_path, metadata_tags=['title', 'author', 'date'])
        documents = loader.load()

        # Check metadata was extracted
        metadata = documents[0]['metadata']
        self.assertEqual(metadata['title'], 'Test Document')
        self.assertEqual(metadata['author'], 'Test Author')
        self.assertEqual(metadata['date'], '2023-05-01')

        # Check attribute was included
        self.assertEqual(metadata['title_attr1'], 'value1')

    def test_split_by_tag(self):
        """Test splitting into multiple documents by tag."""
        loader = XMLLoader(file_path=self.xml_file_path, split_by_tag='section')
        documents = loader.load()

        # Should create one document per section
        self.assertEqual(len(documents), 2)

        # Check first document content
        self.assertIn('Introduction', documents[0]['content'])
        self.assertIn('This is the first paragraph', documents[0]['content'])

        # Check second document content
        self.assertIn('Chapter 1', documents[1]['content'])
        self.assertIn('Content of chapter 1', documents[1]['content'])

        # Check index metadata
        self.assertEqual(documents[0]['metadata']['index'], 0)
        self.assertEqual(documents[1]['metadata']['index'], 1)

    def test_include_tags_in_text(self):
        """Test including tags in the extracted text."""
        loader = XMLLoader(file_path=self.xml_file_path, text_tag='paragraph', include_tags_in_text=True)
        documents = loader.load()

        # Content should include XML tags
        content = documents[0]['content']
        self.assertIn('<paragraph>', content)
        self.assertIn('</paragraph>', content)

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        loader = XMLLoader(file_path='nonexistent.xml')
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_text_tag(self):
        """Test using text_tag parameter."""
        loader = XMLLoader(file_path=self.xml_file_path, text_tag='heading')
        documents = loader.load()

        # Content should only include text from heading tags
        content = documents[0]['content']
        self.assertIn('Introduction', content)
        self.assertIn('Chapter 1', content)

        # Should not include paragraph content
        self.assertNotIn('This is the first paragraph', content)

    @mock.patch('src.llm_rag.document_processing.loaders.file_loaders.XML_AVAILABLE', False)
    def test_xml_not_available(self):
        """Test behavior when XML libraries are not available."""
        loader = XMLLoader(file_path=self.xml_file_path)
        with self.assertRaises(ImportError):
            loader.load()


if __name__ == '__main__':
    unittest.main()
