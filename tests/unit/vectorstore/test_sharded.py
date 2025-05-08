"""Unit tests for the ShardedChromaVectorStore class."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from llm_rag.vectorstore.chroma import ChromaVectorStore
from llm_rag.vectorstore.sharded import ShardedChromaVectorStore


class TestShardedChromaVectorStore(unittest.TestCase):
    """Test cases for the ShardedChromaVectorStore class."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the temporary directory after testing."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that ShardedChromaVectorStore initializes correctly."""
        # Arrange & Act
        with patch.object(ChromaVectorStore, '__init__', return_value=None) as mock_init:
            store = ShardedChromaVectorStore(shard_capacity=5000, base_persist_directory=self.temp_dir, max_workers=2)

            # Assert
            self.assertEqual(store.shard_capacity, 5000)
            self.assertEqual(store.base_persist_directory, self.temp_dir)
            self.assertEqual(store.max_workers, 2)
            self.assertEqual(len(store.shards), 1)
            mock_init.assert_called_once()

    def test_create_new_shard(self):
        """Test the creation of a new shard."""
        # Arrange
        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            store = ShardedChromaVectorStore(base_persist_directory=self.temp_dir)
            initial_shard_count = len(store.shards)

            # Act
            store._create_new_shard()

            # Assert
            self.assertEqual(len(store.shards), initial_shard_count + 1)
            shard_dir = os.path.join(self.temp_dir, f'shard_{initial_shard_count}')
            self.assertTrue(os.path.exists(shard_dir))

    def test_add_documents_within_capacity(self):
        """Test adding documents when capacity is not exceeded."""
        # Arrange
        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            mock_shard = MagicMock()
            mock_shard.get_collection_size.return_value = 0
            mock_shard.add_documents = MagicMock()

            store = ShardedChromaVectorStore(shard_capacity=100, base_persist_directory=self.temp_dir)
            # Replace the auto-created shard with our mock
            store.shards = [mock_shard]

            documents = ['doc1', 'doc2']
            metadatas = [{'source': 'test1'}, {'source': 'test2'}]

            # Act
            store.add_documents(documents, metadatas)

            # Assert
            self.assertEqual(len(store.shards), 1)
            mock_shard.add_documents.assert_called_once_with(documents, metadatas, None)

    def test_add_documents_exceeding_capacity(self):
        """Test adding documents when capacity is exceeded."""
        # Arrange
        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            mock_shard = MagicMock()
            mock_shard.get_collection_size.return_value = 95
            mock_shard.add_documents = MagicMock()

            store = ShardedChromaVectorStore(shard_capacity=100, base_persist_directory=self.temp_dir)

            # Replace the auto-created shard with our mock
            store.shards = [mock_shard]

            # We need to patch the _create_new_shard method
            with patch.object(store, '_create_new_shard') as mock_create:
                # Act
                documents = ['doc' + str(i) for i in range(10)]
                store.add_documents(documents)

                # Assert
                self.assertEqual(len(store.shards), 1)  # We mock _create_new_shard, so no shard is added
                mock_create.assert_called_once()
                # Should have called add_documents on the mock shard
                mock_shard.add_documents.assert_called_once()

    def test_add_documents_with_dict_format(self):
        """Test adding documents in dictionary format with content and metadata."""
        # Arrange
        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            mock_shard = MagicMock()
            mock_shard.get_collection_size.return_value = 0
            mock_shard.add_documents = MagicMock()

            store = ShardedChromaVectorStore(base_persist_directory=self.temp_dir)
            # Replace the auto-created shard with our mock
            store.shards = [mock_shard]

            documents = [
                {'content': 'doc1', 'metadata': {'source': 'test1'}},
                {'content': 'doc2', 'metadata': {'source': 'test2'}},
            ]

            # Act
            store.add_documents(documents)

            # Assert
            # Should extract content and metadata
            expected_docs = ['doc1', 'doc2']
            expected_meta = [{'source': 'test1'}, {'source': 'test2'}]
            mock_shard.add_documents.assert_called_once()
            # Check that the first arg is docs and second is metadata
            args = mock_shard.add_documents.call_args[0]
            self.assertEqual(args[0], expected_docs)
            self.assertEqual(args[1], expected_meta)

    def test_search_aggregates_results(self):
        """Test that search aggregates results from multiple shards."""
        # Arrange
        mock_shard1 = MagicMock()
        mock_shard1.search.return_value = [{'content': 'doc1', 'score': 0.1, 'metadata': {'source': 'shard1'}}]

        mock_shard2 = MagicMock()
        mock_shard2.search.return_value = [{'content': 'doc2', 'score': 0.2, 'metadata': {'source': 'shard2'}}]

        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            store = ShardedChromaVectorStore(base_persist_directory=self.temp_dir)
            # Replace the shards with our mocks
            store.shards = [mock_shard1, mock_shard2]

            # Act
            results = store.search('test query', n_results=2)

            # Assert
            self.assertEqual(len(results), 2)
            # Should be sorted by score in ascending order (lower is better for distance metrics)
            self.assertEqual(results[0]['content'], 'doc1')  # Lower score (0.1) is better
            self.assertEqual(results[1]['content'], 'doc2')  # Higher score (0.2) is worse

            # Each shard's search should be called
            mock_shard1.search.assert_called_once_with('test query', 2)
            mock_shard2.search.assert_called_once_with('test query', 2)

    def test_search_limits_results(self):
        """Test that search limits results to n_results."""
        # Arrange
        mock_shard1 = MagicMock()
        mock_shard1.search.return_value = [{'content': 'doc1', 'score': 0.1}, {'content': 'doc3', 'score': 0.3}]

        mock_shard2 = MagicMock()
        mock_shard2.search.return_value = [{'content': 'doc2', 'score': 0.2}, {'content': 'doc4', 'score': 0.4}]

        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            store = ShardedChromaVectorStore(base_persist_directory=self.temp_dir)
            store.shards = [mock_shard1, mock_shard2]

            # Act
            results = store.search('test query', n_results=3)

            # Assert
            self.assertEqual(len(results), 3)
            # Should have top 3 results sorted by score (lower is better)
            self.assertEqual(results[0]['content'], 'doc1')  # 0.1
            self.assertEqual(results[1]['content'], 'doc2')  # 0.2
            self.assertEqual(results[2]['content'], 'doc3')  # 0.3

    def test_count_aggregates_from_shards(self):
        """Test that count aggregates document counts from all shards."""
        # Arrange
        mock_shard1 = MagicMock()
        mock_shard1.get_collection_size.return_value = 100

        mock_shard2 = MagicMock()
        mock_shard2.get_collection_size.return_value = 200

        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            store = ShardedChromaVectorStore(base_persist_directory=self.temp_dir)
            store.shards = [mock_shard1, mock_shard2]

            # Act
            total_count = store.count()

            # Assert
            self.assertEqual(total_count, 300)

    def test_delete_collection(self):
        """Test that delete_collection removes all shards."""
        # Arrange
        mock_shard1 = MagicMock()
        mock_shard2 = MagicMock()

        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            store = ShardedChromaVectorStore(base_persist_directory=self.temp_dir)
            store.shards = [mock_shard1, mock_shard2]

            # Act
            store.delete_collection()

            # Assert
            mock_shard1.delete_collection.assert_called_once()
            mock_shard2.delete_collection.assert_called_once()
            self.assertEqual(len(store.shards), 0)

    def test_as_retriever(self):
        """Test the as_retriever method creates a retriever."""
        # Arrange
        with patch.object(ChromaVectorStore, '__init__', return_value=None):
            with patch('llm_rag.vectorstore.chroma.ChromaRetriever') as mock_retriever:
                store = ShardedChromaVectorStore(base_persist_directory=self.temp_dir)

                # Act
                store.as_retriever(search_kwargs={'k': 10})

                # Assert
                mock_retriever.assert_called_once_with(vectorstore=store, search_kwargs={'k': 10})


if __name__ == '__main__':
    unittest.main()
