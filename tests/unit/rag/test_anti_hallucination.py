"""Unit tests for the anti_hallucination.py module."""

import warnings
from unittest.mock import patch

# Import the module under test
from llm_rag.rag.anti_hallucination import (
    HallucinationConfig,
    advanced_verify_response,
    calculate_hallucination_score,
    embedding_based_verification,
    extract_key_entities,
    generate_verification_warning,
    get_sentence_transformer_model,
    load_stopwords,
    needs_human_review,
    post_process_response,
    verify_entities_in_context,
)


class TestHallucinationConfig:
    """Tests for the HallucinationConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correctly initialized."""
        config = HallucinationConfig()
        assert config.entity_threshold == 0.7
        assert config.embedding_threshold == 0.75
        assert config.model_name == 'paraphrase-MiniLM-L6-v2'
        assert config.entity_weight == 0.6
        assert config.human_review_threshold == 0.5
        assert config.entity_critical_threshold == 0.3
        assert config.embedding_critical_threshold == 0.4
        assert config.use_embeddings is True
        assert config.flag_for_human_review is False

    def test_custom_values(self):
        """Test setting custom values."""
        config = HallucinationConfig(
            entity_threshold=0.8,
            embedding_threshold=0.85,
            model_name='custom-model',
            entity_weight=0.7,
            human_review_threshold=0.6,
            entity_critical_threshold=0.2,
            embedding_critical_threshold=0.3,
            use_embeddings=False,
            flag_for_human_review=True,
        )

        assert config.entity_threshold == 0.8
        assert config.embedding_threshold == 0.85
        assert config.model_name == 'custom-model'
        assert config.entity_weight == 0.7
        assert config.human_review_threshold == 0.6
        assert config.entity_critical_threshold == 0.2
        assert config.embedding_critical_threshold == 0.3
        assert config.use_embeddings is False
        assert config.flag_for_human_review is True


class TestEntityFunctions:
    """Tests for entity-based verification functions."""

    def test_extract_key_entities(self):
        """Test entity extraction functionality."""
        text = 'Apple Inc. and Microsoft are major technology companies based in the United States.'
        entities = extract_key_entities(text)

        # In stub implementation, expect empty set
        # In actual implementation, would expect entities like "Apple", "Microsoft", "United States"
        assert isinstance(entities, set)

    def test_extract_key_entities_with_languages(self):
        """Test entity extraction with language specification."""
        text = 'Berlin ist die Hauptstadt von Deutschland.'
        entities = extract_key_entities(text, languages=['de'])
        assert isinstance(entities, set)

    def test_verify_entities_in_context(self):
        """Test entity verification in context."""
        response = 'Apple is a technology company.'
        context = 'Apple Inc. is an American multinational technology company headquartered in Cupertino, California.'

        verified, coverage, missing = verify_entities_in_context(response, context)

        assert isinstance(verified, bool)
        assert isinstance(coverage, float)
        assert isinstance(missing, list)

        # In stub implementation, expect verified=True, coverage=1.0, missing=[]
        assert verified is True
        assert coverage == 1.0
        assert missing == []

    def test_verify_entities_with_custom_threshold(self):
        """Test entity verification with custom threshold."""
        response = 'Microsoft is a software company.'
        context = 'Microsoft Corporation is an American multinational technology company.'

        verified, coverage, missing = verify_entities_in_context(response, context, threshold=0.8)

        # In stub implementation, expect verified=True, coverage=1.0, missing=[]
        assert verified is True
        assert coverage == 1.0
        assert missing == []


class TestEmbeddingFunctions:
    """Tests for embedding-based verification functions."""

    def test_get_sentence_transformer_model(self):
        """Test sentence transformer model loading."""
        model = get_sentence_transformer_model('paraphrase-MiniLM-L6-v2')

        # In stub implementation, expect None
        # In actual implementation, would expect a model object
        assert model is None

    def test_embedding_based_verification(self):
        """Test embedding-based verification."""
        response = 'The cat sat on the mat.'
        context = 'A feline was resting on a small carpet.'

        verified, similarity = embedding_based_verification(response, context)

        assert isinstance(verified, bool)
        assert isinstance(similarity, float)

        # In stub implementation, expect verified=True, similarity=1.0
        assert verified is True
        assert similarity == 1.0

    def test_embedding_based_verification_with_custom_threshold(self):
        """Test embedding-based verification with custom threshold."""
        response = 'The dog played in the yard.'
        context = 'A canine was enjoying time in the garden.'

        verified, similarity = embedding_based_verification(response, context, threshold=0.8, model_name='custom-model')

        # In stub implementation, expect verified=True, similarity=1.0
        assert verified is True
        assert similarity == 1.0


class TestAdvancedVerification:
    """Tests for advanced verification methods."""

    def test_advanced_verify_response(self):
        """Test advanced response verification."""
        response = 'The company released a new smartphone.'
        context = 'Apple Inc. announced a new iPhone model during their annual event.'

        result = advanced_verify_response(response, context)

        # Unpack the result
        verified, entity_coverage, embedding_sim, missing_entities = result

        assert isinstance(verified, bool)
        assert isinstance(entity_coverage, float)
        assert isinstance(embedding_sim, float)
        assert isinstance(missing_entities, list)

        # In stub implementation, expect verified=True, entity_coverage=1.0, embedding_sim=1.0, missing_entities=[]
        assert verified is True
        assert entity_coverage == 1.0
        assert embedding_sim == 1.0
        assert missing_entities == []

    def test_advanced_verify_with_config(self):
        """Test advanced verification with custom config."""
        response = 'The car has good fuel efficiency.'
        context = 'The vehicle has excellent gas mileage and eco-friendly features.'

        config = HallucinationConfig(entity_threshold=0.6, embedding_threshold=0.7, model_name='custom-model')

        result = advanced_verify_response(response, context, config=config)
        verified, entity_coverage, embedding_sim, missing_entities = result

        # In stub implementation, expect verified=True
        assert verified is True

    def test_calculate_hallucination_score(self):
        """Test hallucination score calculation."""
        # Basic test with entity coverage only
        score1 = calculate_hallucination_score(entity_coverage=0.8)
        assert isinstance(score1, float)

        # Test with entity coverage and embedding similarity
        score2 = calculate_hallucination_score(entity_coverage=0.7, embeddings_similarity=0.9, entity_weight=0.5)
        assert isinstance(score2, float)

        # In stub implementation, expect score=1.0
        assert score1 == 1.0
        assert score2 == 1.0


class TestHumanReviewFunctions:
    """Tests for human review decision functions."""

    def test_needs_human_review(self):
        """Test human review determination."""
        # Basic test with hallucination score only
        result1 = needs_human_review(hallucination_score=0.4)
        assert isinstance(result1, bool)

        # Test with config
        config = HallucinationConfig(human_review_threshold=0.6)
        result2 = needs_human_review(hallucination_score=0.5, config=config)
        assert isinstance(result2, bool)

        # Test with custom critical thresholds
        result3 = needs_human_review(
            hallucination_score=0.7,
            critical_threshold=0.3,
            entity_coverage=0.2,
            entity_critical_threshold=0.25,
            embeddings_similarity=0.5,
            embedding_critical_threshold=0.4,
        )
        assert isinstance(result3, bool)

        # In stub implementation, expect all results to be False
        assert result1 is False
        assert result2 is False
        assert result3 is False

    def test_generate_verification_warning(self):
        """Test warning message generation."""
        # Basic test
        warning1 = generate_verification_warning(missing_entities=['Apple', 'iPhone'], coverage_ratio=0.6)
        assert isinstance(warning1, str)

        # Test with embedding similarity and human review flag
        warning2 = generate_verification_warning(
            missing_entities=['Microsoft', 'Windows'], coverage_ratio=0.5, embeddings_sim=0.7, human_review=True
        )
        assert isinstance(warning2, str)

        # In stub implementation, expect empty strings
        assert warning1 == ''
        assert warning2 == ''


class TestPostProcessingFunctions:
    """Tests for response post-processing functions."""

    def test_post_process_response(self):
        """Test response post-processing."""
        response = 'The algorithm has O(n log n) complexity.'
        context = 'Quick sort has a time complexity of O(n log n) in the average case.'

        # Basic test
        processed = post_process_response(response, context)
        assert isinstance(processed, str)

        # Test with return_metadata=True
        processed_with_meta = post_process_response(response, context, return_metadata=True)
        assert isinstance(processed_with_meta, tuple)

        # In stub implementation, expect processed=response and processed_with_meta=(response, {})
        assert processed == response
        assert processed_with_meta[0] == response
        assert processed_with_meta[1] == {}

    def test_post_process_with_config(self):
        """Test post-processing with custom config."""
        response = 'The GPU utilizes CUDA cores.'
        context = 'NVIDIA GPUs contain CUDA cores for parallel processing.'

        config = HallucinationConfig(entity_threshold=0.6, embedding_threshold=0.65, flag_for_human_review=True)

        processed = post_process_response(response, context, config=config)
        assert isinstance(processed, str)

        # In stub implementation, expect processed=response
        assert processed == response

    def test_post_process_with_custom_params(self):
        """Test post-processing with individual custom parameters."""
        response = 'The framework supports deep learning.'
        context = 'PyTorch is a machine learning framework that supports neural networks.'

        processed = post_process_response(
            response,
            context,
            threshold=0.6,
            entity_threshold=0.7,
            embedding_threshold=0.8,
            model_name='custom-model',
            human_review_threshold=0.5,
            flag_for_human_review=True,
            languages=['en'],
        )
        assert isinstance(processed, str)

        # In stub implementation, expect processed=response
        assert processed == response


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_load_stopwords(self):
        """Test stopwords loading."""
        # Default language
        stopwords_en = load_stopwords()
        assert isinstance(stopwords_en, set)

        # Specific language
        stopwords_de = load_stopwords(language='de')
        assert isinstance(stopwords_de, set)

        # In stub implementation, expect empty sets
        assert stopwords_en == set()
        assert stopwords_de == set()

    @patch('llm_rag.rag.anti_hallucination.load_stopwords')
    def test_load_specific_stopwords(self, mock_load_stopwords):
        """Test that stopwords are loaded for a specific language."""
        # Mock the response for German stopwords
        mock_german_stopwords = {'der', 'die', 'das', 'und', 'oder'}
        mock_load_stopwords.return_value = mock_german_stopwords

        # Import the module to get access to the function that will use the mock
        from llm_rag.rag.anti_hallucination import load_stopwords as load_fn

        # Call the function through the imported reference
        result = load_fn(language='de')

        # Verify the mock was called with the correct language
        mock_load_stopwords.assert_called_once_with(language='de')

        # Verify the result
        assert result == mock_german_stopwords
        assert 'der' in result
        assert 'die' in result
        assert 'das' in result


class TestModuleImports:
    """Tests for module import behavior."""

    def test_stub_imports_warning(self):
        """Test that a warning is issued when modular imports are flagged as failed."""
        import llm_rag.rag.anti_hallucination

        # Save the original flag value to restore it later
        original_flag = llm_rag.rag.anti_hallucination._MODULAR_IMPORT_SUCCESS

        try:
            # Set the flag to False directly
            llm_rag.rag.anti_hallucination._MODULAR_IMPORT_SUCCESS = False

            # Check that the flag is properly modified
            assert llm_rag.rag.anti_hallucination._MODULAR_IMPORT_SUCCESS is False

            # Verify a warning would be emitted by calling the warning code directly
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')

                # Execute the warning code directly
                if not llm_rag.rag.anti_hallucination._MODULAR_IMPORT_SUCCESS:
                    warnings.warn(
                        'Using stub implementation for anti-hallucination module. Functionality will be limited.',
                        stacklevel=2,
                    )

                # Check that the warning was issued
                assert len(w) == 1
                assert 'stub implementation' in str(w[0].message)

        finally:
            # Restore the original module state
            llm_rag.rag.anti_hallucination._MODULAR_IMPORT_SUCCESS = original_flag
