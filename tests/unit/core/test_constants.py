"""Tests for the constants module."""

import unittest

from src.llm_rag.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEMORY_KEY,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    LOG_LEVELS,
    MODEL_CONFIGS,
)


class TestConstants(unittest.TestCase):
    """Tests for the constants module."""

    def test_model_parameters(self):
        """Test that model parameters are correctly defined."""
        self.assertEqual(DEFAULT_TEMPERATURE, 0.7)
        self.assertEqual(DEFAULT_TOP_P, 0.95)
        self.assertEqual(DEFAULT_MAX_TOKENS, 512)
        self.assertEqual(DEFAULT_REPETITION_PENALTY, 1.1)

    def test_model_configs(self):
        """Test that model configurations are correctly defined."""
        # Check that required models are present
        self.assertIn("llama", MODEL_CONFIGS)
        self.assertIn("phi", MODEL_CONFIGS)
        self.assertIn("mistral", MODEL_CONFIGS)

        # Check llama config
        llama_config = MODEL_CONFIGS["llama"]
        self.assertTrue(llama_config["use_fast_tokenizer"])
        self.assertTrue(llama_config["trust_remote_code"])
        self.assertEqual(llama_config["prompt_template"], "<|begin_of_text|><|prompt|>{prompt}<|answer|>")

        # Check phi config
        phi_config = MODEL_CONFIGS["phi"]
        self.assertTrue(phi_config["use_fast_tokenizer"])
        self.assertFalse(phi_config["trust_remote_code"])
        self.assertEqual(phi_config["prompt_template"], "<|user|>\n{prompt}\n<|assistant|>\n")

        # Check mistral config
        mistral_config = MODEL_CONFIGS["mistral"]
        self.assertTrue(mistral_config["use_fast_tokenizer"])
        self.assertFalse(mistral_config["trust_remote_code"])
        self.assertEqual(mistral_config["prompt_template"], "<s>[INST] {prompt} [/INST]")

    def test_vector_store_parameters(self):
        """Test that vector store parameters are correctly defined."""
        self.assertEqual(DEFAULT_CHUNK_SIZE, 1000)
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, 200)
        self.assertEqual(DEFAULT_TOP_K, 3)

    def test_rag_parameters(self):
        """Test that RAG parameters are correctly defined."""
        self.assertEqual(DEFAULT_MEMORY_KEY, "chat_history")
        self.assertIn("context", DEFAULT_PROMPT_TEMPLATE)
        self.assertIn("question", DEFAULT_PROMPT_TEMPLATE)

    def test_system_parameters(self):
        """Test that system parameters are correctly defined."""
        self.assertEqual(DEFAULT_LOG_LEVEL, "INFO")
        self.assertEqual(LOG_LEVELS["DEBUG"], 10)
        self.assertEqual(LOG_LEVELS["INFO"], 20)
        self.assertEqual(LOG_LEVELS["WARNING"], 30)
        self.assertEqual(LOG_LEVELS["ERROR"], 40)
        self.assertEqual(LOG_LEVELS["CRITICAL"], 50)


if __name__ == "__main__":
    unittest.main()
