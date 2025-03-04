"""Constants for the LLM RAG system.

This module contains constants used throughout the LLM RAG system.
Constants are organized by category and include type annotations for better IDE support.
"""

from typing import Dict, Final, Union

# ===== Model Parameters =====
# Default parameters for language models
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_TOP_P: Final[float] = 0.95
DEFAULT_MAX_TOKENS: Final[int] = 512
DEFAULT_REPETITION_PENALTY: Final[float] = 1.1

# Model-specific configurations
MODEL_CONFIGS: Final[Dict[str, Dict[str, Union[str, int, float, bool]]]] = {
    "llama": {
        "use_fast_tokenizer": True,
        "trust_remote_code": True,
        "prompt_template": "<|begin_of_text|><|prompt|>{prompt}<|answer|>",
    },
    "phi": {
        "use_fast_tokenizer": True,
        "trust_remote_code": False,
        "prompt_template": "<|user|>\n{prompt}\n<|assistant|>\n",
    },
    "mistral": {
        "use_fast_tokenizer": True,
        "trust_remote_code": False,
        "prompt_template": "<s>[INST] {prompt} [/INST]",
    },
}

# ===== Vector Store Parameters =====
# Default parameters for document chunking and retrieval
DEFAULT_CHUNK_SIZE: Final[int] = 1000
DEFAULT_CHUNK_OVERLAP: Final[int] = 200
DEFAULT_TOP_K: Final[int] = 3

# ===== RAG Parameters =====
# Parameters for the RAG pipeline
DEFAULT_MEMORY_KEY: Final[str] = "chat_history"
# Template for RAG prompts
DEFAULT_PROMPT_TEMPLATE: Final[str] = """Answer the question based on the
following context:

Context:
{context}

Question: {question}

Answer:"""

# ===== System Parameters =====
# Logging and system configuration
LOG_LEVELS: Final[Dict[str, int]] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
