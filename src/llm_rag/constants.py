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
DEFAULT_TOP_K: Final[int] = 5

# ===== RAG Parameters =====
# Parameters for the RAG pipeline
DEFAULT_MEMORY_KEY: Final[str] = "chat_history"
# Template for RAG prompts
DEFAULT_PROMPT_TEMPLATE: Final[
    str
] = """You are a helpful assistant that provides accurate information based only on the given context.
If the context doesn't contain enough information to answer the question fully, acknowledge the 
limitations of the available information.
Never make up facts or hallucinate information that is not in the context.

Context:
{context}

Question: {query}

Answer based strictly on the information in the context above. If the context doesn't contain 
relevant information, say "I don't have enough information to answer this question accurately."
"""

SYSTEM_PROMPT = """You are an AI assistant for DIN standards. 
Answer the question based only on the given context.
If you cannot answer the question fully, acknowledge the limitations of the 
available information.
Do not use prior knowledge that is not provided in the context.
"""

SYSTEM_PROMPT_WITH_REASONING = """You are an AI assistant for DIN standards.
First, analyze the question and the provided context.
Then, reason step by step about how to answer the question using only the provided context.
Finally, provide a concise answer based on your reasoning.
If you cannot find sufficient information in the context, say 
"I don't have enough information to answer this question accurately."
"""

# ===== System Parameters =====
# Logging and system configuration
LOG_LEVELS: Final[Dict[str, int]] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
