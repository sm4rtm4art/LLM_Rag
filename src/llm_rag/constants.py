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
    'llama': {
        'use_fast_tokenizer': True,
        'trust_remote_code': True,
        'prompt_template': '<|begin_of_text|><|prompt|>{prompt}<|answer|>',
    },
    'phi': {
        'use_fast_tokenizer': True,
        'trust_remote_code': False,
        'prompt_template': '<|user|>\n{prompt}\n<|assistant|>\n',
    },
    'mistral': {
        'use_fast_tokenizer': True,
        'trust_remote_code': False,
        'prompt_template': '<s>[INST] {prompt} [/INST]',
    },
}

# ===== Vector Store Parameters =====
# Default parameters for document chunking and retrieval
DEFAULT_CHUNK_SIZE: Final[int] = 1000
DEFAULT_CHUNK_OVERLAP: Final[int] = 200
DEFAULT_TOP_K: Final[int] = 5

# ===== RAG Parameters =====
# Parameters for the RAG pipeline
DEFAULT_MEMORY_KEY: Final[str] = 'chat_history'

# System prompts
SYSTEM_PROMPT = (
    'You are a helpful assistant that provides accurate information based on the '
    "context provided. If you don't know the answer, say so."
)

SYSTEM_PROMPT_WITH_REASONING = (
    'You are a helpful assistant that provides accurate information based on the '
    'context provided. First reason step-by-step about the question, then provide '
    "your final answer. If you don't know the answer, say so."
)

# Default prompt template for the RAG system
DEFAULT_PROMPT_TEMPLATE = (
    '\n'
    'You are a helpful assistant that provides accurate information based on '
    'the context provided. If the answer cannot be determined from the context, '
    'acknowledge this limitation.\n'
    '\n'
    'Context:\n'
    '{context}\n'
    '\n'
    'Question: {question}\n'
    '\n'
    'Answer:\n'
)

# ===== System Parameters =====
# Logging and system configuration
LOG_LEVELS: Final[Dict[str, int]] = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}

DEFAULT_LOG_LEVEL: Final[str] = 'INFO'

# Default embedding model
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'

# Default LLM model
DEFAULT_LLM_MODEL = 'TheBloke/Llama-2-7B-Chat-GGUF'

# Default LLM model file
DEFAULT_LLM_MODEL_FILE = 'llama-2-7b-chat.Q4_K_M.gguf'

# Default chunk size
DEFAULT_CHUNK_SIZE = 1000

# Default chunk overlap
DEFAULT_CHUNK_OVERLAP = 200

# Default number of chunks to retrieve
DEFAULT_NUM_CHUNKS = 5

# Default temperature
DEFAULT_TEMPERATURE = 0.7

# Default max tokens
DEFAULT_MAX_TOKENS = 512

# Default vector store path
DEFAULT_VECTOR_STORE_PATH = 'data/vectorstore'

# Default collection name
DEFAULT_COLLECTION_NAME = 'documents'

# Default document directory
DEFAULT_DOCUMENT_DIR = 'data/documents'

# Default document glob pattern
DEFAULT_DOCUMENT_GLOB = '*.pdf'

# Default model directory
DEFAULT_MODEL_DIR = 'models'

# Default log level
DEFAULT_LOG_LEVEL = 'INFO'

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Default log file
DEFAULT_LOG_FILE = 'logs/llm_rag.log'

# Default log file max bytes
DEFAULT_LOG_FILE_MAX_BYTES = 10485760  # 10MB

# Default log file backup count
DEFAULT_LOG_FILE_BACKUP_COUNT = 5

# Default log file encoding
DEFAULT_LOG_FILE_ENCODING = 'utf-8'

# Default log file mode
DEFAULT_LOG_FILE_MODE = 'a'

# Default log file permissions
DEFAULT_LOG_FILE_PERMISSIONS = 0o644

# Default log file directory
DEFAULT_LOG_FILE_DIR = 'logs'

# Default log file name
DEFAULT_LOG_FILE_NAME = 'llm_rag.log'

# Default log file extension
DEFAULT_LOG_FILE_EXTENSION = '.log'

# Default log file date format
DEFAULT_LOG_FILE_DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'

# Default log file name format
DEFAULT_LOG_FILE_NAME_FORMAT = '{name}_{date}{extension}'

# Default log file rotation when
DEFAULT_LOG_FILE_ROTATION_WHEN = 'midnight'

# Default log file rotation interval
DEFAULT_LOG_FILE_ROTATION_INTERVAL = 1

# Default log file rotation backup count
DEFAULT_LOG_FILE_ROTATION_BACKUP_COUNT = 5
