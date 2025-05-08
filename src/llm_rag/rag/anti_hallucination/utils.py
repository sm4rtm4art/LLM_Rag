"""Utility functions for anti-hallucination module.

This module provides utility functions for working with stopwords and caching
models for the anti-hallucination module.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Set

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# Global cache for SentenceTransformer models
_MODEL_CACHE: Dict[str, Any] = {}

# Default stopwords for supported languages
_DEFAULT_STOPWORDS = {
    'en': {
        'a',
        'an',
        'the',
        'and',
        'or',
        'but',
        'if',
        'then',
        'else',
        'when',
        'at',
        'from',
        'by',
        'for',
        'with',
        'about',
        'against',
        'between',
        'into',
        'through',
        'during',
        'before',
        'after',
        'above',
        'below',
        'to',
        'up',
        'down',
        'in',
        'out',
        'on',
        'off',
        'over',
        'under',
        'again',
        'further',
        'once',
        'here',
        'there',
        'where',
        'why',
        'how',
        'all',
        'any',
        'both',
        'each',
        'few',
        'more',
        'most',
        'other',
        'some',
        'such',
        'no',
        'nor',
        'not',
        'only',
        'own',
        'same',
        'so',
        'than',
        'too',
        'very',
        's',
        't',
        'can',
        'will',
        'just',
        'don',
        'should',
        'now',
    },
    'de': {
        'der',
        'die',
        'das',
        'den',
        'dem',
        'des',
        'ein',
        'eine',
        'einer',
        'eines',
        'einem',
        'einen',
        'ist',
        'sind',
        'war',
        'waren',
        'wird',
        'werden',
        'wurde',
        'wurden',
        'hat',
        'haben',
        'hatte',
        'hatten',
        'kann',
        'können',
        'konnte',
        'konnten',
        'muss',
        'müssen',
        'musste',
        'mussten',
        'soll',
        'sollen',
        'sollte',
        'sollten',
        'wollen',
        'wollte',
        'wollten',
        'darf',
        'dürfen',
        'durfte',
        'durften',
        'mag',
        'mögen',
        'mochte',
        'mochten',
        'und',
        'oder',
        'aber',
        'denn',
        'weil',
        'wenn',
        'als',
        'ob',
        'damit',
        'obwohl',
        'während',
        'nachdem',
        'bevor',
        'sobald',
        'seit',
        'bis',
        'indem',
        'ohne',
        'außer',
        'gegen',
        'für',
        'mit',
        'zu',
        'von',
        'bei',
        'nach',
        'aus',
        'über',
        'unter',
        'neben',
        'zwischen',
        'vor',
        'hinter',
        'auf',
        'um',
        'herum',
        'durch',
        'entlang',
        'ich',
        'du',
        'er',
        'sie',
        'es',
        'wir',
        'ihr',
        'mich',
        'dich',
        'ihn',
        'uns',
        'euch',
        'ihnen',
        'mein',
        'dein',
        'sein',
        'unser',
        'euer',
        'ihre',
        'ihres',
        'ihrem',
        'ihren',
    },
}


def load_stopwords(language: str = 'en') -> Set[str]:
    """Load stopwords for a specific language from configuration files.

    Args:
        language: Language code (e.g., 'en' for English, 'de' for German)

    Returns:
        Set of stopwords for the specified language

    """
    stopwords_directory = os.path.join(os.path.dirname(__file__), '..', 'resources', f'stopwords_{language}.json')

    try:
        with open(stopwords_directory, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        logger.debug(
            'Could not load stopwords for %s from %s, using defaults',
            language,
            stopwords_directory,
        )
        return _DEFAULT_STOPWORDS.get(language, set())


def get_sentence_transformer_model(model_name: str) -> Optional[Any]:
    """Retrieve or load a SentenceTransformer model.

    Args:
        model_name: Name of the model to load

    Returns:
        The loaded model or None if SentenceTransformer is unavailable

    """
    if SentenceTransformer is None:
        logger.warning('SentenceTransformer is unavailable.')
        return None

    if model_name not in _MODEL_CACHE:
        logger.info('Loading model: %s', model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    return _MODEL_CACHE[model_name]
