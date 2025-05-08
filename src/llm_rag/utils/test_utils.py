"""Utility functions for testing."""

import os
from pathlib import Path


def is_ci_environment() -> bool:
    """Check if the code is running in a CI environment.

    Returns:
        bool: True if running in CI, False otherwise.

    """
    # Common CI environment variables
    ci_env_vars = [
        'CI',
        'GITHUB_ACTIONS',
        'GITLAB_CI',
        'TRAVIS',
        'CIRCLECI',
        'JENKINS_URL',
        'BITBUCKET_COMMIT',
    ]

    return any(os.environ.get(var) for var in ci_env_vars)


def create_test_data_directory() -> Path:
    """Create a test data directory with sample files if it doesn't exist.

    Returns:
        Path: Path to the test data directory.

    """
    test_dir = Path('./test_data')
    test_dir.mkdir(exist_ok=True)

    # Create a sample test file if it doesn't exist
    test_file = test_dir / 'test.txt'
    if not test_file.exists():
        with open(test_file, 'w') as f:
            f.write('This is a sample test document for RAG testing.')

    return test_dir
