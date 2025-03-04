#!/usr/bin/env python
"""Run tests for the RAG system.

This script runs the tests for the RAG system, including unit tests and
integration tests.
"""

import argparse
import logging
import os
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Run tests for the RAG system.")
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test file",
    )
    return parser


def run_unit_tests():
    """Run unit tests."""
    logger.info("Running unit tests...")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/unit", "-v"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        logger.error("Unit tests failed!")
        print(result.stderr)
        return False
    logger.info("Unit tests passed!")
    return True


def run_integration_tests():
    """Run integration tests."""
    logger.info("Running integration tests...")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/integration", "-v"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        logger.error("Integration tests failed!")
        print(result.stderr)
        return False
    logger.info("Integration tests passed!")
    return True


def run_specific_test(test_file: str):
    """Run a specific test file."""
    logger.info(f"Running test file: {test_file}")

    # Check if the file exists
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False

    # Run the test
    result = subprocess.run(
        ["python", test_file],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        logger.error("Test failed!")
        print(result.stderr)
        return False
    logger.info("Test passed!")
    return True


def main():
    """Run the script."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # If no arguments are provided, show help
    if not (args.unit or args.integration or args.all or args.test):
        parser.print_help()
        return

    # Run tests based on arguments
    if args.all:
        run_unit_tests()
        run_integration_tests()
    elif args.unit:
        run_unit_tests()
    elif args.integration:
        run_integration_tests()
    elif args.test:
        run_specific_test(args.test)


if __name__ == "__main__":
    main()
