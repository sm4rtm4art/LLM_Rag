# Promptfoo Testing for LLM RAG System

This directory contains the configuration and test data for evaluating the LLM RAG system using [promptfoo](https://promptfoo.dev/), a tool for testing and evaluating LLM outputs.

## Directory Structure

```
tests/promptfoo/
├── promptfoo.yaml       # Configuration file for promptfoo
├── run_tests.sh         # Script to install promptfoo and run tests
├── README.md            # This file
└── test_data/           # Directory containing test datasets
    └── test_queries.json # Sample test queries with expected answers
```

## Getting Started

### Prerequisites

- Python 3.8+
- Access to the LLM RAG system

### Running Tests

To run the tests, simply execute the `run_tests.sh` script:

```bash
cd tests/promptfoo
./run_tests.sh
```

This script will:

1. Check if promptfoo is installed and install it if necessary
2. Run the tests defined in `promptfoo.yaml`
3. Generate results in JSON and HTML formats
4. On macOS, automatically open the HTML results in your default browser

### Test Configuration

The `promptfoo.yaml` file contains the configuration for the tests, including:

- Prompts to be tested
- Test cases with specific queries
- Assertions to validate responses
- Output formats and locations

### Test Data

The `test_data/test_queries.json` file contains sample test queries related to DIN standards, each with:

- A query string
- Expected answer
- Expected sources

## Adding New Tests

To add new tests:

1. Add new test queries to `test_data/test_queries.json`
2. Update `promptfoo.yaml` if necessary to include new test cases or assertions

## Viewing Results

After running the tests, results will be available at:

- JSON: `tests/promptfoo/results.json`
- HTML: `tests/promptfoo/results.html`

The HTML report provides a user-friendly interface to review test results, including:

- Overall pass/fail status
- Detailed breakdown of each test case
- Visualization of assertion results
- Comparison between expected and actual outputs

## Troubleshooting

If you encounter issues:

1. Ensure you're running the script from the `tests/promptfoo` directory
2. Check that the LLM RAG system is properly configured and accessible
3. Verify that the test data is correctly formatted
4. Review the console output for any error messages
