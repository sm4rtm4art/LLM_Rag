# 🚀 LLM-RAG Test Runner with Rich Display

This script provides a beautiful, interactive test runner for the LLM-RAG project using the [Rich](https://github.com/Textualize/rich) library to enhance the visual experience of running tests.

## Features

- ✨ **Visual Progress Display**: Shows a spinner and progress bar while tests are running
- 🎨 **Colorful Output**: Highlights test results with colors for better readability
- ⏱️ **Time Tracking**: Shows elapsed time during test execution
- 📊 **Summary Results**: Clear pass/fail summary at the end of the test run

## Usage

From the project root directory:

```bash
# Run all tests with nice display
.github/scripts/test.sh

# Run with various options
.github/scripts/test.sh -v                # Verbose mode
.github/scripts/test.sh -c                # With coverage
.github/scripts/test.sh -p tests/unit     # Run only unit tests
.github/scripts/test.sh -x                # Stop on first failure
```

## Command-line Options

| Option            | Description                                 |
| ----------------- | ------------------------------------------- |
| `-v, --verbose`   | Enable verbose output                       |
| `-x, --exitfirst` | Exit on first test failure                  |
| `-c, --coverage`  | Generate coverage report                    |
| `-p, --path PATH` | Specify path to test files (default: tests) |

You can also pass any additional pytest arguments after these options.

## Implementation Details

The test runner consists of two parts:

1. **`run_tests.py`**: Python script that uses Rich for display
2. **`test.sh`**: Simple Bash wrapper for convenience

The implementation uses subprocess management to run pytest while showing real-time progress.
