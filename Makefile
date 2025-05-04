# Makefile for LLM RAG Project
# This file provides commands for easy setup, development, testing, and running of the project

# Detect OS for platform-specific commands
ifeq ($(OS),Windows_NT)
    # Windows-specific settings
    DETECTED_OS := Windows
    PYTHON := python
    PIP := pip
    RM := powershell Remove-Item -Recurse -Force
    MKDIR := powershell New-Item -ItemType Directory -Force
    PATH_SEP := \\
    VENV := .venv
    VENV_BIN := $(VENV)\Scripts
    ACTIVATE := $(VENV_BIN)\activate
    # Use cmd /c for more reliable command execution on Windows
    CMD_PREFIX := cmd /c
else
    # Unix-like OS (Linux, macOS)
    DETECTED_OS := $(shell uname -s)
    PYTHON := python3
    PIP := pip3
    RM := rm -rf
    MKDIR := mkdir -p
    PATH_SEP := /
    VENV := .venv
    VENV_BIN := $(VENV)/bin
    ACTIVATE := $(VENV_BIN)/activate
    # No prefix needed for Unix
    CMD_PREFIX :=
endif

# Define path-independent commands with consistent usage
VENV_PYTHON := $(VENV_BIN)$(PATH_SEP)python
VENV_PIP := $(VENV_BIN)$(PATH_SEP)pip
VENV_UV := $(VENV_BIN)$(PATH_SEP)uv

# Container tools - provide defaults but allow override
DOCKER_COMPOSE ?= docker compose
DOCKER_COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE_CI_FILE := docker-compose.ci.yml
DOCKER_COMPOSE_API_FILE := docker-compose.api.yml

# Project directories
SRC_DIR := src
TEST_DIR := tests

.PHONY: help \
        # Environment management
        setup-venv install dev-install update clean \
        # Testing
        test test-ocr test-comparison test-cov test-parallel \
        # Code quality
        lint format mypy safety all \
        # Docker
        docker-build docker-run docker-test \
        # API
        api-run api-docker \
        # Document processing
        ocr-pipeline comparison-pipeline setup-ocr-deps \
        # Advanced operations
        deep-clean dependency-graph validate-deps sync-deps \
        # CI
        ci \
        # Documentation
        docs docs-serve benchmark \
        # Test Makefile functionality
        test-makefile \
        # Comprehensive test suite
        test-suite

help:
	@echo "LLM RAG Project Makefile"
	@echo "========================="
	@echo
	@echo "ENVIRONMENT MANAGEMENT:"
	@echo "  setup-venv           - Create a new Python virtual environment with UV"
	@echo "  install              - Install the package and its dependencies"
	@echo "  dev-install          - Install development dependencies"
	@echo "  update               - Update dependencies"
	@echo "  clean                - Remove build artifacts and cache files"
	@echo "  deep-clean           - Remove all generated files including virtual environments"
	@echo
	@echo "TESTING:"
	@echo "  test                 - Run all tests"
	@echo "  test-parallel        - Run tests in parallel for faster execution"
	@echo "  test-ocr             - Run OCR tests"
	@echo "  test-comparison      - Run document comparison tests"
	@echo "  test-cov             - Run tests with coverage report"
	@echo
	@echo "CODE QUALITY:"
	@echo "  lint                 - Run linter (ruff)"
	@echo "  format               - Format code (ruff format)"
	@echo "  mypy                 - Run static type checking"
	@echo "  safety               - Run security checks on dependencies"
	@echo "  all                  - Run lint, format, mypy, and test"
	@echo
	@echo "DOCKER:"
	@echo "  docker-build         - Build the Docker image"
	@echo "  docker-run           - Run the application in Docker"
	@echo "  docker-test          - Run tests in Docker"
	@echo
	@echo "API:"
	@echo "  api-run              - Run the API server locally"
	@echo "  api-docker           - Run the API server in Docker"
	@echo
	@echo "DOCUMENT PROCESSING:"
	@echo "  ocr-pipeline         - Run OCR pipeline"
	@echo "  comparison-pipeline  - Run document comparison pipeline"
	@echo "  setup-ocr-deps       - Install OCR dependencies for current OS ($(DETECTED_OS))"
	@echo
	@echo "DEPENDENCY MANAGEMENT:"
	@echo "  validate-deps        - Validate dependencies for conflicts"
	@echo "  dependency-graph     - Generate dependency graph visualization"
	@echo "  sync-deps            - Synchronize dependencies with lock file"
	@echo
	@echo "CI:"
	@echo "  ci                   - Run all CI checks"
	@echo
	@echo "DOCUMENTATION:"
	@echo "  docs                 - Generate documentation"
	@echo "  docs-serve           - Serve documentation locally"
	@echo "  benchmark            - Run performance benchmarks"
	@echo
	@echo "DEBUGGING:"
	@echo "  test-makefile        - Test Makefile functionality and verify configuration"
	@echo
	@echo "COMPREHENSIVE TEST SUITE:"
	@echo "  test-suite           - Run a sequence of real tests to validate core functionality"

# Virtual environment setup
setup-venv:
	@echo "Creating virtual environment with UV..."
	@uv venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
ifeq ($(DETECTED_OS),Windows)
	@echo "Run '$(VENV_BIN)\activate' to activate it"
else
	@echo "Run 'source $(VENV_BIN)/activate' to activate it"
endif

# Install the package and its dependencies
install: setup-venv
	@echo "Installing dependencies using UV..."
	@$(VENV_UV) pip install -e .
	@echo "Installation complete"

# Install development dependencies
dev-install: install
	@echo "Installing development dependencies with UV..."
	@$(VENV_UV) pip install -e ".[dev]"
	@$(VENV_UV) pip install pre-commit
ifeq ($(DETECTED_OS),Windows)
	@$(VENV_PYTHON) -m pre_commit install
else
	@$(VENV_BIN)/pre-commit install
endif
	@echo "Development installation complete"

# Update dependencies
update: setup-venv
	@echo "Updating dependencies using UV..."
	@$(VENV_UV) pip install --upgrade -e .
	@$(VENV_UV) pip install --upgrade -e ".[dev]"
	@echo "Dependencies updated"

# Enhanced cleanup
clean:
	@echo "Cleaning up build artifacts and cache files..."
	@$(VENV_PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@$(RM) build dist "*.egg-info" .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache coverage.xml .coverage.*
	@echo "Cleanup complete"

# Run all tests
test:
	@echo "Running tests..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m pytest $(TEST_DIR)"

# Run OCR tests
test-ocr:
	@echo "Running OCR tests..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m pytest $(TEST_DIR)/document_processing/ocr/"

# Run document comparison tests
test-comparison:
	@echo "Running document comparison tests..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m pytest $(TEST_DIR)/document_processing/comparison/"

# Run tests with coverage report
test-cov:
	@echo "Running tests with coverage..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=xml --cov-report=term $(TEST_DIR)"
	@echo "Coverage report generated in htmlcov/"

# Run linter
lint:
	@echo "Running linter..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)"

# Format code
format:
	@echo "Formatting code..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m ruff format $(SRC_DIR) $(TEST_DIR)"

# Run static type checking
mypy:
	@echo "Running mypy..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m mypy"

# Run security checks on dependencies
safety:
	@echo "Running safety checks..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m safety check"

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	@$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) build

# Run the application in Docker
docker-run:
	@echo "Running application in Docker..."
	@$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) up

# Run tests in Docker
docker-test:
	@echo "Running tests in Docker..."
	@$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_CI_FILE) up --exit-code-from test

# Run the API server locally
api-run:
	@echo "Running API server locally..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m uvicorn llm_rag.api.main:app --reload"

# Run the API server in Docker
api-docker:
	@echo "Running API server in Docker..."
	@$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_API_FILE) up

# Run all quality checks and tests
all: lint format mypy test

# OCR Pipeline specific targets
ocr-pipeline:
	@echo "Running OCR pipeline..."
	@$(VENV_PYTHON) -m llm_rag.document_processing.ocr.pipeline

# Document Comparison specific targets
comparison-pipeline:
	@echo "Running document comparison pipeline..."
	@$(VENV_PYTHON) -m llm_rag.document_processing.comparison.pipeline

# Setup OCR dependencies (Tesseract, etc.)
setup-ocr-deps:
	@echo "Setting up OCR dependencies for $(DETECTED_OS)..."
ifeq ($(DETECTED_OS),Darwin)
	@echo "Installing Tesseract for macOS..."
	@brew install tesseract
	@echo "OCR dependencies setup complete for macOS"
else ifeq ($(DETECTED_OS),Linux)
	@echo "Installing Tesseract for Linux..."
	@sudo apt-get update && sudo apt-get install -y tesseract-ocr
	@echo "OCR dependencies setup complete for Linux"
else ifeq ($(DETECTED_OS),Windows)
	@echo "For Windows, please install Tesseract OCR manually:"
	@echo "1. Download from https://github.com/UB-Mannheim/tesseract/wiki"
	@echo "2. Add the installation directory to your PATH environment variable"
else
	@echo "Unsupported OS for automatic OCR dependency installation."
	@echo "Please install Tesseract OCR manually for your operating system."
endif

# Run tests in parallel
test-parallel:
	@echo "Running tests in parallel for faster execution..."
	@$(CMD_PREFIX) "$(VENV_PYTHON) -m pytest -xvs $(TEST_DIR) -n auto"

# Deep clean everything including virtual environment
deep-clean: clean
	@echo "Deep cleaning (including virtual environment)..."
	@$(RM) $(VENV)
	@$(RM) .eggs
	@$(RM) *.pyc
	@$(RM) **/*.pyc
	@echo "Deep cleanup complete"

# Dependency validation and visualization
validate-deps:
	@echo "Validating dependencies for conflicts..."
	@$(VENV_UV) pip check
	@echo "Dependencies validated"

# Generate dependency graph
dependency-graph:
	@echo "Generating dependency graph..."
	@$(VENV_PYTHON) -m pipdeptree --graph-output png > dependency-graph.png
	@echo "Dependency graph generated at dependency-graph.png"

# Sync dependencies with lockfile
sync-deps:
	@echo "Syncing dependencies with lockfile..."
	@$(VENV_UV) pip sync
	@echo "Dependencies synced"

# CI task that runs all checks
ci: lint format mypy test-parallel safety
	@echo "All CI checks passed successfully"

# Generate documentation using Sphinx
docs:
	@echo "Generating documentation..."
	@$(VENV_PYTHON) -m sphinx-build -b html docs/source docs/build/html
	@echo "Documentation generated in docs/build/html"

# Serve documentation locally
docs-serve: docs
	@echo "Serving documentation on http://localhost:8000..."
	@$(VENV_PYTHON) -m http.server 8000 --directory docs/build/html

# Benchmark performance
benchmark:
	@echo "Running performance benchmarks..."
	@$(VENV_PYTHON) -m pytest $(TEST_DIR)/benchmarks --benchmark-enable
	@echo "Benchmark complete"

# Test Makefile functionality
test-makefile:
	@echo "Testing Makefile functionality..."
	@echo "1. Checking OS detection..."
	@echo "   Detected OS: $(DETECTED_OS)"
	@echo "   VENV_PYTHON: $(VENV_PYTHON)"
	@echo "   VENV_UV: $(VENV_UV)"
	@echo "   PATH_SEP: $(PATH_SEP)"
	@echo "2. Testing directory commands..."
	@$(MKDIR) test-dir
	@echo "   Created test directory" && [ -d test-dir ] && echo "   ✓ Directory creation works"
	@$(RM) test-dir
	@echo "   Removed test directory" && [ ! -d test-dir ] && echo "   ✓ Directory removal works"
	@echo "3. Testing command prefix..."
	@$(CMD_PREFIX) echo "   ✓ Command prefix works"
	@echo "4. Testing variable expansion..."
	@[ -n "$(VENV)" ] && echo "   ✓ VENV variable expanded correctly"
	@echo "5. Testing conditional logic..."
ifeq ($(DETECTED_OS),Windows)
	@echo "   ✓ Windows-specific code executed"
else
	@echo "   ✓ Unix-specific code executed"
endif
	@echo "Makefile functionality test completed successfully"

# Run a sequence of real tests to validate core functionality
test-suite: test-makefile
	@echo "Running comprehensive Makefile test suite..."
	@echo "This will create temporary files and may take a few minutes."
	@read -p "Press Enter to continue or Ctrl+C to cancel..." x || true

	@echo "1. Testing setup-venv (creating a temporary venv)..."
	@$(eval TEST_VENV := .test-venv)
	@$(eval ORIGINAL_VENV := $(VENV))
	@$(eval VENV := $(TEST_VENV))
	@$(MAKE) setup-venv
	@[ -d "$(TEST_VENV)" ] && echo "   ✓ Virtual environment created successfully"

	@echo "2. Testing clean and deep-clean..."
	@touch test-artifact.pyc
	@mkdir -p test-cache
	@$(MAKE) clean
	@[ ! -f test-artifact.pyc ] && echo "   ✓ Clean command works"
	@$(MAKE) deep-clean
	@[ ! -d "$(TEST_VENV)" ] && echo "   ✓ Deep clean removed test venv"

	@echo "Restoring original configuration..."
	@$(eval VENV := $(ORIGINAL_VENV))
	@echo "Test suite completed successfully"
