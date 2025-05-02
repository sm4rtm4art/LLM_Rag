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
endif

VENV_PYTHON := $(VENV_BIN)$(PATH_SEP)python
VENV_PIP := $(VENV_BIN)$(PATH_SEP)pip

# Container tools - provide defaults but allow override
DOCKER_COMPOSE ?= docker-compose
DOCKER_COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE_CI_FILE := docker-compose.ci.yml
DOCKER_COMPOSE_API_FILE := docker-compose.api.yml

# Project directories
SRC_DIR := src
TEST_DIR := tests

.PHONY: help setup-venv install dev-install update clean test test-ocr test-comparison test-cov lint format mypy safety docker-build docker-run docker-test api-run api-docker all setup-ocr-deps

help:
	@echo "Available commands:"
	@echo "  help                 - Show this help message"
	@echo "  setup-venv           - Create a new Python virtual environment"
	@echo "  install              - Install the package and its dependencies"
	@echo "  dev-install          - Install development dependencies"
	@echo "  update               - Update dependencies"
	@echo "  clean                - Remove build artifacts and cache files"
	@echo "  test                 - Run all tests"
	@echo "  test-ocr             - Run OCR tests"
	@echo "  test-comparison      - Run document comparison tests"
	@echo "  test-cov             - Run tests with coverage report"
	@echo "  lint                 - Run linter (ruff)"
	@echo "  format               - Format code (ruff format)"
	@echo "  mypy                 - Run static type checking"
	@echo "  safety               - Run security checks on dependencies"
	@echo "  docker-build         - Build the Docker image"
	@echo "  docker-run           - Run the application in Docker"
	@echo "  docker-test          - Run tests in Docker"
	@echo "  api-run              - Run the API server locally"
	@echo "  api-docker           - Run the API server in Docker"
	@echo "  all                  - Run lint, format, mypy, and test"
	@echo "  setup-ocr-deps       - Install OCR dependencies for current OS ($(DETECTED_OS))"

# Virtual environment setup
setup-venv:
	@echo "Creating virtual environment..."
ifeq ($(DETECTED_OS),Windows)
	@$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Run '$(VENV_BIN)\activate' to activate it"
else
	@$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Run 'source $(VENV_BIN)/activate' to activate it"
endif

# Install the package and its dependencies
install: setup-venv
	@echo "Installing dependencies..."
	@$(VENV_PIP) install -e .
	@echo "Installation complete"

# Install development dependencies
dev-install: install
	@echo "Installing development dependencies..."
	@$(VENV_PIP) install -e ".[dev]"
ifeq ($(DETECTED_OS),Windows)
	@$(VENV_PYTHON) -m pre_commit install
else
	@$(VENV_BIN)/pre-commit install
endif
	@echo "Development installation complete"

# Update dependencies
update: setup-venv
	@echo "Updating dependencies..."
	@$(VENV_PIP) install --upgrade pip
	@$(VENV_PIP) install --upgrade -e .
	@$(VENV_PIP) install --upgrade -e ".[dev]"
	@echo "Dependencies updated"

# Clean up build artifacts and cache files
clean:
	@echo "Cleaning up..."
	@$(RM) build
	@$(RM) dist
	@$(RM) "*.egg-info"
	@$(RM) .pytest_cache
	@$(RM) .coverage
	@$(RM) htmlcov
	@$(RM) .ruff_cache
	@$(RM) .mypy_cache
	@$(RM) $(SRC_DIR)/**/__pycache__
	@$(RM) $(TEST_DIR)/**/__pycache__
	@echo "Cleanup complete"

# Run all tests
test:
	@echo "Running tests..."
	@$(VENV_PYTHON) -m pytest $(TEST_DIR)

# Run OCR tests
test-ocr:
	@echo "Running OCR tests..."
	@$(VENV_PYTHON) -m pytest $(TEST_DIR)/document_processing/ocr/

# Run document comparison tests
test-comparison:
	@echo "Running document comparison tests..."
	@$(VENV_PYTHON) -m pytest $(TEST_DIR)/document_processing/comparison/

# Run tests with coverage report
test-cov:
	@echo "Running tests with coverage..."
	@$(VENV_PYTHON) -m pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=xml --cov-report=term $(TEST_DIR)
	@echo "Coverage report generated in htmlcov/"

# Run linter
lint:
	@echo "Running linter..."
ifeq ($(DETECTED_OS),Windows)
	@$(VENV_PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)
else
	@$(VENV_BIN)/ruff check $(SRC_DIR) $(TEST_DIR)
endif

# Format code
format:
	@echo "Formatting code..."
ifeq ($(DETECTED_OS),Windows)
	@$(VENV_PYTHON) -m ruff format $(SRC_DIR) $(TEST_DIR)
else
	@$(VENV_BIN)/ruff format $(SRC_DIR) $(TEST_DIR)
endif

# Run static type checking
mypy:
	@echo "Running mypy..."
ifeq ($(DETECTED_OS),Windows)
	@$(VENV_PYTHON) -m mypy
else
	@$(VENV_BIN)/mypy
endif

# Run security checks on dependencies
safety:
	@echo "Running safety checks..."
ifeq ($(DETECTED_OS),Windows)
	@$(VENV_PYTHON) -m safety check
else
	@$(VENV_BIN)/safety check
endif

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
ifeq ($(DETECTED_OS),Windows)
	@$(VENV_PYTHON) -m uvicorn llm_rag.api.main:app --reload
else
	@$(VENV_BIN)/uvicorn llm_rag.api.main:app --reload
endif

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
