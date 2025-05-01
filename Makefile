# Makefile for LLM RAG Project
# This file provides commands for easy setup, development, testing, and running of the project

SHELL := /bin/bash
PYTHON := python3
PIP := pip
VENV := .venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip
VENV_UVICORN := $(VENV_BIN)/uvicorn

SRC_DIR := src
TEST_DIR := tests
DOCKER_COMPOSE := docker-compose
DOCKER_COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE_CI_FILE := docker-compose.ci.yml
DOCKER_COMPOSE_API_FILE := docker-compose.api.yml

.PHONY: help setup-venv install dev-install update clean test test-cov lint format mypy safety docker-build docker-run docker-test api-run api-docker all

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

# Virtual environment setup
setup-venv:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Run 'source $(VENV)/bin/activate' to activate it"

# Install the package and its dependencies
install: setup-venv
	@echo "Installing dependencies..."
	@$(VENV_PIP) install -e .
	@echo "Installation complete"

# Install development dependencies
dev-install: install
	@echo "Installing development dependencies..."
	@$(VENV_PIP) install -e ".[dev]"
	@$(VENV_BIN)/pre-commit install
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
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .ruff_cache/
	@rm -rf .mypy_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
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
	@$(VENV_BIN)/ruff check $(SRC_DIR) $(TEST_DIR)

# Format code
format:
	@echo "Formatting code..."
	@$(VENV_BIN)/ruff format $(SRC_DIR) $(TEST_DIR)

# Run static type checking
mypy:
	@echo "Running mypy..."
	@$(VENV_BIN)/mypy

# Run security checks on dependencies
safety:
	@echo "Running safety checks..."
	@$(VENV_BIN)/safety check

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
	@$(VENV_UVICORN) llm_rag.api.main:app --reload

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
	@echo "Setting up OCR dependencies..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install tesseract; \
	elif [ "$(shell uname)" = "Linux" ]; then \
		sudo apt-get update && sudo apt-get install -y tesseract-ocr; \
	else \
		echo "Please install Tesseract OCR manually for your operating system"; \
	fi
	@echo "OCR dependencies setup complete"
