# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Bash script testing framework with shellcheck integration
- Usage documentation for bash scripts
- CI/CD workflow job for bash script linting and testing
- Dependency on bash-lint job for the build job

### Fixed

- Shellcheck warnings in bash scripts
- Made all bash scripts executable
- Improved code quality in bash scripts with proper quoting and error handling

## [0.1.2] - 2025-03-06

### Added

- Public test data and RAG test scripts for safe GitHub sharing
- Improved test utilities and mocking for CI environment
- Kubernetes test workflow with disk space cleanup
- Conditional deployment based on configuration existence
- UV package manager integration replacing pip

### Changed

- Moved demo scripts to dedicated demos directory
- Improved repository structure and organization
- Updated dependencies in pyproject.toml
- Removed requirements.txt in favor of pyproject.toml
- Enhanced code quality with stricter linting rules

### Fixed

- GitHub Actions workflows for better disk space management
- Integration tests for compatibility with public data
- Linting issues across the codebase
- Test failures in CI environment

## [0.1.1] - 2025-03-03

### Added

- Initial project setup
- RAG pipeline implementation
- Vector store integration with ChromaDB
- Embedding models integration
- Type checking with mypy
- Linting with ruff

### Fixed

- Type annotations
- Docker build issues

## [0.1.0] - 2025-03-03

### Added

- Initial release
