# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Fixed CVE-2023-36464 by replacing PyPDF2 with pypdf>=3.9.0, which addresses the infinite loop vulnerability in PDF parsing.
- Added vulnerability checking script (`scripts/check_vulnerabilities.sh`) that works with both safety auth and API key
- Updated vulnerability checking to use the newer `scan` command instead of the deprecated `check` command

### Changed

- Made all bash scripts executable
- Improved code quality in bash scripts with proper quoting and error handling

## [0.1.2] - 2025-03-06

### Added

- Initial refactoring of the document processing module
- Modularized document loaders for better maintainability
- Enhanced PDF processing capabilities
- Added support for multiple document formats

## [0.1.1] - 2025-02-25

### Added

- Initial project structure
- Basic RAG pipeline implementation
- Document processing capabilities
- Vector database integration
