# LLM-RAG Refactoring Plan

This document outlines the incremental refactoring plan for the LLM-RAG codebase, focusing on maintainability, SOLID principles, and clean code practices while ensuring that the original functionality is preserved at each step.

## Guiding Principles

1. **Keep Original Files Intact**: Maintain the original files as long as possible to preserve backward compatibility
2. **Incremental Changes**: Make small, testable changes rather than large-scale rewrites
3. **SOLID Principles**: Apply Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion
4. **Clean Code Practices**: Ensure code is readable, well-documented, and properly tested
5. **Design Patterns**: Use appropriate patterns where they add value without over-engineering

## Phase 1: Centralized Cross-Cutting Concerns (Completed)

- ✅ Created centralized logging system (`src/llm_rag/utils/logging.py`)
- ✅ Implemented standardized error handling (`src/llm_rag/utils/errors.py`)
- ✅ Added configuration management utilities (`src/llm_rag/utils/config.py`)
- ✅ Updated utils `__init__.py` to re-export all components

## Phase 2: Modular Structure with Backward Compatibility (In Progress)

- ✅ Created a modular structure for pipeline components while keeping original files
- ✅ Extracted core classes to separate modules:
  - `src/llm_rag/rag/pipeline/base.py`: Core RAGPipeline and base classes
  - `src/llm_rag/rag/pipeline/conversational.py`: Conversational pipeline
  - `src/llm_rag/rag/pipeline/document_processor.py`: Document processing utilities
- ✅ Modified original `pipeline.py` to re-export components from new structure
- 🔄 Update imports across the codebase incrementally

## Phase 3: Further Modularization and Enhancement

- Break down large files with Single Responsibility in mind:
  - [ ] Extract retrieval functionality to `src/llm_rag/rag/pipeline/retrieval.py`
  - [ ] Extract context formatting to `src/llm_rag/rag/pipeline/context.py`
  - [ ] Extract generation to `src/llm_rag/rag/pipeline/generation.py`
- Apply design patterns where appropriate:
  - [ ] Strategy pattern for different retrieval mechanisms
  - [ ] Factory pattern for creating pipeline components
  - [ ] Builder pattern for constructing complex pipelines
- Enhance error handling throughout:
  - [ ] Use specific error types consistently
  - [ ] Add comprehensive error recovery strategies
  - [ ] Improve error messages and logging

## Phase 4: Loaders Refactoring (Completed)

- Break down `loaders.py` (1013 lines) into focused modules:
  - [x] Create `src/llm_rag/document_processing/loaders/` directory
  - [x] Extract file-based loaders to separate modules
  - [x] Extract web-based loaders to separate modules
  - [x] Extract database loaders to separate modules
- Implement a loader factory and registry:
  - [x] Create a central registry for loaders
  - [x] Add automatic discovery of loaders
  - [x] Provide a factory function for loader instantiation
- Enhanced web loaders implementation:
  - [x] Added robust HTML parsing with BeautifulSoup support
  - [x] Implemented metadata extraction (title, description, author)
  - [x] Added support for multiple output formats (text, HTML, markdown)
  - [x] Included image URL extraction capabilities
  - [x] Added comprehensive error handling and fallbacks
  - [x] Implemented extensive test coverage
  - [x] Maintained backward compatibility with WebPageLoader alias

## Phase 5: Anti-Hallucination Refactoring (Completed)

- Break down `anti_hallucination.py` (694 lines) into focused modules:
  - [x] Create `src/llm_rag/rag/anti_hallucination/` directory
  - [x] Extract entity verification to a separate module
  - [x] Extract similarity-checking to a separate module
  - [x] Extract post-processing to a separate module
  - [x] Maintain backward compatibility through stub implementations
- Improve configurability and extensibility:
  - [x] Make verification strategies pluggable
  - [x] Enable runtime selection of strategies
  - [x] Add configuration validation
  - [x] Implement configurable model loading and caching
  - [x] Create clear separation between verification components

## Phase 6: Testing and Documentation

- Enhance testing:
  - [ ] Add unit tests for all new modules
  - [ ] Ensure test coverage for all code paths
  - [ ] Add integration tests for end-to-end scenarios
- Improve documentation:
  - [ ] Update all docstrings with comprehensive information
  - [ ] Create high-level architecture documentation
  - [ ] Add usage examples for all components
- Code Quality Tools:
  - [x] Implement variable naming consistency checker
  - [ ] Add pre-commit hook for variable consistency checks
  - [ ] Configure CI/CD pipeline to run consistency checks
  - [ ] Document variable naming conventions and patterns
  - [ ] Create guidelines for handling similar-meaning variables

## Phase 7: Deprecation Strategy

- Only after thorough testing and validation:
  - [ ] Mark original monolithic implementations as deprecated
  - [ ] Add warnings for direct usage of deprecated components
  - [ ] Update all examples to use new structure
  - [ ] Provide migration guidance for users

## Implementation Approach

Throughout this refactoring, we will:

1. **Keep Tests Running**: Ensure all tests pass after each change
2. **Maintain Backward Compatibility**: Keep the original interfaces working
3. **Apply Changes Incrementally**: Make small, focused changes rather than big rewrites
4. **Follow SOLID Principles**: Ensure each module has a single responsibility
5. **Document Changes**: Keep documentation in sync with code changes

The goal is to improve the codebase's maintainability and extensibility while ensuring it remains fully functional throughout the process.
