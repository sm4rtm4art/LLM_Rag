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
  - ✅ Create `src/llm_rag/document_processing/loaders/` directory
  - ✅ Extract file-based loaders to separate modules
  - ✅ Extract web-based loaders to separate modules
  - ✅ Extract database loaders to separate modules
- Implement a loader factory and registry:
  - ✅ Create a central registry for loaders
  - ✅ Add automatic discovery of loaders
  - ✅ Provide a factory function for loader instantiation
- Enhanced web loaders implementation:
  - ✅ Added robust HTML parsing with BeautifulSoup support
  - ✅ Implemented metadata extraction (title, description, author)
  - ✅ Added support for multiple output formats (text, HTML, markdown)
  - ✅ Included image URL extraction capabilities
  - ✅ Added comprehensive error handling and fallbacks
  - ✅ Implemented extensive test coverage
  - ✅ Maintained backward compatibility with WebPageLoader alias

## Phase 5: Anti-Hallucination Refactoring (Completed)

- Break down `anti_hallucination.py` (694 lines) into focused modules:
  - ✅ Create `src/llm_rag/rag/anti_hallucination/` directory
  - ✅ Extract entity verification to a separate module
  - ✅ Extract similarity-checking to a separate module
  - ✅ Extract post-processing to a separate module
  - ✅ Maintain backward compatibility through stub implementations
- Improve configurability and extensibility:
  - ✅ Make verification strategies pluggable
  - ✅ Enable runtime selection of strategies
  - ✅ Add configuration validation
  - ✅ Implement configurable model loading and caching
  - ✅ Create clear separation between verification components

## Phase 6: Testing and Documentation

- Enhance testing:
  - ✅ Add comprehensive tests for error handling module (`utils/errors.py`)
  - ✅ Add tests for pipeline components:
    - ✅ Context formatters (`rag/pipeline/context.py`)
    - ✅ Generators (`rag/pipeline/generation.py`)
    - ✅ Retrievers (`rag/pipeline/retrieval.py`)
  - [ ] Fix test failures due to architectural changes:
    - [ ] Update RAGPipeline constructor arguments
    - [ ] Check for context formatter max_length implementation
    - [ ] Adjust MarkdownContextFormatter tests for new metadata format
    - [ ] Update TemplatedGenerator interface tests
    - [ ] Fix create_retriever and create_generator factory methods
  - [ ] Ensure test coverage for all code paths
  - [ ] Add integration tests for end-to-end scenarios
- Improve documentation:
  - [ ] Update all docstrings with comprehensive information
  - [ ] Create high-level architecture documentation
  - [ ] Add usage examples for all components
- Code Quality Tools:
  - ✅ Implement variable naming consistency checker
  - [ ] Add pre-commit hook for variable consistency checks
  - [ ] Configure CI/CD pipeline to run consistency checks
  - [ ] Document variable naming conventions and patterns
  - [ ] Create guidelines for handling similar-meaning variables
- Fix warnings and deprecation issues:
  - ✅ Add configuration to handle SWIG-related warnings from C/C++ extensions
  - ✅ Fix deprecated import paths in document_processing module
  - [ ] Ensure proper docstrings for all classes and functions
  - [ ] Add type annotations to improve static type checking

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
