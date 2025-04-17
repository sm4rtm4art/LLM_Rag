Objective: Continue improving the OCR pipeline and fix remaining issues in the codebase.

Context:

- The core OCR pipeline implementation is now complete with all tests passing
- Linter configuration has been updated to use 120 characters line length (ruff and mypy)
- Fixed a critical test failure in OCRPipeline.process_pdf related to output formatting
- Discovered that the LLM cleaner is translating non-English content to English rather than just fixing OCR errors
- Remaining tasks focus on code quality, optimization, and edge case handling

Tasks:

1. ✅ Fix linter errors and test failures in the OCR pipeline code

   - ✅ Resolved test failures by updating the test to specify raw output format
   - ✅ Fixed discrepancy between test expectations and actual behavior in OCRPipeline.process_pdf
     - Test expected simple "\n\n" joined text output but received markdown formatted text with page headers
     - Solution: Modified test to explicitly set output_format="raw" before calling process_pdf
   - ✅ All tests now pass with the current implementation

2. Enhance LLM cleaner to handle language preservation and translation

   - [ ] Modify LLMCleaner to detect input document language and preserve it by default
   - [ ] Add a configuration option to control translation behavior (translate_to_language parameter)
   - [ ] For German documents, integrate with appropriate German language models
   - [ ] Add comparison capability to track what was changed while maintaining the document's language
   - [ ] Update prompts to explicitly instruct LLM to preserve language unless translation is requested
   - [ ] Create examples demonstrating both language preservation and translation modes

3. Optimize OCR processing performance

   - [ ] Implement parallel processing option for handling multi-page documents
   - [ ] Add caching mechanism for processed pages to avoid redundant OCR
   - [ ] Consider adding batch processing capabilities for multiple documents

4. Enhance error handling and recovery

   - [ ] Implement more granular error reporting for specific OCR failures
   - [ ] Add retry logic for transient OCR engine errors
   - [ ] Improve logging to capture performance metrics and error patterns

5. Improve output formatting options

   - [ ] Enhance Markdown formatter with better table detection
   - [ ] Add HTML output formatter option
   - [ ] Implement configurable page header/footer templates

6. Documentation and examples

   - [ ] Create comprehensive usage examples for different OCR scenarios
   - [ ] Document performance optimization strategies
   - [ ] Add benchmark results for various document types and configurations

7. Testing improvements

   - [ ] Add more test cases with different document types and languages
   - [ ] Create integration tests with real-world PDFs in multiple languages
   - [ ] Add performance benchmarking tests

Next immediate actions:

1. Update LLMCleaner to preserve original document language by default
2. Add translation as a configurable option for the OCR pipeline
3. Create sample code showing both language preservation and translation modes
4. Document the language handling capabilities in the README

Implementation notes:

- When implementing language handling, update the LLMCleanerConfig to include a translate_to_language parameter
- Modify the prompt templates to include explicit instructions about language preservation
- Tests should verify both language preservation and optional translation functionality
- Keep maintainability and clean code principles in mind for all new features
- Tests should be explicit about expected output formats when testing process_pdf
