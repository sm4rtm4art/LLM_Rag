"""Test suite for the LLM RAG system.

Organization:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions

This test suite is organized into:

1. Unit Tests (/unit):
   - Testing individual components in isolation
   - Using mocks for external dependencies
   - Focusing on interface contracts and edge cases

2. Integration Tests (/integration):
   - Testing component interactions
   - Using real external services (ChromaDB, etc.)
   - Verifying end-to-end workflows

3. Test Categories:
   - vectorstore/: Vector storage and retrieval testing
   - embeddings/: Text embedding generation testing
   - api/: FastAPI endpoint testing

Testing Strategy:
- Unit tests for rapid feedback and interface verification
- Integration tests for real-world behavior validation
- High coverage requirements (>90%)
- Clear test documentation and organization
"""
