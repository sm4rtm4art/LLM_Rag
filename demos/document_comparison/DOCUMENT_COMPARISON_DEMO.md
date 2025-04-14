# Document Comparison Demo

This demo showcases the document comparison functionality implemented in the `llm_rag` package. It demonstrates how the system can:

1. Parse structured documents into logical sections
2. Align corresponding sections between documents
3. Compare aligned sections using embedding-based similarity
4. Generate human-readable differences in Markdown format

## Files in this Demo

- `demo_doc1.md` - First sample document
- `demo_doc2.md` - Second sample document with various modifications
- `demo_comparison.py` - Python script that runs the comparison
- `comparison_result.md` - Generated output showing the differences between documents

## Running the Demo

To run the demo, navigate to the repository root and execute:

```bash
python demos/document_comparison/demo_comparison.py
```

This will compare the two sample documents and generate a formatted diff report in `demos/document_comparison/comparison_result.md`.

## Understanding the Output

The comparison report includes:

- A summary section showing counts of different change types
- A detailed comparison showing each section with its classification:
  - SIMILAR - Sections with high semantic similarity
  - MINOR_CHANGES - Sections with slight modifications
  - MAJOR_CHANGES - Sections with significant content differences
  - REWRITTEN - Sections that convey similar meaning with different wording
  - NEW - Sections present only in the second document
  - DELETED - Sections present only in the first document

## Future Enhancements

As outlined in the EXPANSION_PLAN.md, future versions will incorporate LLM-based semantic comparison to further improve accuracy, particularly for detecting rewrites and nuanced meaning changes.
