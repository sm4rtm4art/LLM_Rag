# Sample Document Comparison

## Summary

Total sections: 18
Similar: 9
Minor Changes: 3
Major Changes: 1
Rewritten: 1
New: 3
Deleted: 1

## Detailed Comparison

### Introduction
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
Introduction
```

### Section 2
**Result**: MAJOR_CHANGES (Similarity: 0.50)

Content differs between documents:
```diff
- This is the first sample document for testing the document comparison module. It contains multiple sections with different types of content.
+ This is the second sample document for testing the document comparison module. It contains similar content as the first document but with some modifications.
```

### Feature Overview
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
Feature Overview
```

### Section 4
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
The document comparison module can:
```

### Section 5
**Result**: MINOR_CHANGES (Similarity: 0.76)

Content differs between documents:
```diff
- - Parse structured documents
- - Align corresponding sections
- - Compare sections using embeddings
- - Generate readable diffs
+ - Parse structured documents
+ - Align corresponding sections
+ - Compare sections using embeddings
+ - Generate readable diffs
+ - Handle different document formats
```

### Implementation Details
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
Implementation Details
```

### Section 7
**Result**: MINOR_CHANGES (Similarity: 0.83)

Content differs between documents:
```diff
- The implementation uses several components:
+ The implementation uses several key components:
```

### Section 8
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
1. Document parser for segmenting documents
2. Section aligner for matching corresponding sections
3. Comparison engine for calculating similarities
4. Diff formatter for generating reports
```

### Document Parser
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
Document Parser
```

### Section 10
**Result**: SIMILAR (Similarity: 0.94)

Content is similar in both documents:
```
The document parser converts structured documents into logical sections based on headings, paragraphs, lists, and other elements.
```

### Section Aligner
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
Section Aligner
```

### Section 12
**Result**: MINOR_CHANGES (Similarity: 0.84)

Content differs between documents:
```diff
- The section aligner matches corresponding sections between two documents using strategies like heading matching and content similarity.
+ The section aligner matches corresponding sections between two documents using strategies like heading matching and content similarity analysis.
```

### Conclusion
**Result**: SIMILAR (Similarity: 1.00)

Content is similar in both documents:
```
Conclusion
```

### Section 14
**Result**: REWRITTEN (Similarity: 0.47)

Content differs between documents:
```diff
- This demo shows how the document comparison module works in practice.
+ This modified demo shows how the document comparison module detects various types of changes.
```

### Sample Document 1
**Result**: DELETED (Similarity: 0.00)

Content present only in source document:
```diff
- Sample Document 1
```

### Sample Document 2
**Result**: NEW (Similarity: 0.00)

Content present only in target document:
```diff
+ Sample Document 2
```

### Embedding Comparison
**Result**: NEW (Similarity: 0.00)

Content present only in target document:
```diff
+ Embedding Comparison
```

### Section 18
**Result**: NEW (Similarity: 0.00)

Content present only in target document:
```diff
+ This is a new section that wasn't in the first document. It explains how the embedding comparison works to determine similarities between sections.
```
