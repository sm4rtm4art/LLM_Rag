# DIN-Formatted XML for LLM RAG Systems

This example demonstrates how to use a DIN (Deutsches Institut für Normung) formatted XML document with the XMLLoader for LLM RAG (Large Language Model Retrieval-Augmented Generation) systems.

## What is a DIN Document?

DIN documents are structured XML documents following standards set by the German Institute for Standardization. The format provides a highly structured way to represent technical standards, specifications, and guidelines with proper metadata, sections, and references.

The example DIN document (`din_document.xml`) represents a fictional technical standard for RAG systems implementation, showcasing how technical requirements, definitions, and guidelines can be structured in a way that's ideal for retrieval in RAG applications.

## Document Structure

The DIN document includes:

- **Metadata**: Document identifier, title, version, dates, committee information, and keywords
- **Sections**: Hierarchical content organized in sections and subsections
- **Definitions**: Formal definitions of key terms
- **Tables**: Structured tabular information
- **Code Examples**: Sample implementation code
- **References**: Bibliographic citations

## Using the XMLLoader with DIN Documents

The XMLLoader class can be used in several ways to extract information from DIN documents:

### Basic Loading

Load the entire document as a single document:

```python
from llm_rag.document_processing.loaders import XMLLoader

loader = XMLLoader("examples/din_document.xml")
documents = loader.load()
```

### Loading with Metadata Extraction

Extract specific metadata from the document:

```python
loader = XMLLoader(
    "examples/din_document.xml",
    metadata_tags=[
        "din:identifier",
        "din:title",
        "din:version",
        "din:language",
        "din:publicationDate"
    ]
)
documents = loader.load()
```

### Section-by-Section Loading

Split the document into sections, creating a separate document for each section:

```python
loader = XMLLoader(
    "examples/din_document.xml",
    split_by_tag="din:section",
    metadata_tags=["din:title"]
)
documents = loader.load()
```

### Extracting Definitions

Extract term definitions as separate documents:

```python
loader = XMLLoader(
    "examples/din_document.xml",
    split_by_tag="din:definition",
    content_tags=["din:term", "din:description"]
)
documents = loader.load()
```

### Extracting Tables

Extract tables as separate documents:

```python
loader = XMLLoader(
    "examples/din_document.xml",
    split_by_tag="din:table",
    metadata_tags=["din:caption"],
    content_tags=["din:row", "din:cell", "din:header"]
)
documents = loader.load()
```

## Running the Demo

A demo script is provided to show these different loading techniques:

```bash
python examples/din_loader_demo.py
```

This script demonstrates various ways to load and extract information from the DIN document.

## Why Use Structured XML for RAG?

Using structured XML documents like the DIN format offers several advantages for RAG systems:

1. **Metadata Richness**: The XML structure preserves important metadata that can be used for filtering and context.
2. **Hierarchical Organization**: The hierarchical nature of XML allows for logical chunking of content.
3. **Semantic Structure**: Tags convey semantic meaning, improving the quality of retrieved content.
4. **Precise Extraction**: Specific content (definitions, tables, code) can be extracted based on tags.
5. **Consistent Format**: Standardized format ensures consistent processing across documents.

## Customizing for Your Needs

The XMLLoader is highly configurable:

- Use `split_by_tag` to create one document per occurrence of a specific tag
- Use `content_tags` to only extract content from specific tags
- Use `metadata_tags` to include specific tag content as metadata
- Set `include_tags_in_text=True` to include the XML tags in the extracted text

These options allow you to tailor the document loading to your specific RAG application requirements.
