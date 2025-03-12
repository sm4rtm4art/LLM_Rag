## XML Files

The `XMLLoader` can be used to load XML files with various options for extracting content and metadata:

```python
from llm_rag.document_processing.loaders import XMLLoader

# Basic usage - extracts all text content from the XML file
loader = XMLLoader("path/to/file.xml")
documents = loader.load()

# Extract content from specific tags only
loader = XMLLoader(
    "path/to/file.xml",
    content_tags=["title", "description"]
)
documents = loader.load()

# Extract metadata from specific tags
loader = XMLLoader(
    "path/to/file.xml",
    metadata_tags=["author", "year", "publisher"]
)
documents = loader.load()

# Split into multiple documents by tag
# This will create a separate document for each book element
loader = XMLLoader(
    "path/to/file.xml",
    split_by_tag="book"
)
documents = loader.load()

# Include tags in the extracted text
loader = XMLLoader(
    "path/to/file.xml",
    include_tags_in_text=True
)
documents = loader.load()
```

For example, with an XML file containing book information:

```xml
<library>
  <book>
    <title>The Great Gatsby</title>
    <author>F. Scott Fitzgerald</author>
    <description>A novel set in the Jazz Age...</description>
  </book>
  <book>
    <title>To Kill a Mockingbird</title>
    <author>Harper Lee</author>
    <description>The story of the Finch family...</description>
  </book>
</library>
```

You can extract different content based on your needs:

```python
# Create one document per book
loader = XMLLoader("books.xml", split_by_tag="book")
documents = loader.load()  # Returns 2 documents

# Extract specific content and metadata
loader = XMLLoader(
    "books.xml",
    content_tags=["description"],
    metadata_tags=["title", "author"]
)
documents = loader.load()  # Returns 1 document with metadata
```
