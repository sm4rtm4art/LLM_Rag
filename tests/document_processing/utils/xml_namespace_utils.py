#!/usr/bin/env python
"""Utility functions for working with namespaced XML documents.

These utilities help with working with DIN documents and other namespaced XML
using the XMLLoader class.

These utilities help with:
1. Extracting namespaces from XML documents
2. Mapping between namespace URIs and prefixes
3. Simplifying XPath queries with namespaces
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union

from llm_rag.document_processing.loaders import XMLLoader

# Update path to the DIN document
DIN_DOCUMENT_PATH = "tests/test_data/xml/din_document.xml"


def extract_namespaces(xml_file: Union[str, Path]) -> Dict[str, str]:
    """Extract all namespaces (prefix to URI mappings) from an XML file.

    Parameters
    ----------
    xml_file : Union[str, Path]
        Path to the XML file

    Returns
    -------
    Dict[str, str]
        Dictionary mapping namespace prefixes to URIs

    """
    xml_file = Path(xml_file)

    # Simple regex approach to extract namespace declarations
    with open(xml_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all xmlns:prefix="uri" patterns
    ns_pattern = r'xmlns:([a-zA-Z0-9_]+)="([^"]+)"'
    namespaces = {prefix: uri for prefix, uri in re.findall(ns_pattern, content)}

    # Check for default namespace (xmlns="uri")
    default_ns = re.search(r'xmlns="([^"]+)"', content)
    if default_ns:
        namespaces[""] = default_ns.group(1)

    return namespaces


def create_namespace_mapping(namespaces: Dict[str, str]) -> Dict[str, str]:
    """Create a mapping from namespace URIs to prefixes for ElementTree.

    Parameters
    ----------
    namespaces : Dict[str, str]
        Dictionary mapping namespace prefixes to URIs

    Returns
    -------
    Dict[str, str]
        Dictionary mapping namespace URIs to prefixes

    """
    return {uri: prefix for prefix, uri in namespaces.items()}


def find_elements_by_tag(
    xml_file: Union[str, Path], tag_name: str, namespaces: Optional[Dict[str, str]] = None
) -> List[ET.Element]:
    """Find all elements with a given tag name in an XML file.

    Parameters
    ----------
    xml_file : Union[str, Path]
        Path to the XML file
    tag_name : str
        Tag name to search for (can include namespace prefix)
    namespaces : Optional[Dict[str, str]], optional
        Dictionary mapping namespace prefixes to URIs, by default None

    Returns
    -------
    List[ET.Element]
        List of matching elements

    """
    xml_file = Path(xml_file)

    # Extract namespaces if not provided
    if namespaces is None:
        namespaces = extract_namespaces(xml_file)

    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Handle namespaced tag
    if ":" in tag_name:
        prefix, local_name = tag_name.split(":", 1)
        if prefix in namespaces:
            # Search using Clark notation {namespace}localname
            namespace = namespaces[prefix]
            elements = root.findall(f".//{{{namespace}}}{local_name}")
            return elements

    # Fallback to direct search
    elements = root.findall(f".//{tag_name}")
    return elements


def load_namespaced_xml(
    xml_file: Union[str, Path],
    split_by_tag: Optional[str] = None,
    content_tags: Optional[List[str]] = None,
    metadata_tags: Optional[List[str]] = None,
) -> List[dict]:
    """Load a namespaced XML document using XMLLoader.

    This is a convenience function for working with namespaced XML documents
    using the XMLLoader class.

    Parameters
    ----------
    xml_file : Union[str, Path]
        Path to the XML file
    split_by_tag : Optional[str], optional
        Tag to split the document by, by default None
    content_tags : Optional[List[str]], optional
        Tags to include as content, by default None
    metadata_tags : Optional[List[str]], optional
        Tags to include as metadata, by default None

    Returns
    -------
    List[dict]
        List of documents extracted from the XML file

    """
    loader = XMLLoader(
        file_path=xml_file, split_by_tag=split_by_tag, content_tags=content_tags, metadata_tags=metadata_tags
    )

    return loader.load()


def extract_namespaced_definitions(
    xml_file: Union[str, Path],
    definition_tag: str = "din:definition",
    term_tag: str = "din:term",
    description_tag: str = "din:description",
) -> Dict[str, str]:
    """Extract term definitions from a namespaced XML document.

    Parameters
    ----------
    xml_file : Union[str, Path]
        Path to the XML file
    definition_tag : str, optional
        Tag containing the definition, by default "din:definition"
    term_tag : str, optional
        Tag containing the term, by default "din:term"
    description_tag : str, optional
        Tag containing the description, by default "din:description"

    Returns
    -------
    Dict[str, str]
        Dictionary mapping terms to their descriptions

    """
    loader = XMLLoader(file_path=xml_file, split_by_tag=definition_tag, content_tags=[term_tag, description_tag])

    documents = loader.load()
    definitions = {}

    for doc in documents:
        content = doc["content"]

        # Extract term and description from the content
        term = None
        description = None

        for line in content.split("\n"):
            if line.startswith(f"{term_tag}:"):
                term = line[len(f"{term_tag}:") :].strip()
            elif line.startswith(f"{description_tag}:"):
                description = line[len(f"{description_tag}:") :].strip()

        if term and description:
            definitions[term] = description

    return definitions


if __name__ == "__main__":
    # Example usage
    xml_file = DIN_DOCUMENT_PATH

    # Make sure the file exists
    if not Path(xml_file).exists():
        print(f"Error: File not found: {xml_file}")
        print("Please run this script from the project root directory")
        exit(1)

    # Example: Extract namespaces
    namespaces = extract_namespaces(xml_file)
    print("\nNamespaces in the document:")
    for prefix, uri in namespaces.items():
        print(f"  {prefix or '(default)'}: {uri}")

    # Example: Extract definitions
    definitions = extract_namespaced_definitions(xml_file)
    print("\nDefinitions in the document:")
    for term, desc in definitions.items():
        if len(desc) > 100:
            print(f"  {term}: {desc[:100]}...")
        else:
            print(f"  {term}: {desc}")

    # Example: Find all section titles
    loader = XMLLoader(file_path=xml_file, split_by_tag="din:section", metadata_tags=["din:title"])

    sections = loader.load()
    print("\nSections in the document:")
    for i, section in enumerate(sections, 1):
        title = section["metadata"].get("din:title", f"Section {i}")
        print(f"  {i}. {title}")
