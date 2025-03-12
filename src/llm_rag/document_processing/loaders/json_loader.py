"""JSON document loader.

This module provides a loader for loading documents from JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..processors import Documents
from .base import DocumentLoader, FileLoader, registry

logger = logging.getLogger(__name__)


class JSONLoader(DocumentLoader, FileLoader):
    """Load documents from JSON files.

    This loader extracts content from JSON files, with options to extract
    specific fields or flatten the JSON structure.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        content_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        jq_filter: Optional[str] = None,
        json_lines: bool = False,
        ndjson: bool = False,
    ):
        """Initialize the JSON loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the JSON file, by default None
        content_key : Optional[str], optional
            Key to use for extracting content, by default None
            If None, the entire JSON is used as content
        metadata_keys : Optional[List[str]], optional
            Keys to use for extracting metadata, by default None
        jq_filter : Optional[str], optional
            JQ filter to apply to the JSON, by default None
        json_lines : bool, optional
            Whether the file contains JSON Lines, by default False
        ndjson : bool, optional
            Whether the file is in NDJSON format (alias for json_lines),
            by default False

        """
        self.file_path = Path(file_path) if file_path else None
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
        self.jq_filter = jq_filter
        self.json_lines = json_lines or ndjson

        # Check if jq is available if a filter is provided
        if jq_filter:
            try:
                import jq

                self._has_jq = True
            except ImportError:
                logger.warning("jq library not available. JQ filtering will be disabled.")
                self._has_jq = False
        else:
            self._has_jq = False

    def load(self) -> Documents:
        """Load documents from the JSON file specified during initialization.

        Returns
        -------
        Documents
            List of documents loaded from the JSON file.

        Raises
        ------
        ValueError
            If file path was not provided during initialization.

        """
        if not self.file_path:
            raise ValueError("No file path provided. Either initialize with a file path or use load_from_file.")

        return self.load_from_file(self.file_path)

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from a JSON file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the JSON file.

        Returns
        -------
        Documents
            List of documents loaded from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        json.JSONDecodeError
            If the file contains invalid JSON.

        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Base metadata
        base_metadata = {"source": str(file_path)}

        try:
            # Handle JSON Lines / NDJSON format
            if self.json_lines:
                return self._load_json_lines(file_path, base_metadata)

            # Handle regular JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Apply JQ filter if provided
            if self.jq_filter and self._has_jq:
                import jq

                data = jq.compile(self.jq_filter).input(data).all()

            # Process the JSON data
            return self._process_json_data(data, base_metadata)

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            raise

    def _load_json_lines(self, file_path: Path, base_metadata: Dict) -> Documents:
        """Load documents from a JSON Lines file.

        Parameters
        ----------
        file_path : Path
            Path to the JSON Lines file.
        base_metadata : Dict
            Base metadata to include in documents.

        Returns
        -------
        Documents
            List of documents loaded from the JSON Lines file.

        """
        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Apply JQ filter if provided
                    if self.jq_filter and self._has_jq:
                        import jq

                        filtered_data = jq.compile(self.jq_filter).input(data).all()
                        # If the filter returns multiple items, add each as a document
                        if isinstance(filtered_data, list) and filtered_data:
                            for item in filtered_data:
                                docs = self._process_json_data(item, {**base_metadata, "line": line_num})
                                documents.extend(docs)
                            continue
                        else:
                            data = filtered_data

                    # Process the line data
                    line_docs = self._process_json_data(data, {**base_metadata, "line": line_num})
                    documents.extend(line_docs)

                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON at line {line_num}: {e}. Skipping.")

        return documents

    def _process_json_data(self, data: Union[Dict, List], metadata: Dict) -> Documents:
        """Process JSON data into documents.

        Parameters
        ----------
        data : Union[Dict, List]
            JSON data to process.
        metadata : Dict
            Base metadata to include in documents.

        Returns
        -------
        Documents
            List of documents extracted from the JSON data.

        """
        documents = []

        # Handle list of items
        if isinstance(data, list):
            for i, item in enumerate(data):
                # Add index to metadata
                item_metadata = {**metadata, "index": i}

                # Process each item
                if isinstance(item, dict):
                    doc = self._extract_document(item, item_metadata)
                    documents.append(doc)
                else:
                    # If item is not a dict, use its string representation as content
                    documents.append({"content": str(item), "metadata": item_metadata})
            return documents

        # Handle single dict
        if isinstance(data, dict):
            documents.append(self._extract_document(data, metadata))
            return documents

        # Handle primitive value
        documents.append({"content": str(data), "metadata": metadata})

        return documents

    def _extract_document(self, data: Dict, metadata: Dict) -> Dict:
        """Extract a document from a JSON object.

        Parameters
        ----------
        data : Dict
            JSON object to extract from.
        metadata : Dict
            Base metadata to include.

        Returns
        -------
        Dict
            Document with content and metadata.

        """
        doc_metadata = metadata.copy()

        # Extract specified metadata keys
        for key in self.metadata_keys:
            if key in data:
                doc_metadata[key] = data[key]

        # Extract content
        if self.content_key and self.content_key in data:
            content = data[self.content_key]
            # Handle different content types
            if not isinstance(content, str):
                content = str(content)
        else:
            # Use entire JSON as content
            content = json.dumps(data, ensure_ascii=False)

        return {"content": content, "metadata": doc_metadata}


# Register the loader
registry.register(JSONLoader, extensions=["json", "jsonl", "ndjson"])
