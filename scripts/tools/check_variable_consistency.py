#!/usr/bin/env python
"""Check variable naming consistency across the codebase.

This script scans Python files in the project to identify potential
inconsistencies in variable naming patterns and suggests standardized alternatives.

Usage:
    python -m scripts.tools.check_variable_consistency [--path PATH] [--fix]
"""

import argparse
import ast
import logging
import os
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class VariableCategory(Enum):
    """Categories of variables to check for consistency."""

    DOCUMENT = 'document'
    DIRECTORY = 'directory'
    VECTORSTORE = 'vectorstore'
    THRESHOLD = 'threshold'
    CONFIG = 'config'
    MODEL = 'model'
    EMBEDDING = 'embedding'


# Standard patterns for each category
STANDARD_PATTERNS = {
    VariableCategory.DOCUMENT: {
        'preferred': ['documents', 'document_chunks'],
        'alternatives': ['docs', 'doc', 'document_list', 'document_collection'],
    },
    VariableCategory.DIRECTORY: {
        'preferred': ['*_dir', '*_directory'],
        'alternatives': ['*_path', '*_folder', '*_location'],
    },
    VariableCategory.VECTORSTORE: {
        'preferred': ['vectorstore'],
        'alternatives': ['vector_store', 'vector_db', 'vs', 'vector_database'],
    },
    VariableCategory.THRESHOLD: {
        'preferred': ['*_threshold'],
        'alternatives': ['*_thres', '*_cutoff', '*_limit'],
    },
    VariableCategory.CONFIG: {
        'preferred': ['config'],
        'alternatives': ['cfg', 'conf', 'configuration', 'settings'],
    },
    VariableCategory.MODEL: {
        'preferred': ['model_name', 'model'],
        'alternatives': ['model_id', 'mdl', 'mod'],
    },
    VariableCategory.EMBEDDING: {
        'preferred': ['embedding_function', 'embeddings'],
        'alternatives': ['embed_fn', 'emb', 'embedding_fn'],
    },
}


class VariableVisitor(ast.NodeVisitor):
    """AST visitor to extract variable assignments and usages."""

    def __init__(self):
        """Initialize the visitor with empty data structures."""
        self.variable_assignments = []
        self.function_args = []
        self.class_attributes = []

    def visit_Assign(self, node):
        """Extract variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_assignments.append(target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Extract function arguments."""
        for arg in node.args.args:
            self.function_args.append(arg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Extract class attributes."""
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                self.class_attributes.append(child.target.id)
        self.generic_visit(node)


def categorize_variable(name: str) -> Optional[VariableCategory]:
    """Categorize a variable name based on patterns."""
    if any(word in name for word in ['document', 'doc', 'docs']):
        return VariableCategory.DOCUMENT
    elif any(word in name for word in ['dir', 'directory', 'path', 'folder']):
        return VariableCategory.DIRECTORY
    elif any(word in name for word in ['vectorstore', 'vector_store', 'vector']):
        return VariableCategory.VECTORSTORE
    elif 'threshold' in name or 'thres' in name or 'cutoff' in name:
        return VariableCategory.THRESHOLD
    elif any(word in name for word in ['config', 'cfg', 'conf', 'settings']):
        return VariableCategory.CONFIG
    elif 'model' in name or 'mdl' in name:
        return VariableCategory.MODEL
    elif 'embed' in name:
        return VariableCategory.EMBEDDING
    return None


def check_variable_name(name: str) -> Tuple[bool, Optional[str], Optional[VariableCategory]]:
    """Check if a variable name follows standard patterns and suggest alternatives."""
    category = categorize_variable(name)

    if not category:
        return True, None, None

    # Check against preferred patterns
    for pattern in STANDARD_PATTERNS[category]['preferred']:
        if '*' in pattern:
            # Handle wildcard patterns
            prefix, suffix = pattern.split('*')
            if (not prefix or name.startswith(prefix)) and (not suffix or name.endswith(suffix)):
                return True, None, category
        elif name == pattern:
            return True, None, category

    # If not matching preferred, suggest alternatives
    suggestion = None
    for preferred in STANDARD_PATTERNS[category]['preferred']:
        if '*' in preferred:
            # Try to adapt the preferred pattern
            prefix, suffix = preferred.split('*')

            # Extract the middle part from the variable
            middle = name
            for alt_pattern in STANDARD_PATTERNS[category]['alternatives']:
                if '*' in alt_pattern:
                    alt_prefix, alt_suffix = alt_pattern.split('*')
                    if name.startswith(alt_prefix) and name.endswith(alt_suffix):
                        middle = name[len(alt_prefix) : -len(alt_suffix) if alt_suffix else None]
                        suggestion = f'{prefix}{middle}{suffix}'
                        break
        else:
            # For exact matches, just use the preferred version
            suggestion = preferred

    return False, suggestion, category


def analyze_file(file_path: str) -> Dict[str, List[Tuple[str, str, VariableCategory]]]:
    """Analyze a Python file for variable naming inconsistencies."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.warning(f'Syntax error in {file_path}, skipping')
        return {}

    visitor = VariableVisitor()
    visitor.visit(tree)

    # Combine all variable names
    all_vars = visitor.variable_assignments + visitor.function_args + visitor.class_attributes

    # Check each variable name
    inconsistencies = defaultdict(list)
    for var_name in all_vars:
        is_consistent, suggestion, category = check_variable_name(var_name)
        if not is_consistent and suggestion:
            inconsistencies[file_path].append((var_name, suggestion, category))

    return inconsistencies


def find_python_files(base_path: str) -> List[str]:
    """Find all Python files in the given path recursively."""
    python_files = []

    # Directories and files to exclude
    exclude_dirs = [
        '__pycache__',
        '.pytest_cache',
        '.ruff_cache',
        '.mypy_cache',
        '.git',
        '.llm_rag',
        'venv',
        '.venv',
        'env',
        '.env',
    ]
    exclude_files = [
        '*.pyc',
        '*_pb2.py',  # Skip generated protobuf files
        '*.pyi',  # Skip type stub files
    ]

    for root, dirs, files in os.walk(base_path):
        # Modify dirs in-place to exclude certain directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith('.egg-info')]

        for file in files:
            if file.endswith('.py') and not any(file.endswith(ext.replace('*', '')) for ext in exclude_files):
                file_path = os.path.join(root, file)
                python_files.append(file_path)

    return python_files


def main():
    """Check variable naming consistency across the codebase."""
    parser = argparse.ArgumentParser(description='Check variable naming consistency across the codebase')
    parser.add_argument(
        '--path',
        type=str,
        default='.',
        help='Path to check for Python files (default: current directory)',
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to automatically fix inconsistencies (experimental)',
    )

    args = parser.parse_args()

    base_path = args.path
    python_files = find_python_files(base_path)

    logger.info(f'Found {len(python_files)} Python files to analyze')

    all_inconsistencies = {}
    for file_path in python_files:
        inconsistencies = analyze_file(file_path)
        if inconsistencies:
            all_inconsistencies.update(inconsistencies)

    # Group by category
    by_category = defaultdict(list)
    for file_path, issues in all_inconsistencies.items():
        for var_name, suggestion, category in issues:
            by_category[category].append((file_path, var_name, suggestion))

    # Print report
    total_issues = sum(len(issues) for issues in all_inconsistencies.values())

    if total_issues == 0:
        logger.info('No variable naming inconsistencies found!')
        return

    logger.info(f'Found {total_issues} variable naming inconsistencies:')

    for category in VariableCategory:
        if category in by_category:
            print(f'\n{category.value.upper()} VARIABLES:')
            for file_path, var_name, suggestion in by_category[category]:
                rel_path = os.path.relpath(file_path, base_path)
                print(f"  {rel_path}: '{var_name}' → '{suggestion}'")

    # Output summary and recommendations
    print('\nSUMMARY:')
    print(f'  Total issues: {total_issues}')
    print('  Categories with issues:')
    for category, issues in by_category.items():
        print(f'    - {category.value}: {len(issues)} issues')

    print('\nRECOMMENDATIONS:')
    print('  1. Update the variable names to match the preferred patterns')
    print('      Examples:')
    print("        - 'docs' → 'documents'")
    print("        - 'vector_store' → 'vectorstore'")
    print("        - 'output_path' → 'output_dir'")
    print('  2. Add this check to your pre-commit hooks')
    print('  3. For variables that should be exceptions, document them')


if __name__ == '__main__':
    main()
