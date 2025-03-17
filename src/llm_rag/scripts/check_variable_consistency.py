"""Tool to check variable naming consistency across the codebase.

This script analyzes Python files to identify variables that might have similar meanings but different names,
helping maintain consistent naming conventions.
"""

import argparse
import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VariableConsistencyChecker:
    """Analyzes Python files for variable naming consistency."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the checker with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model to use

        """
        self.model = SentenceTransformer(model_name)
        self.variable_groups: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for variable names.

        Args:
            file_path: Path to the Python file to analyze

        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    # Get context from surrounding code
                    start = max(0, node.lineno - 2)
                    end = min(len(content.splitlines()), node.lineno + 2)
                    context = "\n".join(content.splitlines()[start:end])

                    self.variable_groups[node.id].append((str(file_path), str(node.lineno), context))
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")

    def find_similar_variables(self, similarity_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find variables with similar meanings but different names.

        Args:
            similarity_threshold: Minimum similarity score to consider variables similar

        Returns:
            List of tuples containing (var1, var2, similarity_score)

        """
        similar_vars = []
        var_names = list(self.variable_groups.keys())

        if not var_names:
            return similar_vars

        # Get embeddings for all variable names
        embeddings = self.model.encode(var_names)

        # Compare each pair of variables
        for i in range(len(var_names)):
            for j in range(i + 1, len(var_names)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )

                if similarity > similarity_threshold:
                    similar_vars.append((var_names[i], var_names[j], similarity))

        return sorted(similar_vars, key=lambda x: x[2], reverse=True)

    def generate_report(self, similar_vars: List[Tuple[str, str, float]]) -> str:
        """Generate a report of variable naming inconsistencies.

        Args:
            similar_vars: List of similar variable pairs

        Returns:
            Formatted report string

        """
        if not similar_vars:
            return "No significant variable naming inconsistencies found."

        report = ["Variable Naming Consistency Report", "=" * 30, ""]

        for var1, var2, similarity in similar_vars:
            report.append(f"Similar Variables (similarity: {similarity:.2f}):")
            report.append(f"  - {var1}")
            report.append(f"  - {var2}")

            # Show usage locations
            report.append("\n  Usage locations:")
            for file_path, line_no, context in self.variable_groups[var1]:
                report.append(f"    {file_path}:{line_no}")
                report.append(f"    Context:\n{context}")
            for file_path, line_no, context in self.variable_groups[var2]:
                report.append(f"    {file_path}:{line_no}")
                report.append(f"    Context:\n{context}")
            report.append("")

        return "\n".join(report)


def main():
    """Check variable naming consistency across the codebase."""
    parser = argparse.ArgumentParser(description="Check variable naming consistency")
    parser.add_argument(
        "--path",
        type=str,
        default="src/llm_rag",
        help="Path to the codebase to analyze",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for variable comparison (0.0-1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for the report (default: stdout)",
    )
    args = parser.parse_args()

    checker = VariableConsistencyChecker()
    code_path = Path(args.path)

    if not code_path.exists():
        logger.error(f"Path {code_path} does not exist")
        return 1

    # Analyze all Python files
    for file_path in code_path.rglob("*.py"):
        logger.info(f"Analyzing {file_path}")
        checker.analyze_file(file_path)

    # Find similar variables
    similar_vars = checker.find_similar_variables(args.threshold)

    # Generate and output report
    report = checker.generate_report(similar_vars)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    exit(main())
