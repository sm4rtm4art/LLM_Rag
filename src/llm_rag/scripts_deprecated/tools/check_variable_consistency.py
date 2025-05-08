"""Tool to check variable naming consistency across the codebase."""

import argparse
import logging
from pathlib import Path

from src.llm_rag.scripts.check_variable_consistency import VariableConsistencyChecker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Check variable naming consistency across the codebase."""
    parser = argparse.ArgumentParser(description='Check variable naming consistency')
    parser.add_argument(
        '--path',
        type=str,
        default='src/llm_rag',
        help='Path to the codebase to analyze',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for variable comparison (0.0-1.0)',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for the report (default: stdout)',
    )
    args = parser.parse_args()

    code_path = Path(args.path)
    if not code_path.exists():
        logger.error(f'Path {code_path} does not exist')
        return 1

    # Initialize and run the checker
    checker = VariableConsistencyChecker()
    similar_vars = checker.find_similar_variables(args.threshold)
    report = checker.generate_report(similar_vars)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f'Report written to {args.output}')
    else:
        print(report)

    return 0


if __name__ == '__main__':
    exit(main())
