#!/usr/bin/env python3
"""Demo script for the document comparison module."""

import os
import sys
import traceback
from pathlib import Path

# Handle imports properly to avoid E402 issues
# Add project root to path before imports if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    os.environ['PYTHONPATH'] = f'{project_root}:{os.environ.get("PYTHONPATH", "")}'

# Import after path setup
# fmt: off
# ruff: noqa: E402
from src.llm_rag.document_processing.comparison.alignment import AlignmentConfig, AlignmentMethod, SectionAligner
from src.llm_rag.document_processing.comparison.comparison_engine import (
    ComparisonConfig,
    ComparisonResult,
    EmbeddingComparisonEngine,
)
from src.llm_rag.document_processing.comparison.document_parser import DocumentParser

# fmt: on


def format_simple_report(section_comparisons):
    """Format a simple text report of the comparison."""
    lines = []
    lines.append('# Sample Document Comparison')
    lines.append('')
    lines.append('## Summary')
    lines.append('')

    # Count by result type
    result_counts = {'SIMILAR': 0, 'MINOR_CHANGES': 0, 'MAJOR_CHANGES': 0, 'REWRITTEN': 0, 'NEW': 0, 'DELETED': 0}

    for comp in section_comparisons:
        result_name = comp.result.name
        if result_name in result_counts:
            result_counts[result_name] += 1

    lines.append(f'Total sections: {len(section_comparisons)}')
    for result_name, count in result_counts.items():
        lines.append(f'{result_name.replace("_", " ").title()}: {count}')

    lines.append('')
    lines.append('## Detailed Comparison')
    lines.append('')

    for i, comparison in enumerate(section_comparisons):
        section_header = f'### Section {i + 1}'

        # Try to use heading text if available
        if (
            comparison.alignment_pair.source_section
            and comparison.alignment_pair.source_section.section_type.value == 'heading'
        ):
            heading_text = comparison.alignment_pair.source_section.content
            section_header = f'### {heading_text}'
        elif (
            comparison.alignment_pair.target_section
            and comparison.alignment_pair.target_section.section_type.value == 'heading'
        ):
            heading_text = comparison.alignment_pair.target_section.content
            section_header = f'### {heading_text}'

        lines.append(section_header)
        lines.append(f'**Result**: {comparison.result.name} (Similarity: {comparison.similarity_score:.2f})')
        lines.append('')

        # Format content based on result type
        if comparison.result == ComparisonResult.SIMILAR:
            if comparison.alignment_pair.source_section:
                lines.append('Content is similar in both documents:')
                lines.append('```')
                lines.append(comparison.alignment_pair.source_section.content)
                lines.append('```')
        elif comparison.result == ComparisonResult.DELETED:
            if comparison.alignment_pair.source_section:
                lines.append('Content present only in source document:')
                lines.append('```diff')
                lines.append('- ' + comparison.alignment_pair.source_section.content.replace('\n', '\n- '))
                lines.append('```')
        elif comparison.result == ComparisonResult.NEW:
            if comparison.alignment_pair.target_section:
                lines.append('Content present only in target document:')
                lines.append('```diff')
                lines.append('+ ' + comparison.alignment_pair.target_section.content.replace('\n', '\n+ '))
                lines.append('```')
        else:
            # For changes and rewrites
            if comparison.alignment_pair.source_section and comparison.alignment_pair.target_section:
                lines.append('Content differs between documents:')
                lines.append('```diff')
                lines.append('- ' + comparison.alignment_pair.source_section.content.replace('\n', '\n- '))
                lines.append('+ ' + comparison.alignment_pair.target_section.content.replace('\n', '\n+ '))
                lines.append('```')

        lines.append('')

    return '\n'.join(lines)


def main():
    """Run a demo comparison between two markdown documents."""
    print('Document Comparison Module Demo')
    print('===============================')

    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent

    # Define paths to the sample documents
    doc1_path = script_dir / 'demo_doc1.md'
    doc2_path = script_dir / 'demo_doc2.md'

    # Ensure the documents exist
    for path in [doc1_path, doc2_path]:
        if not path.exists():
            print(f'Error: Document not found at {path}')
            return

    print(f'Comparing:\n  - {doc1_path}\n  - {doc2_path}\n')

    try:
        # 1. Read the document contents
        with open(doc1_path, 'r') as f:
            doc1_content = f.read()

        with open(doc2_path, 'r') as f:
            doc2_content = f.read()

        # 2. Parse the documents into sections
        print('Parsing documents...')
        document_parser = DocumentParser()
        doc1_sections = document_parser._parse_markdown(doc1_content)
        doc2_sections = document_parser._parse_markdown(doc2_content)

        print(f'Document 1: {len(doc1_sections)} sections')
        print(f'Document 2: {len(doc2_sections)} sections')

        # 3. Align corresponding sections
        print('Aligning sections...')
        alignment_config = AlignmentConfig(method=AlignmentMethod.HYBRID, similarity_threshold=0.7)
        aligner = SectionAligner(config=alignment_config)
        aligned_pairs = aligner.align_sections(doc1_sections, doc2_sections)

        print(f'Created {len(aligned_pairs)} section alignments')

        # 4. Compare aligned sections
        print('Comparing sections...')
        comparison_config = ComparisonConfig(
            similar_threshold=0.9, minor_change_threshold=0.7, major_change_threshold=0.5, rewritten_threshold=0.3
        )
        comparison_engine = EmbeddingComparisonEngine(config=comparison_config)
        section_comparisons = comparison_engine.compare_sections(aligned_pairs)

        print(f'Completed {len(section_comparisons)} section comparisons')

        # 5. Format the comparison results using our custom formatter
        print('Formatting results...')
        result = format_simple_report(section_comparisons)

        # Save the result to a file
        output_path = script_dir / 'comparison_result.md'
        with open(output_path, 'w') as f:
            f.write(result)

        print(f'\nComparison complete! Results saved to {output_path}')

        # Compute and print statistics
        comparison_stats = {result_type.name: 0 for result_type in ComparisonResult}
        for comparison in section_comparisons:
            comparison_stats[comparison.result.name] += 1

        print('\nComparison Statistics:')
        print(f'  Total sections: {len(section_comparisons)}')
        print(f'  Similar sections: {comparison_stats["SIMILAR"]}')
        print(f'  Minor changes: {comparison_stats["MINOR_CHANGES"]}')
        print(f'  Major changes: {comparison_stats["MAJOR_CHANGES"]}')
        print(f'  Rewritten sections: {comparison_stats["REWRITTEN"]}')
        print(f'  New sections: {comparison_stats["NEW"]}')
        print(f'  Deleted sections: {comparison_stats["DELETED"]}')

        print('\nOpen comparison_result.md to see the full comparison report.')

    except Exception as e:
        print(f'Error during comparison: {str(e)}')
        traceback.print_exc()


if __name__ == '__main__':
    main()
