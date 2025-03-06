#!/usr/bin/env python3
"""
Test script for structural image extraction from PDFs.

This script demonstrates the new structural image extraction capability
using PyMuPDF to identify actual images in PDF documents rather than
just text references to images.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test structural image extraction from PDFs"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to a PDF file or directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (defaults to current directory)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save extracted images to disk"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare structural extraction with text-based extraction"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def extract_images_from_pdf(
    pdf_path, output_dir=None, save_images=False, compare=False, verbose=False
):
    """Extract images from a PDF file using structural extraction.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        save_images: Whether to save extracted images to disk
        compare: Whether to compare with text-based extraction
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with extraction results
    """
    try:
        from scripts.analytics.pdf_extractor import PDFStructureExtractor
    except ImportError:
        logger.error("Could not import PDFStructureExtractor")
        logger.error("Make sure you're running from the project root directory")
        return None
    
    # Create extractors
    structural_extractor = PDFStructureExtractor(
        output_dir=output_dir,
        save_images=save_images,
        use_structural_image_extraction=True,
        verbose=verbose
    )
    
    # Extract with structural method
    logger.info(f"Extracting images from {pdf_path}")
    structural_result = structural_extractor.extract_from_pdf(
        pdf_path, output_dir
    )
    
    if compare:
        # Create text-based extractor
        text_extractor = PDFStructureExtractor(
            output_dir=output_dir,
            save_images=save_images,
            use_structural_image_extraction=False,
            verbose=verbose
        )
        
        # Extract with text-based method
        logger.info(
            f"Extracting images from {pdf_path} using text-based extraction"
        )
        text_result = text_extractor.extract_from_pdf(pdf_path, output_dir)
        
        # Compare results
        structural_images = structural_result.get('images', [])
        text_images = text_result.get('images', [])
        
        logger.info(
            f"Structural extraction found {len(structural_images)} images"
        )
        logger.info(f"Text-based extraction found {len(text_images)} images")
        
        # Add comparison to result
        structural_result['comparison'] = {
            'structural_count': len(structural_images),
            'text_based_count': len(text_images),
            'difference': len(structural_images) - len(text_images)
        }
    
    return structural_result


def process_directory(
    directory, output_dir=None, save_images=False, compare=False, verbose=False
):
    """Process all PDF files in a directory.
    
    Args:
        directory: Directory containing PDF files
        output_dir: Directory to save output files
        save_images: Whether to save extracted images to disk
        compare: Whether to compare with text-based extraction
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary mapping file paths to extraction results
    """
    results = {}
    pdf_files = list(Path(directory).glob("**/*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return results
    
    logger.info(f"Processing {len(pdf_files)} PDF files in {directory}")
    
    for pdf_file in pdf_files:
        pdf_path = str(pdf_file)
        logger.info(f"Processing {pdf_path}")
        
        try:
            result = extract_images_from_pdf(
                pdf_path, 
                output_dir=output_dir,
                save_images=save_images,
                compare=compare,
                verbose=verbose
            )
            results[pdf_path] = result
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            results[pdf_path] = {"error": str(e)}
    
    return results


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.getcwd()
    
    # Process PDF file or directory
    path = Path(args.pdf_path)
    
    if path.is_file() and path.suffix.lower() == '.pdf':
        # Process single PDF file
        result = extract_images_from_pdf(
            str(path),
            output_dir=output_dir,
            save_images=args.save_images,
            compare=args.compare,
            verbose=args.verbose
        )
        
        # Print summary
        if result:
            images = result.get('images', [])
            logger.info(f"Found {len(images)} images in {path}")
            
            if args.verbose:
                # Print details of first 5 images
                for i, image in enumerate(images[:5]):
                    logger.info(f"Image {i+1}:")
                    for key, value in image.items():
                        if key != 'text' and key != 'surrounding_text':
                            logger.info(f"  {key}: {value}")
            
            if args.compare:
                comparison = result.get('comparison', {})
                logger.info("Comparison results:")
                logger.info(
                    f"  Structural extraction: "
                    f"{comparison.get('structural_count', 0)} images"
                )
                logger.info(
                    f"  Text-based extraction: "
                    f"{comparison.get('text_based_count', 0)} images"
                )
                logger.info(
                    f"  Difference: {comparison.get('difference', 0)} images"
                )
    
    elif path.is_dir():
        # Process directory of PDF files
        results = process_directory(
            str(path),
            output_dir=output_dir,
            save_images=args.save_images,
            compare=args.compare,
            verbose=args.verbose
        )
        
        # Print summary
        total_structural = 0
        total_text_based = 0
        
        for pdf_path, result in results.items():
            if 'error' in result:
                continue
                
            images = result.get('images', [])
            logger.info(f"Found {len(images)} images in {pdf_path}")
            
            if args.compare:
                comparison = result.get('comparison', {})
                structural_count = comparison.get('structural_count', 0)
                text_based_count = comparison.get('text_based_count', 0)
                
                total_structural += structural_count
                total_text_based += text_based_count
        
        if args.compare:
            logger.info("\nOverall comparison results:")
            logger.info(
                f"  Total structural extraction: {total_structural} images"
            )
            logger.info(
                f"  Total text-based extraction: {total_text_based} images"
            )
            logger.info(
                f"  Difference: {total_structural - total_text_based} images"
            )
    
    else:
        logger.error(f"Invalid path: {path}")
        logger.error(
            "Please provide a PDF file or a directory containing PDF files"
        )
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 