#!/usr/bin/env python3
"""Test script to diagnose import issues with EnhancedPDFProcessor."""

import logging
import os
import sys
import traceback
from importlib import import_module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Execute import tests to diagnose module issues."""
    try:
        # Print Python path and environment
        logger.info('Python executable: %s', sys.executable)
        logger.info('Python version: %s', sys.version)
        logger.info('PYTHONPATH: %s', os.environ.get('PYTHONPATH', 'Not set'))
        logger.info('Current directory: %s', os.getcwd())
        logger.info('sys.path: %s', sys.path)

        # Try direct import
        logger.info('Trying direct import...')
        try:
            from scripts.analytics.analyze_pdf_enhanced import EnhancedPDFProcessor

            logger.info('Successfully imported EnhancedPDFProcessor')
            # Test creating an instance - using _ to indicate intentionally unused
            _ = EnhancedPDFProcessor()
            logger.info('Successfully created EnhancedPDFProcessor instance')
        except ImportError as e:
            logger.error('Failed to import EnhancedPDFProcessor: %s', e)
            logger.error('Traceback: %s', traceback.format_exc())

            # Try importing the module
            logger.info('Trying to import the module...')
            try:
                from scripts.analytics import analyze_pdf_enhanced

                logger.info('Successfully imported scripts.analytics.analyze_pdf_enhanced')
                if hasattr(analyze_pdf_enhanced, 'EnhancedPDFProcessor'):
                    logger.info('EnhancedPDFProcessor class found in module')
                    # Test creating an instance
                    _ = analyze_pdf_enhanced.EnhancedPDFProcessor()
                    logger.info('Successfully created EnhancedPDFProcessor instance')
                else:
                    logger.error('EnhancedPDFProcessor class not found in module')
            except ImportError as e:
                logger.error('Failed to import scripts.analytics.analyze_pdf_enhanced: %s', e)
                logger.error('Traceback: %s', traceback.format_exc())

                # Try alternative import approach
                logger.info('Trying alternative import approach...')
                try:
                    module = import_module('scripts.analytics.analyze_pdf_enhanced')
                    logger.info('Successfully imported module via import_module')

                    # Check if EnhancedPDFProcessor is available
                    if hasattr(module, 'EnhancedPDFProcessor'):
                        logger.info('EnhancedPDFProcessor class found in module')
                    else:
                        logger.error('EnhancedPDFProcessor class not found in module')
                except Exception as e:
                    logger.error('Alternative import approach failed: %s', e)

        # Try importing PDFStructureExtractor
        logger.info('Trying to import PDFStructureExtractor...')
        try:
            from scripts.analytics.pdf_extractor import PDFStructureExtractor

            logger.info('Successfully imported PDFStructureExtractor')
            # Test creating an instance
            _ = PDFStructureExtractor()
            logger.info('Successfully created PDFStructureExtractor instance')
        except Exception as e:
            logger.error('Failed to import PDFStructureExtractor: %s', e)
            logger.error('Traceback: %s', traceback.format_exc())

    except Exception as e:
        logger.error('Unexpected error: %s', e)
        logger.error('Traceback: %s', traceback.format_exc())


if __name__ == '__main__':
    main()
