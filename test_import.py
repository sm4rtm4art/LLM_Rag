#!/usr/bin/env python3
"""
Test script to diagnose import issues with EnhancedPDFProcessor
"""

import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test imports."""
    try:
        # Print Python path and environment
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")
        
        # Add project root to path if not already there
        project_root = os.path.abspath(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            logger.info(f"Added {project_root} to Python path")
        
        # Try to import directly
        logger.info("Attempting direct import...")
        try:
            from scripts.analytics.rag_integration import EnhancedPDFProcessor
            logger.info("Direct import successful!")
            
            # Test creating an instance
            processor = EnhancedPDFProcessor()
            logger.info("Successfully created EnhancedPDFProcessor instance")
            
        except ImportError as e:
            logger.error(f"Direct import failed: {e}")
            
            # Try with importlib
            logger.info("Attempting import with importlib...")
            import importlib.util
            
            module_path = os.path.join(project_root, 'scripts', 'analytics', 'rag_integration.py')
            if os.path.exists(module_path):
                logger.info(f"Module path exists: {module_path}")
                try:
                    spec = importlib.util.spec_from_file_location("rag_integration", module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        EnhancedPDFProcessor = module.EnhancedPDFProcessor
                        logger.info("Importlib import successful!")
                        
                        # Test creating an instance
                        processor = EnhancedPDFProcessor()
                        logger.info("Successfully created EnhancedPDFProcessor instance")
                    else:
                        logger.error("Failed to load module spec")
                except Exception as e:
                    logger.error(f"Importlib import failed: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.error(f"Module path does not exist: {module_path}")
        
        # Try to import PDFStructureExtractor directly
        logger.info("Attempting to import PDFStructureExtractor...")
        try:
            from scripts.analytics.pdf_extractor import PDFStructureExtractor
            logger.info("Successfully imported PDFStructureExtractor")
            
            # Test creating an instance
            extractor = PDFStructureExtractor()
            logger.info("Successfully created PDFStructureExtractor instance")
        except Exception as e:
            logger.error(f"Failed to import PDFStructureExtractor: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 