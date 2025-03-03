#!/usr/bin/env python3

import importlib.util
import sys


def check_module_available(module_name):
    """Check if a module is available for import."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None


print("Python version:", sys.version)
print("sys.path:", sys.path)

# Check if llm_rag module is available
if check_module_available("llm_rag"):
    print("llm_rag module is available")
else:
    print("llm_rag module is NOT available")

# Check if src.llm_rag module is available
if check_module_available("src.llm_rag"):
    print("src.llm_rag module is available")
else:
    print("src.llm_rag module is NOT available")
