"""Test script to verify llama-cpp-python installation.

This script checks if the llama-cpp package can be imported successfully
and prints system information.
"""

import os
import platform
import sys

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print()

print("Testing llama-cpp-python import...")
try:
    from llama_cpp import Llama

    print("✅ Successfully imported llama_cpp.Llama")

    # Try to initialize the model class (without loading a model)
    print("Testing Llama class initialization...")
    try:
        # Just test the class, don't actually load a model
        llama_class = Llama
        print(f"✅ Llama class exists: {llama_class}")
        print("llama-cpp-python is correctly installed!")
    except Exception as e:
        print(f"❌ Error initializing Llama class: {e}")

except ImportError as e:
    print(f"❌ Error importing llama_cpp: {e}")

print("\nChecking for CUDA support...")
try:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("torch not installed, can't check CUDA support")
