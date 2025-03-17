#!/usr/bin/env python
"""Script to check the project structure and identify any issues."""

import os
from pathlib import Path

# Get the project root (two directories up from this script)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent


def check_directory(path, indent=""):
    """Check a directory structure and print its contents."""
    if not os.path.exists(path):
        print(f"{indent}❌ {path} does not exist")
        return

    if not os.path.isdir(path):
        print(f"{indent}❌ {path} is not a directory")
        return

    print(f"{indent}✅ {path}")

    # Check if the directory has an __init__.py file
    init_path = os.path.join(path, "__init__.py")
    if os.path.exists(init_path):
        print(f"{indent}  ✅ Has __init__.py")
    else:
        print(f"{indent}  ❌ Missing __init__.py")

    # List all files and directories
    for item in sorted(os.listdir(path)):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            check_directory(item_path, indent + "  ")
        else:
            print(f"{indent}  📄 {item}")


def main():
    """Check the project structure."""
    print("\n🔍 Checking project structure...\n")

    # Use absolute paths based on project root
    src_dir = os.path.join(project_root, "src")
    check_directory(src_dir)

    print("\n✅ Check completed!")


if __name__ == "__main__":
    main()
