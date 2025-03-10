#!/usr/bin/env python
"""Script to check the project structure and identify any issues."""

import os


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
    """Main entry point for the script."""
    # Check the pipeline module structure
    pipeline_path = "./src/llm_rag/rag/pipeline"

    print("\n🔍 Checking project structure...\n")

    # Check the entire src directory structure
    check_directory("./src")

    print("\n✅ Check completed!")


if __name__ == "__main__":
    main()
