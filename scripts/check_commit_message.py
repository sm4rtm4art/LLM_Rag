#!/usr/bin/env python
"""Check if commit message follows semantic versioning format."""

import re
import sys


def main() -> int:
    """Check if commit message follows semantic versioning format."""
    try:
        with open(sys.argv[1], "r") as f:
            commit_msg = f.read().strip()
    except (IndexError, FileNotFoundError):
        print("Error: Could not read commit message file")
        return 1

    # Pattern for conventional commits
    # format: type(scope): description
    pattern = r"^(feat|fix|docs|style|refactor|perf|test|chore)" r"(\([a-z0-9-]+\))?(!)?: .+"

    if not re.match(pattern, commit_msg):
        print("Error: Commit message does not follow semantic versioning format")
        print("Expected format: type(scope): description")
        print("Examples:")
        print("  feat: add new feature")
        print("  fix(auth): fix login issue")
        print("  docs: update README")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
