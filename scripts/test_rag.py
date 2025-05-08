#!/usr/bin/env python
"""Test script for the RAG CLI."""

import subprocess


def main() -> None:
    """Run the RAG CLI with a test query."""
    query = 'What is DIN?'

    # Build the command
    cmd = [
        'python',
        '-m',
        'scripts.rag_cli',
        '--vector-db',
        'data/vector_db',
        '--collection-name',
        'test_subset',
        '--no-device-map',
    ]

    # Run the command
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Send the query
    stdout, stderr = process.communicate(input=f'{query}\nexit\n')

    # Print the output
    print('STDOUT:')
    print(stdout)

    if stderr:
        print('\nSTDERR:')
        print(stderr)

    print(f'\nExit code: {process.returncode}')


if __name__ == '__main__':
    main()
