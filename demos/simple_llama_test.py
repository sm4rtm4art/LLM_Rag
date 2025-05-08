"""Simple test script for llama-cpp-python.

This script loads a GGUF model and runs a simple query.
"""

import argparse
import os
import sys

from llama_cpp import Llama


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description='Simple test for llama-cpp-python')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/llama-2-7b-chat.Q4_K_M.gguf',
        help='Path to the GGUF model file',
    )
    parser.add_argument(
        '--n-gpu-layers',
        type=int,
        default=0,
        help='Number of layers to offload to GPU (0 for CPU only)',
    )
    parser.add_argument(
        '--n-ctx',
        type=int,
        default=2048,
        help='Context size for the model',
    )
    parser.add_argument(
        '--query',
        type=str,
        default='What is Retrieval-Augmented Generation?',
        help='Query to run',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )
    return parser


def main() -> None:
    """Run the main application logic."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Check if the model file exists
    if not os.path.exists(args.model_path):
        print(f'Error: Model file not found at {args.model_path}')
        print('Please download a GGUF model and place it in the models directory.')
        sys.exit(1)

    print(f'Loading model from {args.model_path}...')
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        verbose=args.verbose,
    )

    print(f'Running query: {args.query}')
    output = llm(
        args.query,
        max_tokens=512,
        temperature=0.7,
        stop=['Human:', 'Assistant:'],
    )

    print('\nResponse:')
    print(output['choices'][0]['text'])


if __name__ == '__main__':
    main()
