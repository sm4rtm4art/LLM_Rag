#!/usr/bin/env python3
"""Script to run pytest with a nice loading indicator using Rich."""

import argparse
import subprocess
import sys
import time

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def run_tests(args=None):
    """Run pytest with a nice spinner and progress indicator."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run pytest with a progress spinner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-x", "--exitfirst", action="store_true", help="Exit on first error")
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("-p", "--path", type=str, default="tests", help="Path to test files")
    parser.add_argument("extra_args", nargs="*", help="Additional arguments to pass to pytest")
    parsed_args = parser.parse_args(args)

    # Build the pytest command
    command = ["pytest"]

    if parsed_args.verbose:
        command.append("-v")

    if parsed_args.exitfirst:
        command.append("-x")

    if parsed_args.coverage:
        command.extend(["--cov=src/llm_rag", "--cov-report=xml"])

    command.append(parsed_args.path)

    # Add any extra arguments
    if parsed_args.extra_args:
        command.extend(parsed_args.extra_args)

    # Log the command
    console.print(f"[dim]Running: {' '.join(command)}[/dim]")

    # Start the subprocess without waiting
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Set up a spinner to show progress while tests are running
    with Progress(
        SpinnerColumn(style="bold green"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Running tests...", total=None)

        # Poll the process every 0.1 seconds until done
        while process.poll() is None:
            time.sleep(0.1)
            progress.update(task)

    # Process has completed, get the return code
    return_code = process.returncode

    # Collect and display the output
    stdout, stderr = process.communicate()

    # Print the output with proper styling
    if stdout:
        console.print(stdout)

    if stderr:
        console.print("[bold red]Errors:[/bold red]")
        console.print(stderr)

    # Show a summary based on the return code
    if return_code == 0:
        console.print("[bold green]✓ All tests passed![/bold green]")
    else:
        console.print(f"[bold red]✗ Tests failed with exit code {return_code}[/bold red]")

    return return_code


if __name__ == "__main__":
    sys.exit(run_tests())
