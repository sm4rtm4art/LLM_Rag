"""Tests for core functionality."""

from llm_rag.main import main


def test_main_output(capsys):
    """Test if main function prints the expected output"""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from llm-rag!\n"
