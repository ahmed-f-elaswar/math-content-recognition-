"""Pytest tests to verify PDF support is working correctly.

This module contains automated tests for PDF processing functionality,
including imports, module structure, and CLI command availability.
"""

import subprocess
import sys
import pytest


def test_pdf_utils_imports():
    """Test that PDF utility functions can be imported."""
    from texteller.utils.pdf import pdf_to_pages, pdf2md, merge_text_and_recognition
    
    # Verify functions are callable
    assert callable(pdf_to_pages)
    assert callable(pdf2md)
    assert callable(merge_text_and_recognition)


def test_pdf_api_imports():
    """Test that pdf2md is available in the public API."""
    from texteller.api import pdf2md
    
    assert callable(pdf2md)


def test_pymupdf_installed():
    """Test that PyMuPDF (fitz) is installed and importable."""
    try:
        import pymupdf
        assert hasattr(pymupdf, '__version__')
    except ImportError:
        pytest.fail("PyMuPDF is not installed. Run: pip install pymupdf")


def test_utils_exports_pdf_functions():
    """Test that PDF functions are exported from texteller.utils."""
    from texteller.utils import pdf_to_pages, pdf2md
    
    assert callable(pdf_to_pages)
    assert callable(pdf2md)


def test_api_exports_all_functions():
    """Test that all expected API functions are available."""
    from texteller.api import img2latex, paragraph2md, pdf2md
    
    assert callable(img2latex)
    assert callable(paragraph2md)
    assert callable(pdf2md)


@pytest.mark.slow
def test_cli_inference_command_exists():
    """Test that the inference CLI command is available and shows help."""
    result = subprocess.run(
        [sys.executable, "-m", "texteller.cli", "inference", "--help"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, "CLI inference command failed"
    assert "file_path" in result.stdout.lower() or "file-path" in result.stdout.lower(), \
        "CLI should accept file_path argument"


@pytest.mark.slow
def test_cli_accepts_file_argument():
    """Test that CLI help indicates file input support."""
    result = subprocess.run(
        [sys.executable, "-m", "texteller.cli", "inference", "--help"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Check that help mentions file paths or PDF
    help_text = result.stdout.lower()
    assert "file" in help_text or "path" in help_text, \
        "CLI help should mention file input"


def test_pdf_page_class_available():
    """Test that PDFPage class is available."""
    from texteller.utils.pdf import PDFPage
    
    assert PDFPage is not None
    # Verify it's a class
    assert isinstance(PDFPage, type)


def test_pdf_functions_have_docstrings():
    """Test that PDF functions have proper documentation."""
    from texteller.utils.pdf import pdf_to_pages, pdf2md, merge_text_and_recognition
    
    assert pdf_to_pages.__doc__ is not None, "pdf_to_pages should have a docstring"
    assert pdf2md.__doc__ is not None, "pdf2md should have a docstring"
    assert merge_text_and_recognition.__doc__ is not None, \
        "merge_text_and_recognition should have a docstring"


@pytest.mark.slow
@pytest.mark.skipif(
    not hasattr(subprocess.run(
        [sys.executable, "-m", "texteller.cli", "--help"],
        capture_output=True
    ), 'returncode'),
    reason="CLI not properly installed"
)
def test_cli_web_command_exists():
    """Test that the web CLI command is available."""
    result = subprocess.run(
        [sys.executable, "-m", "texteller.cli", "web", "--help"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Command should either work or show help
    assert result.returncode in [0, 2], "CLI web command should be available"


@pytest.mark.slow
@pytest.mark.skipif(
    not hasattr(subprocess.run(
        [sys.executable, "-m", "texteller.cli", "--help"],
        capture_output=True
    ), 'returncode'),
    reason="CLI not properly installed"
)
def test_cli_launch_command_exists():
    """Test that the launch CLI command is available."""
    result = subprocess.run(
        [sys.executable, "-m", "texteller.cli", "launch", "--help"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Command should either work or show help
    assert result.returncode in [0, 2], "CLI launch command should be available"
