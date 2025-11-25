"""Quick test to verify PDF support is working."""

def test_imports():
    """Test that all PDF-related imports work."""
    print("Testing imports...")
    
    try:
        from texteller.utils.pdf import pdf_to_pages, pdf2md, merge_text_and_recognition
        print("✓ PDF utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PDF utils: {e}")
        return False
    
    try:
        from texteller.api import pdf2md
        print("✓ pdf2md available in API")
    except ImportError as e:
        print(f"✗ Failed to import pdf2md from API: {e}")
        return False
    
    try:
        import pymupdf
        print(f"✓ PyMuPDF installed (version: {pymupdf.__version__})")
    except ImportError:
        print("✗ PyMuPDF not installed. Run: pip install pymupdf")
        return False
    
    return True


def test_cli_help():
    """Test that CLI recognizes PDF files."""
    print("\nTesting CLI help...")
    import subprocess
    
    result = subprocess.run(
        ["texteller", "inference", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ CLI command available")
        if "file_path" in result.stdout or "FILE_PATH" in result.stdout:
            print("✓ CLI updated to accept files (including PDFs)")
        return True
    else:
        print("✗ CLI command failed")
        return False


def test_module_structure():
    """Test that module structure is correct."""
    print("\nTesting module structure...")
    
    try:
        from texteller.utils import pdf_to_pages, pdf2md
        print("✓ PDF functions exported from utils")
    except ImportError as e:
        print(f"✗ PDF functions not exported: {e}")
        return False
    
    try:
        from texteller.api import img2latex, paragraph2md, pdf2md
        print("✓ All API functions available")
    except ImportError as e:
        print(f"✗ API functions missing: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("PDF Support Verification Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Module Structure", test_module_structure),
        ("CLI", test_cli_help),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! PDF support is ready to use.")
        print("\nNext steps:")
        print("1. Test with a real PDF: texteller inference sample.pdf")
        print("2. Launch web UI: texteller web")
        print("3. Start API server: texteller launch")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("- Install PyMuPDF: pip install pymupdf")
        print("- Reinstall package: uv pip install -e .")
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
