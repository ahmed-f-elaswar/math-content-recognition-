# PDF Support Implementation Summary

## Overview
Added comprehensive PDF support to TexTeller across all interfaces (CLI, Web, and API Server). The system now:
1. Extracts text and converts PDF pages to images
2. Processes each page for formulas and text
3. Combines original PDF text with recognized content
4. Maintains original document order
5. Outputs markdown with LaTeX formulas

## Files Created

### 1. `texteller/utils/pdf.py`
Core PDF processing utilities:
- `PDFPage`: Class representing a PDF page with text and image
- `pdf_to_pages()`: Converts PDF to list of pages
- `merge_text_and_recognition()`: Combines original and recognized text
- `pdf2md()`: Main function to convert entire PDF to markdown

### 2. `examples/client_demo_pdf.py`
Example client demonstrating how to upload PDFs to the API server

### 3. `examples/pdf_usage_example.py`
Python API usage example for PDF processing

### 4. `README_PDF.md`
Complete documentation for PDF features

## Files Modified

### 1. `texteller/utils/__init__.py`
- Added PDF utility imports

### 2. `texteller/api/__init__.py`
- Exported `pdf2md` function

### 3. `texteller/cli/commands/inference.py`
- Added PDF file detection (`.pdf` extension)
- Integrated PDF processing workflow
- Added `--output-file` option for saving results
- Added `--num-beams` option

### 4. `texteller/cli/commands/web/streamlit_demo.py`
- Updated file uploader to accept PDF files
- Added PDF detection and handling logic
- Integrated PDF processing in the UI
- Added PDF-specific rendering

### 5. `texteller/cli/commands/launch/server.py`
- Added PDF processing models to server initialization
- Implemented `predict_pdf()` method
- Updated ingress to handle both PDF and image uploads
- Server now accepts 'pdf' form field

### 6. `pyproject.toml`
- Added `pymupdf>=1.24.0` dependency for PDF support

## Usage Examples

### CLI
```bash
# Process PDF
texteller inference document.pdf --output-file output.md --num-beams 3

# Process image (existing)
texteller inference image.png --output-format katex
```

### Web Interface
```bash
texteller web
# Upload PDF, PNG, or JPG files
```

### Python API
```python
from texteller.api import pdf2md, load_model, load_tokenizer, ...

markdown = pdf2md(
    pdf_path="document.pdf",
    latexdet_model=latexdet_model,
    textdet_model=textdet_model,
    textrec_model=textrec_model,
    latexrec_model=latexrec_model,
    tokenizer=tokenizer,
    device=get_device(),
    num_beams=1,
    dpi=300,
)
```

### API Server
```python
import requests

# Upload PDF
with open("document.pdf", 'rb') as pdf_file:
    files = {'pdf': pdf_file}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)

# Upload image (existing)
with open("image.png", 'rb') as img:
    files = {'img': img}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)
```

## Installation

Install the updated package:
```bash
# Install with PDF support
uv pip install -e .

# Or install PyMuPDF separately
pip install pymupdf
```

## Features

✅ **CLI Interface**: Process PDFs via command line
✅ **Web Interface**: Upload and process PDFs in browser
✅ **API Server**: RESTful endpoint for PDF processing
✅ **Text Preservation**: Keeps original PDF text when available
✅ **Formula Recognition**: Detects and converts math formulas
✅ **Order Preservation**: Maintains document structure
✅ **Markdown Output**: Clean, readable format with LaTeX
✅ **Configurable DPI**: Adjustable image quality
✅ **Beam Search**: Improved accuracy with multiple beams

## Technical Details

### PDF Processing Flow
1. Open PDF with PyMuPDF (fitz)
2. For each page:
   - Extract text using `page.get_text("text")`
   - Render page as high-res image using `page.get_pixmap()`
   - Convert to RGB numpy array
3. Process page image through paragraph2md:
   - Detect formulas
   - Detect and recognize text
   - Convert formulas to LaTeX
4. Merge original text with recognized content
5. Combine all pages into single markdown document

### Output Format
```markdown
# Document: filename.pdf

## Page 1

### Original Text
[PDF extracted text if available]

### Recognized Content (with formulas)
[OCR text with $inline$ and $$display$$ formulas]

---

## Page 2
...
```

## Testing

Test the implementation:

```bash
# 1. Test CLI
texteller inference test.pdf --output-file test.md

# 2. Test Web UI
texteller web
# Upload a PDF in browser

# 3. Test API Server
texteller launch
# Run client_demo_pdf.py
```

## Next Steps / Future Improvements

1. **Batch Processing**: Process multiple pages in parallel
2. **Layout Detection**: Better preserve complex layouts
3. **Table Recognition**: Improved table handling
4. **Image Extraction**: Save embedded images separately
5. **Streaming**: Process large PDFs page-by-page
6. **Progress Tracking**: Show processing progress for multi-page docs
7. **Caching**: Cache processed pages for faster re-processing

## Compatibility

- Requires Python 3.10+
- PyMuPDF (pymupdf) for PDF handling
- All existing image processing features remain unchanged
- Backward compatible with existing code
