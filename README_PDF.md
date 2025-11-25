# PDF Support for TexTeller

TexTeller now supports PDF files! The system extracts text and images from PDFs, processes mathematical formulas, and combines everything in the original order.

## Features

- **PDF Processing**: Convert entire PDF documents to markdown with recognized formulas
- **Text Extraction**: Preserves original PDF text when available
- **Formula Recognition**: Detects and converts mathematical formulas to LaTeX
- **Order Preservation**: Maintains the original document structure
- **Multiple Interfaces**: Available in CLI, Web UI, and API server

## Installation

Install with PDF support:

```bash
uv pip install texteller
pip install pymupdf  # Required for PDF support
```

Or install from source with PDF dependencies:

```bash
uv pip install -e .
```

## Usage

### 1. Command Line Interface (CLI)

Process a PDF file:

```bash
texteller inference document.pdf --output-file output.md
```

Process with custom beam search:

```bash
texteller inference document.pdf --output-file output.md --num-beams 5
```

Process an image (original functionality):

```bash
texteller inference image.png --output-format katex
```

### 2. Web Interface

Launch the web demo:

```bash
texteller web
```

Then:
1. Open http://localhost:8501 in your browser
2. Upload a PDF, PNG, or JPG file
3. Wait for processing
4. View the markdown output with rendered formulas

### 3. Python API

```python
from texteller.api import (
    load_model,
    load_tokenizer,
    load_latexdet_model,
    load_textdet_model,
    load_textrec_model,
    pdf2md,
)
from texteller.utils import get_device

# Load models
latexrec_model = load_model()
tokenizer = load_tokenizer()
latexdet_model = load_latexdet_model()
textdet_model = load_textdet_model()
textrec_model = load_textrec_model()

# Process PDF
markdown = pdf2md(
    pdf_path="document.pdf",
    latexdet_model=latexdet_model,
    textdet_model=textdet_model,
    textrec_model=textrec_model,
    latexrec_model=latexrec_model,
    tokenizer=tokenizer,
    device=get_device(),
    num_beams=1,
    dpi=300,  # Resolution for page rendering
)

# Save output
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

### 4. API Server

Start the server:

```bash
texteller launch
```

Send PDF via HTTP request:

```python
import requests

server_url = "http://127.0.0.1:8000/predict"

# Upload PDF
with open("document.pdf", 'rb') as pdf_file:
    files = {'pdf': pdf_file}
    response = requests.post(server_url, files=files)

markdown_result = response.text
print(markdown_result)
```

For images (original functionality):

```python
with open("image.png", 'rb') as img:
    files = {'img': img}
    response = requests.post(server_url, files=files)
```

## How It Works

1. **PDF Parsing**: Each page is converted to high-resolution images (default 300 DPI)
2. **Text Extraction**: Original text is extracted from the PDF when available
3. **Formula Detection**: Mathematical formulas are detected in the rendered page images
4. **OCR Processing**: Text regions are recognized using OCR
5. **Formula Recognition**: Formulas are converted to LaTeX using TexTeller
6. **Combination**: All content is merged in the original order
7. **Output**: Returns markdown with both text and LaTeX formulas

## Output Format

The output is markdown with:
- Page headers (`## Page N`)
- Original PDF text (when available)
- Recognized content with formulas
- Inline formulas: `$formula$`
- Display formulas: `$$formula$$`

Example output:

```markdown
# Document: example.pdf

## Page 1

### Original Text

This is a quadratic equation.

### Recognized Content (with formulas)

This is a quadratic equation: $ax^2 + bx + c = 0$

The solution is:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

---

## Page 2

...
```

## Configuration Options

- `--num-beams`: Beam search parameter for better accuracy (default: 1)
- `--output-file`: Save output to file
- `--dpi`: Resolution for PDF rendering (default: 300)

## Performance Tips

1. **DPI Setting**: Lower DPI (150-200) for faster processing, higher (300-600) for better accuracy
2. **Beam Search**: Higher `num_beams` (3-5) improves accuracy but slows processing
3. **GPU**: Use CUDA-enabled GPU for significant speedup
4. **Batch Processing**: Process multiple pages in parallel (coming soon)

## Limitations

- Complex layouts may not preserve exact formatting
- Scanned PDFs work better than text-based PDFs for formula recognition
- Very large PDFs may require significant memory
- Processing time increases with document length and complexity

## Troubleshooting

**ImportError: No module named 'fitz' or 'pymupdf'**
```bash
pip install pymupdf
```

**Poor formula recognition**
- Increase DPI: `--dpi 600`
- Increase beam search: `--num-beams 5`
- Ensure high-quality PDF source

**Out of memory**
- Reduce DPI
- Process fewer pages at once
- Use CPU instead of GPU for very large documents

## Examples

See the `examples/` directory:
- `client_demo_pdf.py`: API client example
- `pdf_usage_example.py`: Python API usage

## Contributing

PDF support is a new feature. Feedback and contributions are welcome!
