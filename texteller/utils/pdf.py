"""PDF processing utilities for extracting text and images."""

import io
import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz  # fallback
    except ImportError:
        fitz = None

from texteller.logger import get_logger

_logger = get_logger()


class PDFPage:
    """Represents a single page from a PDF document with extracted content.
    
    This class encapsulates both the original text content and the rendered
    image of a PDF page. It's used during PDF-to-markdown conversion.
    
    Attributes:
        page_num: Page number (1-indexed)
        text: Text extracted directly from the PDF
        image: Rendered page as RGB numpy array (H, W, 3)
        recognized_content: OCR and formula recognition results (optional)
    """
    
    def __init__(self, page_num: int, text: str, image: np.ndarray):
        """Initialize a PDF page.
        
        Args:
            page_num: Page number (1-indexed)
            text: Extracted text from the PDF
            image: Page rendered as RGB numpy array
        """
        self.page_num = page_num
        self.text = text
        self.image = image
        self.recognized_content = None


def pdf_to_pages(pdf_path: str, dpi: int = 300) -> List[PDFPage]:
    """Convert a PDF document to a list of page objects.
    
    Each page is extracted with both its text content and a high-resolution
    image rendering. The text is useful when the PDF has selectable text,
    while the image is used for OCR and formula recognition.
    
    Args:
        pdf_path: Path to the PDF file to process
        dpi: Resolution for rendering pages as images. Higher DPI produces
             better quality images but increases processing time.
             Recommended: 150-200 for speed, 300-600 for quality.
             Default is 300.
        
    Returns:
        List of PDFPage objects, one per page in the document
        
    Raises:
        ImportError: If PyMuPDF (pymupdf) is not installed
        FileNotFoundError: If the PDF file doesn't exist
        
    Example:
        >>> pages = pdf_to_pages('document.pdf', dpi=300)
        >>> print(f'Extracted {len(pages)} pages')
        >>> print(f'Page 1 text: {pages[0].text[:100]}')
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF (pymupdf) is required for PDF support. "
            "Install it with: pip install pymupdf"
        )
    
    pages = []
    doc = fitz.open(pdf_path)
    
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text("text")
            
            # Render page as image
            zoom = dpi / 72  # 72 is the default DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array (RGB)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img.convert("RGB"))
            
            pages.append(PDFPage(page_num + 1, text, img_array))
            
    finally:
        doc.close()
    
    _logger.info(f"Extracted {len(pages)} pages from PDF")
    return pages


def merge_text_and_recognition(
    original_text: str, 
    recognized_text: str, 
    page_num: int
) -> str:
    """Intelligently merge original PDF text with OCR/formula recognition.
    
    This function combines the original extractable text from the PDF
    (when available) with the OCR and formula recognition results.
    The merging strategy depends on the quality of the original text:
    
    - If original text is substantial (>20 chars), both are included with
      clear section headers for comparison
    - If original text is minimal or empty (scanned PDFs), only the
      recognized content is included
    
    Args:
        original_text: Text extracted directly from the PDF
        recognized_text: Text and formulas recognized via OCR/ML models
        page_num: Page number for the section header
        
    Returns:
        Formatted markdown string with appropriate headers and content
        
    Example:
        >>> original = "This is a mathematical equation."
        >>> recognized = "This is a mathematical equation: $E = mc^2$"
        >>> result = merge_text_and_recognition(original, recognized, 1)
        >>> print(result)
        ## Page 1
        
        ### Original Text
        This is a mathematical equation.
        
        ### Recognized Content (with formulas)
        This is a mathematical equation: $E = mc^2$
    """
    result = f"## Page {page_num}\n\n"
    
    # If PDF has extractable text
    if original_text and len(original_text.strip()) > 20:
        result += "### Original Text\n\n"
        result += original_text.strip() + "\n\n"
        
        result += "### Recognized Content (with formulas)\n\n"
        result += recognized_text.strip() + "\n\n"
    else:
        # No extractable text, just use recognition
        result += recognized_text.strip() + "\n\n"
    
    return result


def pdf2md(
    pdf_path: str,
    latexdet_model,
    textdet_model,
    textrec_model,
    latexrec_model,
    tokenizer,
    device=None,
    num_beams: int = 1,
    dpi: int = 300,
) -> str:
    """Convert a complete PDF document to markdown with LaTeX formulas.
    
    This is the main entry point for PDF processing. It orchestrates the
    entire pipeline:
    1. Extract pages with text and images
    2. Detect and recognize formulas on each page
    3. Detect and recognize regular text via OCR
    4. Merge everything into a coherent markdown document
    
    The resulting markdown includes:
    - Page headers
    - Original PDF text (when available)
    - Recognized content with inline ($formula$) and display ($$formula$$) math
    - Preserved reading order (top-to-bottom, left-to-right)
    
    Args:
        pdf_path: Path to the PDF file to process
        latexdet_model: ONNX InferenceSession for detecting LaTeX formulas
        textdet_model: PaddleOCR TextDetector for detecting text regions
        textrec_model: PaddleOCR TextRecognizer for recognizing text
        latexrec_model: TexTeller model for converting formulas to LaTeX
        tokenizer: RobertaTokenizerFast for the LaTeX recognition model
        device: torch.device for computation (None = auto-detect)
        num_beams: Number of beams for beam search (1 = greedy, higher = better quality)
        dpi: Resolution for rendering PDF pages as images (150-600 recommended)
        
    Returns:
        Complete markdown document as a string, with all pages processed
        
    Raises:
        ImportError: If PyMuPDF is not installed
        FileNotFoundError: If the PDF file doesn't exist
        
    Example:
        >>> from texteller.api import (
        ...     load_model, load_tokenizer, load_latexdet_model,
        ...     load_textdet_model, load_textrec_model, pdf2md
        ... )
        >>> 
        >>> # Load all required models
        >>> latexdet = load_latexdet_model()
        >>> textdet = load_textdet_model()
        >>> textrec = load_textrec_model()
        >>> latexrec = load_model()
        >>> tokenizer = load_tokenizer()
        >>> 
        >>> # Convert PDF to markdown
        >>> markdown = pdf2md(
        ...     pdf_path='math_paper.pdf',
        ...     latexdet_model=latexdet,
        ...     textdet_model=textdet,
        ...     textrec_model=textrec,
        ...     latexrec_model=latexrec,
        ...     tokenizer=tokenizer,
        ...     num_beams=3,
        ...     dpi=300
        ... )
        >>> 
        >>> # Save result
        >>> with open('output.md', 'w', encoding='utf-8') as f:
        ...     f.write(markdown)
    """
    from texteller.api.inference import paragraph2md
    
    pages = pdf_to_pages(pdf_path, dpi=dpi)
    full_markdown = f"# Document: {Path(pdf_path).name}\n\n"
    
    # Process each page
    for page in pages:
        _logger.info(f"Processing page {page.page_num}...")
        
        # Save page image to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(page.image).save(tmp.name)
            
            # Recognize content from image
            recognized = paragraph2md(
                img_path=tmp.name,
                latexdet_model=latexdet_model,
                textdet_model=textdet_model,
                textrec_model=textrec_model,
                latexrec_model=latexrec_model,
                tokenizer=tokenizer,
                device=device,
                num_beams=num_beams,
            )
            
            # Merge with original text
            page_content = merge_text_and_recognition(
                page.text, recognized, page.page_num
            )
            full_markdown += page_content + "\n\n---\n\n"
    
    return full_markdown.strip()
