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
    """Represents a page from a PDF with its content."""
    
    def __init__(self, page_num: int, text: str, image: np.ndarray):
        self.page_num = page_num
        self.text = text  # extracted text
        self.image = image  # page as image (RGB numpy array)
        self.recognized_content = None  # will store OCR/formula recognition result


def pdf_to_pages(pdf_path: str, dpi: int = 300) -> List[PDFPage]:
    """
    Convert PDF to list of pages with both text and images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering pages as images
        
    Returns:
        List of PDFPage objects
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
    """
    Merge original PDF text with recognized content.
    
    Strategy:
    - If original text is empty or very short, use recognized text
    - Otherwise, combine both with clear separation
    
    Args:
        original_text: Text extracted from PDF
        recognized_text: Text recognized from OCR/formula recognition
        page_num: Page number
        
    Returns:
        Combined text in markdown format
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
    """
    Convert entire PDF to markdown with formulas.
    
    Args:
        pdf_path: Path to PDF file
        latexdet_model: LaTeX detection model
        textdet_model: Text detection model
        textrec_model: Text recognition model
        latexrec_model: LaTeX recognition model
        tokenizer: Tokenizer for LaTeX model
        device: Torch device
        num_beams: Beam search parameter
        dpi: Resolution for page rendering
        
    Returns:
        Complete markdown document
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
