"""Public API for TexTeller.

This module exports all user-facing functions for:
- Loading models
- Converting images to LaTeX
- Processing mixed text/formula images
- Converting PDFs to markdown
- Format conversion and detection
"""

from .detection import latex_detect
from .format import format_latex
from .inference import img2latex, paragraph2md
from .katex import to_katex
from .load import (
    load_latexdet_model,
    load_model,
    load_textdet_model,
    load_textrec_model,
    load_tokenizer,
)
from texteller.utils.pdf import pdf2md

__all__ = [
    "to_katex",
    "format_latex",
    "img2latex",
    "paragraph2md",
    "pdf2md",
    "load_model",
    "load_tokenizer",
    "load_latexdet_model",
    "load_textrec_model",
    "load_textdet_model",
    "latex_detect",
]
