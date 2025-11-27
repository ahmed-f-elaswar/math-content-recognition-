"""TexTeller: End-to-end LaTeX formula recognition from images.

This package provides tools for:
- Converting images containing LaTeX formulas to LaTeX code
- Detecting and recognizing text and formulas in mixed-content images
- Converting PDF documents to markdown with formula recognition
"""

from importlib.metadata import version
from texteller.api import *

__version__ = version("texteller")
