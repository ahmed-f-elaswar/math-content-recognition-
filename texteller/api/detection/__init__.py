"""LaTeX formula detection module.

Provides functionality for detecting and classifying LaTeX formulas
in images as either isolated (display) or embedded (inline) formulas.
"""

from .detect import latex_detect

__all__ = ["latex_detect"]
