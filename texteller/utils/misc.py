"""Miscellaneous utility functions."""

from textwrap import dedent


def lines_dedent(s: str) -> str:
    """Remove common leading whitespace from multi-line strings.
    
    Useful for cleaning up indented multi-line strings in source code
    while preserving their relative indentation.
    
    Args:
        s: Multi-line string with common leading whitespace
        
    Returns:
        String with common leading whitespace removed and trailing whitespace stripped
        
    Example:
        >>> text = '''
        ...     Line 1
        ...     Line 2
        ... '''
        >>> lines_dedent(text)
        'Line 1\\nLine 2'
    """
    return dedent(s).strip()
