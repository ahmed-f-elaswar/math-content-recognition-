class Point:
    """Represents a 2D point in image coordinates.
    
    Attributes:
        x: Horizontal coordinate (pixels from left edge)
        y: Vertical coordinate (pixels from top edge)
    """
    
    def __init__(self, x: int, y: int):
        """Initialize a point with integer coordinates.
        
        Args:
            x: Horizontal coordinate
            y: Vertical coordinate
        """
        self.x = int(x)
        self.y = int(y)

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"


class Bbox:
    """Represents a bounding box for detected text or formula regions.
    
    This class is used to represent detected regions in images, including
    both text (OCR) and mathematical formulas. It provides methods for
    spatial comparisons and sorting.
    
    Attributes:
        p: Upper-left corner point
        h: Height in pixels
        w: Width in pixels
        label: Region type ('text', 'embedding', 'isolated', etc.)
        confidence: Detection confidence score (0-1)
        content: Recognized text or LaTeX formula
        THREADHOLD: Threshold for same-row detection (class constant)
    """
    
    THREADHOLD: float = 0.4  # Threshold for determining if boxes are on same row

    def __init__(self, x: int, y: int, h: int, w: int, label: str = None, confidence: float = 0, content: str = None):
        """Initialize a bounding box.
        
        Args:
            x: X-coordinate of upper-left corner
            y: Y-coordinate of upper-left corner
            h: Height of the bounding box
            w: Width of the bounding box
            label: Optional label identifying the content type
            confidence: Optional confidence score from detector
            content: Optional recognized text or LaTeX content
        """
        self.p = Point(x, y)
        self.h = int(h)
        self.w = int(w)
        self.label = label
        self.confidence = confidence
        self.content = content

    @property
    def ul_point(self) -> Point:
        """Get the upper-left corner point."""
        return self.p

    @property
    def ur_point(self) -> Point:
        """Get the upper-right corner point."""
        return Point(self.p.x + self.w, self.p.y)

    @property
    def ll_point(self) -> Point:
        """Get the lower-left corner point."""
        return Point(self.p.x, self.p.y + self.h)

    @property
    def lr_point(self) -> Point:
        """Get the lower-right corner point."""
        return Point(self.p.x + self.w, self.p.y + self.h)

    def same_row(self, other: 'Bbox') -> bool:
        """Determine if this bounding box is on the same row as another.
        
        Uses vertical overlap and relative position to determine if two boxes
        are on the same horizontal line (e.g., text on the same line).
        
        Args:
            other: Another Bbox to compare with
            
        Returns:
            True if boxes are on the same row, False otherwise
        """
        if (self.p.y >= other.p.y and self.ll_point.y <= other.ll_point.y) or (
            self.p.y <= other.p.y and self.ll_point.y >= other.ll_point.y
        ):
            return True
        if self.ll_point.y <= other.p.y or self.p.y >= other.ll_point.y:
            return False
        return 1.0 * abs(self.p.y - other.p.y) / max(self.h, other.h) < self.THREADHOLD

    def __lt__(self, other: 'Bbox') -> bool:
        """Compare bounding boxes for sorting (top-to-bottom, left-to-right).
        
        Boxes are ordered by:
        1. Vertical position (top to bottom)
        2. Horizontal position (left to right) for boxes on the same row
        
        This ordering is natural for reading text in most languages.
        
        Args:
            other: Another Bbox to compare with
            
        Returns:
            True if this box should come before the other in sorted order
        """
        if not self.same_row(other):
            return self.p.y < other.p.y
        else:
            return self.p.x < other.p.x

    def __repr__(self) -> str:
        return f"Bbox(upper_left_point={self.p}, h={self.h}, w={self.w}), label={self.label}, confident={self.confidence}, content={self.content})"
