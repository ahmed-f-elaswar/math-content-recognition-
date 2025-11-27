"""Constants used throughout the TexTeller package.

This module defines model parameters, image processing settings,
and URLs for downloading pre-trained models.
"""

# Formula image (grayscale) normalization parameters
IMAGE_MEAN: float = 0.9545467  # Mean value for grayscale formula images
IMAGE_STD: float = 0.15394445  # Standard deviation for grayscale formula images

# Model architecture parameters
VOCAB_SIZE: int = 15000  # Size of the LaTeX vocabulary
FIXED_IMG_SIZE: int = 448  # Fixed input image size (height and width)
IMG_CHANNELS: int = 1  # Number of image channels (1 for grayscale)
MAX_TOKEN_SIZE: int = 1024  # Maximum number of tokens in generated LaTeX

# Training data augmentation parameters
MAX_RESIZE_RATIO: float = 1.15  # Maximum scaling ratio for random resizing
MIN_RESIZE_RATIO: float = 0.75  # Minimum scaling ratio for random resizing

# Image validation thresholds
MIN_HEIGHT: int = 12  # Minimum acceptable image height in pixels
MIN_WIDTH: int = 30  # Minimum acceptable image width in pixels

# Pre-trained model URLs
LATEX_DET_MODEL_URL: str = (
    "https://huggingface.co/TonyLee1256/texteller_det/resolve/main/rtdetr_r50vd_6x_coco.onnx"
)
TEXT_REC_MODEL_URL: str = (
    "https://huggingface.co/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_server_rec.onnx"
)
TEXT_DET_MODEL_URL: str = (
    "https://huggingface.co/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_det.onnx"
)
