"""Device detection and management utilities for PyTorch."""

from typing import Literal

import torch


def str2device(device_str: Literal["cpu", "cuda", "mps"]) -> torch.device:
    """Convert a device string to a torch.device object.
    
    Args:
        device_str: String specifying the device ('cpu', 'cuda', or 'mps')
        
    Returns:
        torch.device object for the specified device
        
    Raises:
        ValueError: If device_str is not a valid device type
    """
    if device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "cuda":
        return torch.device("cuda")
    elif device_str == "mps":
        return torch.device("mps")
    else:
        raise ValueError(f"Invalid device: {device_str}")


def get_device(device_index: int = None) -> torch.device:
    """Automatically detect the best available device for inference.
    
    Checks for available devices in order of preference:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)

    Args:
        device_index: Optional GPU device index if multiple are available.
                     Currently not used; always selects first available GPU.

    Returns:
        torch.device: Best available device (cuda, mps, or cpu)
        
    Example:
        >>> device = get_device()
        >>> print(device)
        device(type='cuda', index=0)
    """
    if cuda_available():
        return str2device("cuda")
    elif mps_available():
        return str2device("mps")
    else:
        return str2device("cpu")


def cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available.
    
    Returns:
        True if CUDA is available and PyTorch was built with CUDA support
    """
    return torch.cuda.is_available()


def mps_available() -> bool:
    """Check if MPS (Apple Silicon GPU) is available.
    
    Returns:
        True if running on Apple Silicon with MPS backend enabled
    """
    return torch.backends.mps.is_available()
