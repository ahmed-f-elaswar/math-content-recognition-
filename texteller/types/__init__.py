"""Type definitions for TexTeller.

This module defines type aliases and exports core types used throughout
the TexTeller package.

Types:
    TexTellerModel: Union type for TexTeller model instances
                   (PyTorch or ONNX Runtime)
    Bbox: Bounding box class for detected regions
"""

from typing import TypeAlias, Union

from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import VisionEncoderDecoderModel

from .bbox import Bbox

# Type alias for TexTeller model - can be either PyTorch or ONNX version
TexTellerModel: TypeAlias = Union[VisionEncoderDecoderModel, ORTModelForVision2Seq]


__all__ = ["Bbox", "TexTellerModel"]
