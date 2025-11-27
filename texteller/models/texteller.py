"""TexTeller model implementation.

Provides the TexTeller model class for LaTeX formula recognition from images.
Based on Vision Encoder-Decoder architecture from Hugging Face Transformers.
"""

from pathlib import Path
from typing import Optional

from transformers import RobertaTokenizerFast, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from texteller.constants import (
	FIXED_IMG_SIZE,
	IMG_CHANNELS,
	MAX_TOKEN_SIZE,
	VOCAB_SIZE,
)
from texteller.globals import Globals
from texteller.types import TexTellerModel
from texteller.utils import cuda_available


class TexTeller(VisionEncoderDecoderModel):
	"""TexTeller vision-to-LaTeX model.
	
	A specialized VisionEncoderDecoderModel for converting images of
	mathematical formulas to LaTeX code. The model uses:
	- Vision encoder: Processes grayscale images (448x448)
	- Text decoder: Generates LaTeX sequences (up to 1024 tokens)
	- Vocabulary: 15,000 LaTeX tokens
	
	Attributes:
		Inherits all attributes from VisionEncoderDecoderModel
	"""
	
	def __init__(self):
		"""Initialize a new TexTeller model with default configuration."""
		config = VisionEncoderDecoderConfig.from_pretrained(Globals().repo_name)
		config.encoder.image_size = FIXED_IMG_SIZE
		config.encoder.num_channels = IMG_CHANNELS
		config.decoder.vocab_size = VOCAB_SIZE
		config.decoder.max_position_embeddings = MAX_TOKEN_SIZE

		super().__init__(config=config)

	@classmethod
	def from_pretrained(cls, model_dir: Optional[str] = None, use_onnx: bool = False) -> TexTellerModel:
		"""Load a pre-trained TexTeller model.
		
		Args:
			model_dir: Directory containing model weights. If None, loads from
			          Hugging Face Hub (OleehyO/TexTeller)
			use_onnx: If True, load ONNX-optimized version for faster inference.
			         Requires 'optimum' package.
			         
		Returns:
			Loaded model instance (either PyTorch or ONNX Runtime)
			
		Example:
			>>> # Load default model
			>>> model = TexTeller.from_pretrained()
			>>> 
			>>> # Load ONNX version
			>>> model = TexTeller.from_pretrained(use_onnx=True)
			>>> 
			>>> # Load from local directory
			>>> model = TexTeller.from_pretrained('/path/to/model')
		"""
		if model_dir is None or model_dir == Globals().repo_name:
			if not use_onnx:
				return VisionEncoderDecoderModel.from_pretrained(Globals().repo_name)
			else:
				from optimum.onnxruntime import ORTModelForVision2Seq
				return ORTModelForVision2Seq.from_pretrained(
					Globals().repo_name,
					provider="CUDAExecutionProvider"
					if cuda_available()
					else "CPUExecutionProvider",
				)
		# Load from local directory
		model_dir = Path(model_dir).resolve()
		return VisionEncoderDecoderModel.from_pretrained(str(model_dir))

	@classmethod
	def get_tokenizer(cls, tokenizer_dir: Optional[str] = None) -> RobertaTokenizerFast:
		"""Load the tokenizer for TexTeller.
		
		Args:
			tokenizer_dir: Directory containing tokenizer files. If None, loads
			              from Hugging Face Hub (OleehyO/TexTeller)
			              
		Returns:
			RobertaTokenizerFast instance configured for LaTeX
			
		Example:
			>>> tokenizer = TexTeller.get_tokenizer()
			>>> tokens = tokenizer.encode('x^2 + y^2 = z^2')
		"""
		if tokenizer_dir is None or tokenizer_dir == Globals().repo_name:
			return RobertaTokenizerFast.from_pretrained(Globals().repo_name)
		tokenizer_dir = Path(tokenizer_dir).resolve()
		return RobertaTokenizerFast.from_pretrained(str(tokenizer_dir))
