"""Ray Serve deployment classes for TexTeller API server.

This module implements the REST API server for TexTeller using Ray Serve.
It provides a scalable deployment architecture with two main components:

- TexTellerServer: The prediction server that handles model inference
- Ingress: The HTTP handler that routes requests to the server

The server supports:
    - Image processing (formula and paragraph recognition)
    - PDF processing (multi-page with text extraction)
    - Multiple replicas for load balancing
    - GPU/CPU resource allocation
    - ONNX runtime optimization
    - Beam search for improved accuracy

API Usage:
    POST /predict with form data:
        - 'img': Image file (jpg, png, etc.)
        - 'pdf': PDF file
    
    Returns: Recognized LaTeX or markdown content as plain text

Examples:
    Using curl to process an image::
    
        $ curl -X POST -F "img=@equation.png" http://localhost:8000/predict
        \\frac{1}{2}mv^2
    
    Using curl to process a PDF::
    
        $ curl -X POST -F "pdf=@document.pdf" http://localhost:8000/predict
        # Page 1
        ...
    
    Using Python requests::
    
        import requests
        
        # Image processing
        with open("image.jpg", "rb") as f:
            response = requests.post(
                "http://localhost:8000/predict",
                files={"img": f}
            )
        print(response.text)
        
        # PDF processing
        with open("document.pdf", "rb") as f:
            response = requests.post(
                "http://localhost:8000/predict",
                files={"pdf": f}
            )
        print(response.text)
"""

import numpy as np
import cv2
import tempfile
from pathlib import Path
from starlette.requests import Request
from ray import serve
from ray.serve.handle import DeploymentHandle
from texteller.api import (
    load_model,
    load_tokenizer,
    img2latex,
    load_latexdet_model,
    load_textdet_model,
    load_textrec_model,
)
from texteller.utils import get_device, pdf2md
from texteller.globals import Globals
from typing import Literal

@serve.deployment(
	num_replicas=Globals().num_replicas,
	ray_actor_options={
		"num_cpus": Globals().ncpu_per_replica,
		"num_gpus": Globals().ngpu_per_replica * 1.0 / 2,
	},
)
class TexTellerServer:
	"""TexTeller prediction server deployment.
	
	This class implements the core prediction logic for the TexTeller API server.
	It handles both image and PDF processing requests, loading and managing all
	required models (LaTeX recognition, text detection, text recognition).
	
	The deployment is configured via Ray Serve with configurable resource allocation
	and replica count for scalability.
	
	Attributes:
		model: The main LaTeX recognition model.
		tokenizer: Tokenizer for the LaTeX recognition model.
		latexdet_model: Model for detecting LaTeX regions in images.
		textdet_model: PaddleOCR text detection model.
		textrec_model: PaddleOCR text recognition model.
		num_beams (int): Number of beams for beam search.
		out_format (str): Output format ('latex' or 'katex').
		keep_style (bool): Whether to preserve LaTeX styling.
	
	Methods:
		predict: Process an image and return LaTeX formula.
		predict_pdf: Process a PDF and return combined markdown content.
	"""
	def __init__(
		self,
		checkpoint_dir: str,
		tokenizer_dir: str,
		use_onnx: bool = False,
		out_format: Literal["latex", "katex"] = "katex",
		keep_style: bool = False,
		num_beams: int = 1,
	) -> None:
		"""Initialize the TexTeller server with models and configuration.
		
		Args:
			checkpoint_dir (str): Path to model checkpoint directory, or None to use
				default HuggingFace model.
			tokenizer_dir (str): Path to tokenizer directory, or None to use default.
			use_onnx (bool, optional): Whether to use ONNX runtime for inference.
				Defaults to False.
			out_format (Literal["latex", "katex"], optional): Output format for
				LaTeX formulas. Defaults to "katex".
			keep_style (bool, optional): Whether to preserve styling (bold, italic)
				in LaTeX output. Defaults to False.
			num_beams (int, optional): Number of beams for beam search during
				inference. Higher values improve accuracy. Defaults to 1.
		"""
		self.model = load_model(
			model_dir=checkpoint_dir,
			use_onnx=use_onnx,
		)
		self.tokenizer = load_tokenizer(tokenizer_dir=tokenizer_dir)
		self.latexdet_model = load_latexdet_model()
		self.textdet_model = load_textdet_model()
		self.textrec_model = load_textrec_model()
		self.num_beams = num_beams
		self.out_format = out_format
		self.keep_style = keep_style

		if not use_onnx:
			self.model = self.model.to(get_device())

	def predict(self, image_nparray: np.ndarray) -> str:
		"""Predict LaTeX formula from an image array.
		
		Args:
			image_nparray (np.ndarray): Image array in RGB format with shape (H, W, C).
		
		Returns:
			str: Recognized LaTeX string in the configured output format.
		
		Examples:
			>>> import numpy as np
			>>> server = TexTellerServer(None, None)
			>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
			>>> latex = server.predict(img)
		"""
		return img2latex(
			model=self.model,
			tokenizer=self.tokenizer,
			images=[image_nparray],
			device=get_device(),
			out_format=self.out_format,
			keep_style=self.keep_style,
			num_beams=self.num_beams,
		)[0]
	
	def predict_pdf(self, pdf_bytes: bytes) -> str:
		"""Process a PDF file and return combined markdown content.
		
		Extracts images from each page of the PDF, performs recognition on detected
		LaTeX regions and text regions, and combines the results with original text
		content in the proper reading order.
		
		Args:
			pdf_bytes (bytes): PDF file content as bytes.
		
		Returns:
			str: Markdown content with recognized formulas and text, organized by pages.
		
		Examples:
			>>> server = TexTellerServer(None, None)
			>>> with open("paper.pdf", "rb") as f:
			...     pdf_data = f.read()
			>>> markdown = server.predict_pdf(pdf_data)
			>>> print(markdown[:100])
			# Page 1
			...
		
		Notes:
			- Creates a temporary file to process the PDF
			- Automatically cleans up the temporary file after processing
			- Large PDFs may take significant time to process
		"""
		# Save PDF to temp file
		with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
			tmp.write(pdf_bytes)
			tmp_path = tmp.name
		
		try:
			result = pdf2md(
				pdf_path=tmp_path,
				latexdet_model=self.latexdet_model,
				textdet_model=self.textdet_model,
				textrec_model=self.textrec_model,
				latexrec_model=self.model,
				tokenizer=self.tokenizer,
				device=get_device(),
				num_beams=self.num_beams,
			)
			return result
		finally:
			# Clean up temp file
			Path(tmp_path).unlink(missing_ok=True)

@serve.deployment()
class Ingress:
	"""HTTP ingress for routing requests to TexTeller server.
	
	This class handles incoming HTTP requests at the /predict endpoint and routes
	them to the TexTellerServer deployment based on the file type (image or PDF).
	
	It performs request parsing, image decoding, and response formatting.
	
	Attributes:
		texteller_server (DeploymentHandle): Handle to the TexTellerServer deployment
			for making remote prediction requests.
	"""
	
	def __init__(self, rec_server: DeploymentHandle) -> None:
		"""Initialize the ingress with a TexTellerServer handle.
		
		Args:
			rec_server (DeploymentHandle): Handle to the deployed TexTellerServer instance.
		"""
		self.texteller_server = rec_server

	async def __call__(self, request: Request) -> str:
		"""Handle incoming HTTP prediction requests.
		
		This method processes POST requests to /predict, accepting either an image
		or a PDF file. It decodes the uploaded file, routes it to the appropriate
		prediction method, and returns the result.
		
		Args:
			request (Request): Starlette request object containing form data with
				either 'img' (image file) or 'pdf' (PDF file).
		
		Returns:
			str: Recognized LaTeX string for images, or markdown content for PDFs.
		
		Examples:
			Image request::
			
				POST /predict
				Content-Type: multipart/form-data
				
				img=<image_binary_data>
				
				Response: "\\frac{1}{2}mv^2"
			
			PDF request::
			
				POST /predict
				Content-Type: multipart/form-data
				
				pdf=<pdf_binary_data>
				
				Response: "# Page 1\n..."
		
		Notes:
			- Images are decoded from bytes and converted to RGB numpy arrays
			- PDFs are passed as raw bytes to the PDF processing pipeline
			- The method is async to handle concurrent requests efficiently
		"""
		form = await request.form()
		
		# Check if it's a PDF or image
		if "pdf" in form:
			# PDF processing
			pdf_bytes = await form["pdf"].read()
			pred = await self.texteller_server.predict_pdf.remote(pdf_bytes)
			return pred
		else:
			# Image processing
			img_rb = await form["img"].read()

			img_nparray = np.frombuffer(img_rb, np.uint8)
			img_nparray = cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)
			img_nparray = cv2.cvtColor(img_nparray, cv2.COLOR_BGR2RGB)

			pred = await self.texteller_server.predict.remote(img_nparray)
			return pred
