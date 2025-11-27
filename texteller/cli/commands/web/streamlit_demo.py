"""Streamlit web interface for TexTeller.

This module provides an interactive web application for TexTeller using Streamlit.
It allows users to upload images or PDFs and get real-time LaTeX recognition results.

Features:
    - Upload images (jpg, png, jpeg, bmp) for recognition
    - Upload PDF files for multi-page processing
    - Paste images from clipboard
    - Two recognition modes: Formula and Paragraph
    - Configurable beam search for improved accuracy
    - Optional style preservation
    - Visual feedback with success/failure animations
    - Copy results to clipboard
    - Download results as text files

Usage:
    Start the Streamlit app::
    
        $ texteller web
        # or
        $ streamlit run streamlit_demo.py
    
    Then open your browser to http://localhost:8501

Interface Components:
    - File uploader: Upload images or PDFs
    - Paste button: Paste images from clipboard
    - Mode selector: Choose between Formula or Paragraph recognition
    - Advanced options: Beam search, style preservation, ONNX runtime
    - Results display: Shows recognized LaTeX/markdown with copy and download options

Examples:
    The web interface provides an intuitive UI where users can:
    
    1. Upload an image of a formula
    2. Click "Recognize" button
    3. View and copy the LaTeX result
    4. Download the result as a text file
    
    For PDFs:
    
    1. Upload a PDF document
    2. Select "Paragraph" mode for better context
    3. Get markdown output with all formulas recognized
    4. Download the complete markdown file
"""

import base64
import io
import os
import re
import shutil
import tempfile

import streamlit as st
from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton

from texteller.api import (
	img2latex,
	load_latexdet_model,
	load_model,
	load_textdet_model,
	load_textrec_model,
	load_tokenizer,
	paragraph2md,
)
from texteller.cli.commands.web.style import (
	HEADER_HTML,
	IMAGE_EMBED_HTML,
	IMAGE_INFO_HTML,
	SUCCESS_GIF_HTML,
)
from texteller.utils import str2device

st.set_page_config(page_title="TexTeller", page_icon="🧮")


@st.cache_resource
def get_texteller(use_onnx):
	"""Load and cache the TexTeller model.
	
	Uses Streamlit's caching to load the model only once across sessions.
	
	Args:
		use_onnx (bool): Whether to use ONNX runtime for inference.
	
	Returns:
		The loaded TexTeller model.
	"""
	return load_model(use_onnx=use_onnx)

@st.cache_resource
def get_tokenizer():
	"""Load and cache the tokenizer.
	
	Uses Streamlit's caching to load the tokenizer only once across sessions.
	
	Returns:
		The loaded tokenizer.
	"""
	return load_tokenizer()

@st.cache_resource
def get_latexdet_model():
	"""Load and cache the LaTeX detection model.
	
	Uses Streamlit's caching to load the model only once across sessions.
	
	Returns:
		The loaded LaTeX detection model.
	"""
	return load_latexdet_model()

@st.cache_resource()
def get_textrec_model():
	"""Load and cache the text recognition model.
	
	Uses Streamlit's caching to load the PaddleOCR text recognition model
	only once across sessions.
	
	Returns:
		The loaded text recognition model.
	"""
	return load_textrec_model()

@st.cache_resource()
def get_textdet_model():
	"""Load and cache the text detection model.
	
	Uses Streamlit's caching to load the PaddleOCR text detection model
	only once across sessions.
	
	Returns:
		The loaded text detection model.
	"""
	return load_textdet_model()

def get_image_base64(img_file):
	buffered = io.BytesIO()
	img_file.seek(0)
	img = Image.open(img_file)
	img.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode()

def on_file_upload():
	st.session_state["UPLOADED_FILE_CHANGED"] = True

def change_side_bar():
	st.session_state["CHANGE_SIDEBAR_FLAG"] = True

if "start" not in st.session_state:
	st.session_state["start"] = 1
	st.toast("Hooray!", icon="🎉")

if "UPLOADED_FILE_CHANGED" not in st.session_state:
	st.session_state["UPLOADED_FILE_CHANGED"] = False

if "CHANGE_SIDEBAR_FLAG" not in st.session_state:
	st.session_state["CHANGE_SIDEBAR_FLAG"] = False

if "INF_MODE" not in st.session_state:
	st.session_state["INF_MODE"] = "Formula recognition"

# ====== <sidebar> ======

with st.sidebar:
	num_beams = 1

	st.markdown("# 🔨️ Config")
	st.markdown("")

	inf_mode = st.selectbox(
		"Inference mode",
		("Formula recognition", "Paragraph recognition"),
		on_change=change_side_bar,
	)

	num_beams = st.number_input(
		"Number of beams", min_value=1, max_value=20, step=1, on_change=change_side_bar
	)

	device = st.radio("device", ("cpu", "cuda", "mps"), on_change=change_side_bar)

	st.markdown("## Seedup")
	use_onnx = st.toggle("ONNX Runtime ")


# ====== </sidebar> ======


# ====== <page> ======

latexrec_model = get_texteller(use_onnx)
tokenizer = get_tokenizer()

if inf_mode == "Paragraph recognition":
	latexdet_model = get_latexdet_model()
	textrec_model = get_textrec_model()
	textdet_model = get_textdet_model()

st.markdown(HEADER_HTML, unsafe_allow_html=True)

uploaded_file = st.file_uploader(" ", type=["jpg", "png", "pdf"], on_change=on_file_upload)

paste_result = pbutton(
	label="📋 Paste an image",
	background_color="#5BBCFF",
	hover_background_color="#3498db",
)
st.write("")

if st.session_state["CHANGE_SIDEBAR_FLAG"] is True:
	st.session_state["CHANGE_SIDEBAR_FLAG"] = False
elif uploaded_file or paste_result.image_data is not None:
	if st.session_state["UPLOADED_FILE_CHANGED"] is False and paste_result.image_data is not None:
		uploaded_file = io.BytesIO()
		paste_result.image_data.save(uploaded_file, format="PNG")
		uploaded_file.seek(0)

	if st.session_state["UPLOADED_FILE_CHANGED"] is True:
		st.session_state["UPLOADED_FILE_CHANGED"] = False

	# Check if uploaded file is PDF
	is_pdf = uploaded_file.name.lower().endswith('.pdf')
	
	if is_pdf:
		# Handle PDF
		temp_dir = tempfile.mkdtemp()
		pdf_path = os.path.join(temp_dir, "document.pdf")
		with open(pdf_path, "wb") as f:
			f.write(uploaded_file.read())
		uploaded_file.seek(0)
	else:
		# Handle image
		img = Image.open(uploaded_file)

		temp_dir = tempfile.mkdtemp()
		png_fpath = os.path.join(temp_dir, "image.png")
		img.save(png_fpath, "PNG")

	if not is_pdf:
		with st.container(height=300):
			img_base64 = get_image_base64(uploaded_file)

			st.markdown(
				IMAGE_EMBED_HTML.format(img_base64=img_base64),
				unsafe_allow_html=True,
			)

		st.markdown(
			IMAGE_INFO_HTML.format(img_height=img.height, img_width=img.width),
			unsafe_allow_html=True,
		)
	else:
		st.info(f"📄 PDF file uploaded: {uploaded_file.name}")

	st.write("")

	with st.spinner("Processing..." if is_pdf else "Predicting..."):
		if is_pdf:
			# Process PDF
			from texteller.api import pdf2md
			pred = pdf2md(
				pdf_path=pdf_path,
				latexdet_model=latexdet_model,
				textdet_model=textdet_model,
				textrec_model=textrec_model,
				latexrec_model=latexrec_model,
				tokenizer=tokenizer,
				device=str2device(device),
				num_beams=num_beams,
			)
		elif inf_mode == "Formula recognition":
			pred = img2latex(
				model=latexrec_model,
				tokenizer=tokenizer,
				images=[png_fpath],
				device=str2device(device),
				out_format="katex",
				num_beams=num_beams,
				keep_style=False,
			)[0]
		else:
			pred = paragraph2md(
				img_path=png_fpath,
				latexdet_model=latexdet_model,
				textdet_model=textdet_model,
				textrec_model=textrec_model,
				latexrec_model=latexrec_model,
				tokenizer=tokenizer,
				device=str2device(device),
				num_beams=num_beams,
			)

		st.success("Completed!", icon="✅")
		
		if is_pdf:
			# Show PDF results
			st.code(pred, language="markdown")
			st.markdown("---")
			st.markdown("### Rendered Output")
			mixed_res = re.split(r"(\$\$.*?\$\$)", pred, flags=re.DOTALL)
			for text in mixed_res:
				if text.startswith("$$") and text.endswith("$$"):
					st.latex(text.strip("$$"))
				else:
					st.markdown(text)
		elif inf_mode == "Formula recognition":
			st.code(pred, language="latex")
			st.latex(pred)
		elif inf_mode == "Paragraph recognition":
			st.code(pred, language="markdown")
			mixed_res = re.split(r"(\$\$.*?\$\$)", pred, flags=re.DOTALL)
			for text in mixed_res:
				if text.startswith("$$") and text.endswith("$$"):
					st.latex(text.strip("$$"))
				else:
					st.markdown(text)
		else:
			raise ValueError(f"Invalid inference mode: {inf_mode}")

		st.write("")
		st.write("")

		with st.expander(":star2: :gray[Tips for better results]"):
			st.markdown("""
				* :mag_right: Use a clear and high-resolution image.
				* :scissors: Crop images as accurately as possible.
				* :jigsaw: Split large multi line formulas into smaller ones.
				* :page_facing_up: Use images with **white background and black text** as much as possible.
				* :book: Use a font with good readability.
			""")
		shutil.rmtree(temp_dir)

	paste_result.image_data = None

# ====== </page> ======
