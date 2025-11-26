"""
Example usage of TexTeller for both image and PDF processing.

This demonstrates:
1. Direct Python API usage (processing files locally)
2. HTTP API Server usage (sending files to a running server)
"""

# ============================================
# Method 1: Direct Python API
# ============================================

from texteller.api import (
    load_model,
    load_tokenizer,
    load_latexdet_model,
    load_textdet_model,
    load_textrec_model,
    img2latex,
    paragraph2md,
    pdf2md,
)
from texteller.utils import get_device

print("=" * 50)
print("Method 1: Direct Python API")
print("=" * 50)

# Load all required models
print("\nLoading models...")
latexrec_model = load_model()
tokenizer = load_tokenizer()
latexdet_model = load_latexdet_model()
textdet_model = load_textdet_model()
textrec_model = load_textrec_model()

# Process an image
print("\n--- Processing Image ---")
img_path = "path/to/your/image.jpg"
latex_result = img2latex(
    img_path=img_path,
    model=latexrec_model,
    tokenizer=tokenizer,
    device=get_device(),
    num_beams=1,
)
print(f"LaTeX formula: {latex_result}")

# Process a PDF
print("\n--- Processing PDF ---")
pdf_path = "path/to/your/document.pdf"
markdown_result = pdf2md(
    pdf_path=pdf_path,
    latexdet_model=latexdet_model,
    textdet_model=textdet_model,
    textrec_model=textrec_model,
    latexrec_model=latexrec_model,
    tokenizer=tokenizer,
    device=get_device(),
    num_beams=1,
    dpi=300,  # Higher DPI = better quality, slower processing
)

# Save result
output_path = "output.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown_result)

print(f"Markdown saved to: {output_path}")
print(f"Preview: {markdown_result[:200]}...")


# ============================================
# Method 2: HTTP API Server
# ============================================

import requests

print("\n\n" + "=" * 50)
print("Method 2: HTTP API Server")
print("=" * 50)
print("\nFirst, start the server: texteller launch")
print("Then run this section:\n")

server_url = "http://127.0.0.1:8000/predict"

# Send image to server
print("--- Sending Image ---")
img_path = "/path/to/your/image.jpg"
try:
    with open(img_path, "rb") as img:
        files = {"img": img}
        response = requests.post(server_url, files=files)
    print(f"Result: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Send PDF to server
print("\n--- Sending PDF ---")
pdf_path = "/path/to/your/document.pdf"
try:
    with open(pdf_path, "rb") as pdf_file:
        files = {"pdf": pdf_file}
        response = requests.post(server_url, files=files)
    print(f"Result preview: {response.text[:200]}...")
except Exception as e:
    print(f"Error: {e}")
