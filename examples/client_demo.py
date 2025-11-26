"""Example client for testing TexTeller server with images and PDFs."""

import requests

server_url = "http://127.0.0.1:8000/predict"

# For image files
img_path = "/path/to/your/image.jpg"
with open(img_path, "rb") as img:
    files = {"img": img}
    response = requests.post(server_url, files=files)

print("=== Image Recognition Result ===")
print(response.text)

# For PDF files
pdf_path = "/path/to/your/document.pdf"
with open(pdf_path, "rb") as pdf_file:
    files = {"pdf": pdf_file}
    response = requests.post(server_url, files=files)

print("\n=== PDF Recognition Result ===")
print(response.text)
