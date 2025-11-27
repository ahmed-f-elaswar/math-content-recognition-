"""Example client for testing TexTeller server with images and PDFs.

This script demonstrates how to interact with the TexTeller API server using
Python's requests library. It shows examples of sending both image and PDF
files to the server for recognition.

Prerequisites:
    - TexTeller server must be running (start with: texteller launch)
    - requests library must be installed (pip install requests)

Usage:
    1. Start the TexTeller server in a separate terminal::
    
        $ texteller launch
    
    2. Update the file paths in this script to point to your files
    3. Run this script::
    
        $ python client_demo.py

Examples:
    The script demonstrates two types of requests:
    
    1. Image Recognition:
       - Sends an image file to the server
       - Receives LaTeX formula as response
    
    2. PDF Recognition:
       - Sends a PDF file to the server
       - Receives markdown content with recognized formulas

API Endpoint:
    POST http://127.0.0.1:8000/predict
        - For images: files={'img': image_file}
        - For PDFs: files={'pdf': pdf_file}
        - Returns: Plain text response with recognized content

Notes:
    - Update server_url if your server runs on a different host/port
    - Replace file paths with actual paths to your files
    - The server must be running before executing this script
"""

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
