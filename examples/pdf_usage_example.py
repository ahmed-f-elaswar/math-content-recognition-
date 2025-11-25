"""Example usage of PDF processing functionality."""

from texteller.api import (
    load_model,
    load_tokenizer,
    load_latexdet_model,
    load_textdet_model,
    load_textrec_model,
    pdf2md,
)
from texteller.utils import get_device

# Load all required models
print("Loading models...")
latexrec_model = load_model()
tokenizer = load_tokenizer()
latexdet_model = load_latexdet_model()
textdet_model = load_textdet_model()
textrec_model = load_textrec_model()

# Process PDF
pdf_path = "path/to/your/document.pdf"
print(f"Processing PDF: {pdf_path}")

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
print("\n=== Preview ===")
print(markdown_result[:500] + "...")
