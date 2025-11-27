"""CLI command for formula inference from images and PDFs.

This module provides the command-line interface for performing OCR and LaTeX
recognition on images and PDF files using TexTeller models.

Features:
    - Process individual images for formula or paragraph recognition
    - Process PDF files with automatic page extraction and text combination
    - Support for custom model and tokenizer paths
    - Configurable output format (LaTeX or KaTeX)
    - Optional style preservation (bold, italic, etc.)
    - Beam search support for improved accuracy
    - Save output to file or display in console

Examples:
    Recognize a formula from an image::
    
        $ texteller inference equation.png
        Predicted LaTeX: ```
        \\frac{1}{2}mv^2
        ```
    
    Process with custom output format::
    
        $ texteller inference image.jpg --output-format latex
    
    Save output to file::
    
        $ texteller inference image.png --output-file result.txt
    
    Process PDF and save as markdown::
    
        $ texteller inference paper.pdf --output-file output.md
    
    Use beam search for better accuracy::
    
        $ texteller inference complex.png --num-beams 5
"""

import click
from pathlib import Path
from texteller.api import (
    img2latex,
    pdf2md,
    load_model,
    load_tokenizer,
    load_latexdet_model,
    load_textdet_model,
    load_textrec_model,
)
from texteller.utils import get_device

@click.command()
@click.argument("file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option(
	"--model-path",
	type=click.Path(exists=True, file_okay=False, dir_okay=True),
	default=None,
	help="Path to the model dir path, if not provided, will use model from huggingface repo",
)
@click.option(
	"--tokenizer-path",
	type=click.Path(exists=True, file_okay=False, dir_okay=True),
	default=None,
	help="Path to the tokenizer dir path, if not provided, will use tokenizer from huggingface repo",
)
@click.option(
	"--output-format",
	type=click.Choice(["latex", "katex"]),
	default="katex",
	help="Output format, either latex or katex",
)
@click.option(
	"--keep-style",
	is_flag=True,
	default=False,
	help="Whether to keep the style of the LaTeX (e.g. bold, italic, etc.)",
)
@click.option(
	"--output-file",
	type=click.Path(file_okay=True, dir_okay=False),
	default=None,
	help="Output file path (for PDF processing, saves as .md file)",
)
@click.option(
	"--num-beams",
	type=int,
	default=1,
	help="Number of beams for beam search",
)
def inference(file_path, model_path, tokenizer_path, output_format, keep_style, output_file, num_beams):
	"""Perform OCR and LaTeX recognition on images or PDF files.
	
	This command processes the input file and outputs the recognized LaTeX or markdown
	content. For images, it performs direct recognition. For PDFs, it extracts images
	from each page, performs recognition, and combines results with text content.
	
	Args:
		file_path (str): Path to input file (image or PDF). Supported formats:
			Images: .jpg, .png, .jpeg, .bmp
			Documents: .pdf
		model_path (str, optional): Path to custom model directory. If not provided,
			downloads from HuggingFace repository.
		tokenizer_path (str, optional): Path to custom tokenizer directory. If not
			provided, downloads from HuggingFace repository.
		output_format (str): Output format - 'latex' or 'katex'. Defaults to 'katex'.
		keep_style (bool): Whether to preserve LaTeX styling (bold, italic, etc.).
			Defaults to False.
		output_file (str, optional): Path to save output file. For PDFs, saves as
			.md file. If not provided, prints to console.
		num_beams (int): Number of beams for beam search. Higher values improve
			accuracy but increase computation time. Defaults to 1.
	
	Examples:
		Basic image recognition::
		
			$ texteller inference equation.png
		
		PDF processing with beam search::
		
			$ texteller inference paper.pdf --num-beams 3 --output-file output.md
		
		Use custom model::
		
			$ texteller inference image.jpg --model-path ./my_model
	"""
	file_ext = Path(file_path).suffix.lower()
	
	if file_ext == ".pdf":
		# PDF processing
		click.echo("Processing PDF file...")
		model = load_model(model_dir=model_path)
		tokenizer = load_tokenizer(tokenizer_dir=tokenizer_path)
		latexdet_model = load_latexdet_model()
		textdet_model = load_textdet_model()
		textrec_model = load_textrec_model()
		
		result = pdf2md(
			pdf_path=file_path,
			latexdet_model=latexdet_model,
			textdet_model=textdet_model,
			textrec_model=textrec_model,
			latexrec_model=model,
			tokenizer=tokenizer,
			device=get_device(),
			num_beams=num_beams,
		)
		
		if output_file:
			with open(output_file, "w", encoding="utf-8") as f:
				f.write(result)
			click.echo(f"Markdown output saved to: {output_file}")
		else:
			click.echo("=== PDF Content (Markdown) ===")
			click.echo(result)
	else:
		# Image processing (existing functionality)
		model = load_model(model_dir=model_path)
		tknz = load_tokenizer(tokenizer_dir=tokenizer_path)
		pred = img2latex(
			model=model,
			tokenizer=tknz,
			images=[file_path],
			out_format=output_format,
			keep_style=keep_style,
			num_beams=num_beams,
		)[0]
		
		if output_file:
			with open(output_file, "w", encoding="utf-8") as f:
				f.write(pred)
			click.echo(f"Output saved to: {output_file}")
		else:
			click.echo(f"Predicted LaTeX: ```\n{pred}\n```")
