"""CLI command for formula inference from images and PDFs."""

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
	"""CLI command for formula inference from images and PDFs."""
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
