"""CLI commands for launching TexTeller API server.

This module provides the command for starting a Ray Serve-based REST API server
for TexTeller. The server exposes a /predict endpoint for processing both images
and PDF files.

Server Features:
    - REST API endpoint for image and PDF processing
    - Configurable number of replicas for load balancing
    - GPU/CPU resource allocation per replica
    - Support for custom models and tokenizers
    - ONNX runtime support for optimized inference
    - Beam search configuration for improved accuracy

API Endpoint:
    POST /predict
        - For images: Upload with key 'img'
        - For PDFs: Upload with key 'pdf'
        - Returns: Recognized LaTeX or markdown content

Examples:
    Start server on default settings::
    
        $ texteller launch
        # Runs on http://0.0.0.0:8000/predict
    
    Start on custom port::
    
        $ texteller launch --port 9000
    
    Start with multiple replicas (load balancing)::
    
        $ texteller launch --num-replicas 4
    
    Configure resource allocation::
    
        $ texteller launch --ngpu-per-replica 0.5 --ncpu-per-replica 2
    
    Use custom model::
    
        $ texteller launch --checkpoint_dir ./my_model --tokenizer_dir ./my_tokenizer
    
    Enable ONNX runtime::
    
        $ texteller launch --use-onnx
"""

import sys
import time
import click
from ray import serve
from texteller.globals import Globals
from texteller.utils import get_device

@click.command()
@click.option(
	"-ckpt",
	"--checkpoint_dir",
	type=click.Path(exists=True, file_okay=False, dir_okay=True),
	default=None,
	help="Path to the checkpoint directory, if not provided, will use model from huggingface repo",
)
@click.option(
	"-tknz",
	"--tokenizer_dir",
	type=click.Path(exists=True, file_okay=False, dir_okay=True),
	default=None,
	help="Path to the tokenizer directory, if not provided, will use tokenizer from huggingface repo",
)
@click.option(
	"-p",
	"--port",
	type=int,
	default=8000,
	help="Port to run the server on",
)
@click.option(
	"--num-replicas",
	type=int,
	default=1,
	help="Number of replicas to run the server on",
)
@click.option(
	"--ncpu-per-replica",
	type=float,
	default=1.0,
	help="Number of CPUs per replica",
)
@click.option(
	"--ngpu-per-replica",
	type=float,
	default=1.0,
	help="Number of GPUs per replica",
)
@click.option(
	"--num-beams",
	type=int,
	default=1,
	help="Number of beams to use",
)
@click.option(
	"--use-onnx",
	is_flag=True,
	type=bool,
	default=False,
	help="Use ONNX runtime",
)
def launch(
	checkpoint_dir,
	tokenizer_dir,
	port,
	num_replicas,
	ncpu_per_replica,
	ngpu_per_replica,
	num_beams,
	use_onnx,
):
	"""Launch the TexTeller API server using Ray Serve.
	
	Starts a REST API server that can process both images and PDF files for OCR
	and LaTeX recognition. The server uses Ray Serve for scalable deployment with
	support for multiple replicas and resource allocation.
	
	Args:
		checkpoint_dir (str, optional): Path to custom model checkpoint directory.
			If not provided, downloads from HuggingFace repository.
		tokenizer_dir (str, optional): Path to custom tokenizer directory.
			If not provided, downloads from HuggingFace repository.
		port (int): Port number to run the server on. Defaults to 8000.
		num_replicas (int): Number of server replicas for load balancing.
			More replicas handle higher concurrent load but require more resources.
			Defaults to 1.
		ncpu_per_replica (float): Number of CPU cores allocated per replica.
			Can be fractional (e.g., 0.5 for half a core). Defaults to 1.0.
		ngpu_per_replica (float): Number of GPUs allocated per replica.
			Can be fractional for GPU sharing (e.g., 0.5). Requires CUDA device.
			Defaults to 1.0.
		num_beams (int): Number of beams for beam search during inference.
			Higher values improve accuracy but increase computation. Defaults to 1.
		use_onnx (bool): Whether to use ONNX runtime for optimized inference.
			Defaults to False.
	
	Raises:
		SystemExit: If ngpu_per_replica > 0 but CUDA is not available.
	
	Examples:
		Start basic server::
		
			$ texteller launch
		
		High-load configuration with 4 replicas::
		
			$ texteller launch --num-replicas 4 --ngpu-per-replica 0.25
		
		CPU-only deployment::
		
			$ texteller launch --ngpu-per-replica 0 --ncpu-per-replica 4
	
	Notes:
		- Server runs until interrupted (Ctrl+C)
		- Endpoint available at http://0.0.0.0:{port}/predict
		- Requires Ray Serve (install with: pip install texteller[train])
	"""
	device = get_device()
	if ngpu_per_replica > 0 and not device.type == "cuda":
		click.echo(
			click.style(
				f"Error: --ngpu-per-replica > 0 but detected device is {device.type}",
				fg="red",
			)
		)
		sys.exit(1)

	Globals().num_replicas = num_replicas
	Globals().ncpu_per_replica = ncpu_per_replica
	Globals().ngpu_per_replica = ngpu_per_replica
	from texteller.cli.commands.launch.server import Ingress, TexTellerServer

	serve.start(http_options={"host": "0.0.0.0", "port": port})
	rec_server = TexTellerServer.bind(
		checkpoint_dir=checkpoint_dir,
		tokenizer_dir=tokenizer_dir,
		use_onnx=use_onnx,
		num_beams=num_beams,
	)
	ingress = Ingress.bind(rec_server)

	serve.run(ingress, route_prefix="/predict")

	while True:
		time.sleep(1)
