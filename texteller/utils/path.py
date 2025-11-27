"""File system path utilities.

Provides helper functions for common path operations like resolving,
creating, and removing files and directories.
"""

from pathlib import Path
from typing import Literal, Union
from texteller.logger import get_logger

_logger = get_logger(__name__)

def resolve_path(path: Union[str, Path]) -> str:
	"""Resolve a path to its absolute form, expanding user home directory.
	
	Args:
		path: Path as string or Path object
		
	Returns:
		Absolute path as string with ~ expanded and symlinks resolved
		
	Example:
		>>> resolve_path('~/data/model.onnx')
		'/home/user/data/model.onnx'
	"""
	if isinstance(path, str):
		path = Path(path)
	return str(path.expanduser().resolve())

def touch(path: Union[str, Path]) -> None:
	"""Create an empty file or update its timestamp.
	
	Args:
		path: Path to file
	"""
	if isinstance(path, str):
		path = Path(path)
	path.touch(exist_ok=True)

def mkdir(path: Union[str, Path]) -> None:
	"""Create a directory and any necessary parent directories.
	
	Equivalent to 'mkdir -p' in Unix. Does nothing if directory exists.
	
	Args:
		path: Path to directory to create
	"""
	if isinstance(path, str):
		path = Path(path)
	path.mkdir(parents=True, exist_ok=True)

def rmfile(path: Union[str, Path]) -> None:
	"""Remove a file.
	
	Args:
		path: Path to file to remove
		
	Raises:
		FileNotFoundError: If file does not exist
	"""
	if isinstance(path, str):
		path = Path(path)
	path.unlink(missing_ok=False)

def rmdir(path: Union[str, Path], mode: Literal["empty", "recursive"] = "empty") -> None:
	"""Remove a directory.
	
	Args:
		path: Path to directory to remove
		mode: Removal mode:
		     - 'empty': Only remove if directory is empty (default)
		     - 'recursive': Remove directory and all contents
		     
	Raises:
		OSError: If mode='empty' and directory is not empty
		ValueError: If mode is not 'empty' or 'recursive'
	"""
	if isinstance(path, str):
		path = Path(path)

	if mode == "empty":
		path.rmdir()
		_logger.info(f"Removed empty directory: {path}")
	elif mode == "recursive":
		import shutil
		shutil.rmtree(path)
		_logger.info(f"Recursively removed directory and all contents: {path}")
	else:
		raise ValueError(f"Invalid mode: {mode}. Must be 'empty' or 'all'")
