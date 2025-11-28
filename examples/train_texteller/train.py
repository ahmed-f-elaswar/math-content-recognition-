"""Training script for TexTeller model.

This script demonstrates how to fine-tune or train a TexTeller model from scratch
using the HuggingFace Transformers library. It includes data loading, preprocessing,
and training configuration.

Features:
    - Load datasets in imagefolder format
    - Filter images by minimum dimensions
    - Tokenize LaTeX formulas
    - Split data into train/eval sets
    - Apply data augmentation for training
    - Train with HuggingFace Trainer

Dataset Format:
    The dataset should be organized in imagefolder format::
    
        dataset/
            train/
                metadata.jsonl  # Contains image paths and LaTeX labels
                image1.jpg
                image2.png
                ...

Prerequisites:
    - Install training dependencies: pip install texteller[train]
    - Prepare your dataset in the correct format
    - Update train_config.yaml with your training parameters

Usage:
    1. Prepare your dataset in examples/train_texteller/dataset/
    2. Configure training parameters in train_config.yaml
    3. Run the training script::
    
        $ python train.py

Configuration:
    Training parameters can be set in train_config.yaml or modified in the
    training_config variable. Key parameters include:
    - output_dir: Where to save model checkpoints
    - num_train_epochs: Number of training epochs
    - per_device_train_batch_size: Batch size per device
    - learning_rate: Learning rate for optimizer
    - save_steps: How often to save checkpoints

Examples:
    Train from scratch::
    
        model = load_model()
        enable_train = True
    
    Train from pre-trained checkpoint::
    
        model = load_model("/path/to/checkpoint")
        enable_train = True
    
    Use custom tokenizer::
    
        tokenizer = load_tokenizer("/path/to/tokenizer")

Notes:
    - The script filters images smaller than MIN_HEIGHT x MIN_WIDTH
    - Data is split 90/10 for train/eval
    - Training data uses augmentation, eval data does not
    - Set enable_train = False to skip training (for testing)
"""

from functools import partial

import yaml
from datasets import load_dataset
from transformers import (
	Trainer,
	TrainingArguments,
)

from texteller import load_model, load_tokenizer
from texteller.constants import MIN_HEIGHT, MIN_WIDTH

from examples.train_texteller.utils import (
	collate_fn,
	filter_fn,
	img_inf_transform,
	img_train_transform,
	tokenize_fn,
)


def train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer, training_config):
	"""Train the TexTeller model using HuggingFace Trainer.
	
	Sets up training arguments and trains the model on the provided datasets.
	After training, saves the final model and tokenizer.
	
	Args:
		model: The TexTeller model to train.
		tokenizer: The tokenizer for processing LaTeX strings.
		train_dataset: Training dataset with preprocessed images and labels.
		eval_dataset: Evaluation dataset for validation during training.
		collate_fn_with_tokenizer: Data collation function with tokenizer bound.
		training_config: Dictionary with training configuration parameters.
	
	Returns:
		trainer: The HuggingFace Trainer instance after training.
	
	Examples:
		>>> model = load_model()
		>>> tokenizer = load_tokenizer()
		>>> trainer = train(model, tokenizer, train_ds, eval_ds, collate_fn, config)
	
	Notes:
		- Training arguments are loaded from training_config
		- Model checkpoints are saved according to TrainingArguments.save_steps
		- Final model is saved to output_dir/final_model/
	"""
	training_args = TrainingArguments(**training_config)
	trainer = Trainer(
		model,
		training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		data_collator=collate_fn_with_tokenizer,
	)

	trainer.train(resume_from_checkpoint=None)
	
	# Save final model and tokenizer
	final_model_path = f"{training_args.output_dir}/final_model"
	print(f"\nSaving final model to: {final_model_path}")
	trainer.save_model(final_model_path)
	tokenizer.save_pretrained(final_model_path)
	print(f"✓ Model and tokenizer saved successfully!")
	
	return trainer


if __name__ == "__main__":
	import os
	from pathlib import Path
	
	# Get script directory
	script_dir = Path(__file__).parent
	
	# Load training configuration from script directory
	config_file = script_dir / "train_config.yaml"
	with open(config_file, 'r') as f:
		training_config = yaml.safe_load(f)
	
	# Load dataset from script directory subfolder
	dataset_path = script_dir / "dataset"
	print(f"Loading dataset from: {dataset_path}")
	
	# Load and prepare dataset
	dataset = load_dataset("imagefolder", data_dir=str(dataset_path))["train"]
	dataset = dataset.filter(
		lambda x: x["image"].height > MIN_HEIGHT and x["image"].width > MIN_WIDTH
	)
	dataset = dataset.shuffle(seed=42)
	dataset = dataset.flatten_indices()

	tokenizer = load_tokenizer()
	filter_fn_with_tokenizer = partial(filter_fn, tokenizer=tokenizer)
	dataset = dataset.filter(filter_fn_with_tokenizer, num_proc=8)

	map_fn = partial(tokenize_fn, tokenizer=tokenizer)
	tokenized_dataset = dataset.map(
		map_fn, batched=True, remove_columns=dataset.column_names, num_proc=8
	)

	# Split dataset into train and eval, ratio 9:1
	split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
	train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
	train_dataset = train_dataset.with_transform(img_train_transform)
	eval_dataset = eval_dataset.with_transform(img_inf_transform)
	collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

	# Load pretrained model (auto-downloads from HuggingFace if not cached)
	model = load_model()
	
	# Start training
	trainer = train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer, training_config)
	
	print(f"\n{'='*50}")
	print("Training Summary:")
	print(f"  Output directory: {training_config['output_dir']}")
	print(f"  Final model: {training_config['output_dir']}/final_model")
	print(f"  Checkpoints: {training_config['output_dir']}/checkpoint-*")
	print(f"{'='*50}")
