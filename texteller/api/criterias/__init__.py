"""Stopping criteria for text generation in TexTeller.

This module provides custom stopping criteria for controlling the text generation
process during LaTeX recognition. These criteria can be used with HuggingFace
Transformers models to terminate generation based on specific conditions.

Classes:
    DetectRepeatingNgramCriteria: Stops generation when n-grams repeat.

Examples:
    Using the repeating n-gram criteria::
    
        from texteller.api.criterias import DetectRepeatingNgramCriteria
        from transformers import GenerationConfig
        
        # Stop if any 3-gram (trigram) repeats
        criteria = DetectRepeatingNgramCriteria(n=3)
        
        # Use with model.generate()
        output = model.generate(
            input_ids,
            generation_config=config,
            stopping_criteria=[criteria]
        )
"""

from .ngram import DetectRepeatingNgramCriteria


__all__ = ["DetectRepeatingNgramCriteria"]
