"""N-gram repetition detection for stopping text generation.

This module provides a stopping criterion that detects repeating n-grams
during text generation, which is useful for preventing infinite loops and
redundant output in LaTeX generation.

Classes:
    DetectRepeatingNgramCriteria: StoppingCriteria that halts generation
        when any n-gram repeats.

Examples:
    Basic usage with a transformer model::
    
        from texteller.api.criterias.ngram import DetectRepeatingNgramCriteria
        
        # Stop if any 3-gram repeats
        criteria = DetectRepeatingNgramCriteria(n=3)
        
        # Use in generation
        output = model.generate(
            input_ids,
            max_length=512,
            stopping_criteria=[criteria]
        )
    
    Prevent repetitive LaTeX patterns::
    
        # Stop if any 4-gram (e.g., "\\frac{1}{2}") repeats
        criteria = DetectRepeatingNgramCriteria(n=4)
        
        latex = model.generate(
            image_features,
            stopping_criteria=[criteria]
        )
"""

import torch
from transformers import StoppingCriteria


class DetectRepeatingNgramCriteria(StoppingCriteria):
    """Stops generation efficiently if any n-gram repeats.

    This criteria maintains a set of encountered n-grams and stops generation
    when a previously seen n-gram appears again. This prevents the model from
    generating repetitive or infinite sequences.
    
    The criterion checks only the latest n-gram at each generation step for
    efficiency. If the n-gram has been seen before, generation stops immediately.
    
    Attributes:
        n (int): The size of n-grams to check (e.g., 3 for trigrams).
        seen_ngrams (set): Set of tuples representing seen n-grams.
    
    Examples:
        Detect repeating trigrams::
        
            criteria = DetectRepeatingNgramCriteria(n=3)
            # If sequence generates: [1, 2, 3, 4, 5, 1, 2, 3]
            # Stops at the second occurrence of (1, 2, 3)
        
        Use with beam search::
        
            from transformers import GenerationConfig
            
            config = GenerationConfig(
                max_length=512,
                num_beams=5,
                stopping_criteria=[DetectRepeatingNgramCriteria(n=4)]
            )
            
            output = model.generate(inputs, generation_config=config)
    """

    def __init__(self, n: int):
        """Initialize the n-gram repetition detector.
        
        Args:
            n (int): The size of the n-gram to check for repetition. Must be
                positive. Common values are 3-5.
        
        Raises:
            ValueError: If n is not positive.
        
        Examples:
            >>> criteria = DetectRepeatingNgramCriteria(n=3)
            >>> print(criteria.n)
            3
        """
        if n <= 0:
            raise ValueError("n-gram size 'n' must be positive.")
        self.n = n
        # Stores tuples of token IDs representing seen n-grams
        self.seen_ngrams = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores.

        Return:
            `bool`: `True` if generation should stop, `False` otherwise.
        """
        batch_size, seq_length = input_ids.shape

        # Need at least n tokens to form the first n-gram
        if seq_length < self.n:
            return False

        # --- Efficient Check ---
        # Consider only the first sequence in the batch for simplicity
        if batch_size > 1:
            # If handling batch_size > 1, you'd need a list of sets, one per batch item.
            # Or decide on a stopping policy (e.g., stop if *any* sequence repeats).
            # For now, we'll focus on the first sequence.
            pass  # No warning needed every step, maybe once in __init__ if needed.

        sequence = input_ids[0]  # Get the first sequence

        # Get the latest n-gram (the one ending at the last token)
        last_ngram_tensor = sequence[-self.n :]
        # Convert to a hashable tuple for set storage and lookup
        last_ngram_tuple = tuple(last_ngram_tensor.tolist())

        # Check if this n-gram has been seen before *at any prior step*
        if last_ngram_tuple in self.seen_ngrams:
            return True  # Stop generation
        else:
            # It's a new n-gram, add it to the set and continue
            self.seen_ngrams.add(last_ngram_tuple)
            return False  # Continue generation
