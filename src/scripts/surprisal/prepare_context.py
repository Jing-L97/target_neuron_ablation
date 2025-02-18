#!/usr/bin/env python3
"""Script to select prior contexts for given words from The Pile."""

import typing as t
from pathlib import Path
import json
import random
from datasets import load_dataset
from collections import defaultdict, Counter
from neuron_analyzer import settings

def load_target_words(file_path: Path) -> list[str]:
    """Load target words from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def extract_context(
    text: str,
    target_word: str,
    context_size: int
) -> list[str]:
    """Extract context before target word."""
    words = text.split()
    contexts = []
    
    for i, word in enumerate(words):
        if word == target_word and i >= context_size:
            context = words[i-context_size:i]
            contexts.append(context)
    
    return contexts

class ContextSelector:
    def __init__(
        self,
        target_words: list[str],
        context_size: int,
        min_context_freq: int
    ):
        """
        Initialize context selector.
        
        Args:
            target_words: List of words to find contexts for
            context_size: Number of words in prior context
            min_context_freq: Minimum frequency threshold for contexts
        """
        self.target_words = target_words
        self.context_size = context_size
        self.min_context_freq = min_context_freq
        
        # Store contexts and their frequencies
        self.contexts: dict[str, list[list[str]]] = defaultdict(list)
        self.context_counts: dict[str, Counter] = defaultdict(Counter)

    def process_text(self, text: str) -> None:
        """Process text to find contexts for target words."""
        for word in self.target_words:
            new_contexts = extract_context(text, word, self.context_size)
            for context in new_contexts:
                context_key = ' '.join(context)
                self.contexts[word].append(context)
                self.context_counts[word][context_key] += 1

    def select_contexts(
        self,
        n_contexts: int
    ) -> dict[str, list[list[str]]]:
        """
        Select n random contexts for each word above frequency threshold.
        
        Args:
            n_contexts: Number of contexts to select per word
            
        Returns:
            Dictionary mapping words to lists of selected contexts
        """
        selected_contexts: dict[str, list[list[str]]] = {}
        
        for word in self.target_words:
            # Filter contexts by frequency
            valid_contexts = [
                context for context in self.contexts[word]
                if self.context_counts[word][' '.join(context)] >= self.min_context_freq
            ]
            
            # Randomly select contexts
            if valid_contexts:
                n_select = min(n_contexts, len(valid_contexts))
                selected_contexts[word] = random.sample(valid_contexts, n_select)
            else:
                selected_contexts[word] = []
                
        return selected_contexts

def save_contexts(
    contexts: dict[str, list[list[str]]],
    output_path: Path
) -> None:
    """Save selected contexts to file."""
    # Convert lists to strings for better readability
    output_data = {
        word: [' '.join(context) for context in word_contexts]
        for word, word_contexts in contexts.items()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

def main(
    words_file: Path,
    n_contexts: int = 5,
    context_size: int = 5,
    min_context_freq: int = 2,
    output_path: Path | None = None,
    split: str = "train"
) -> dict[str, list[list[str]]]:
    """
    Main function to select contexts for target words.
    
    Args:
        words_file: File containing target words
        n_contexts: Number of contexts to select per word
        context_size: Size of prior context
        min_context_freq: Minimum frequency for contexts
        output_path: Optional path to save results
        split: Dataset split to use
        
    Returns:
        Dictionary mapping words to selected contexts
    """
    # Load target words
    target_words = load_target_words(words_file)
    
    # Initialize context selector
    selector = ContextSelector(
        target_words,
        context_size,
        min_context_freq
    )
    
    # Load and process The Pile
    dataset = load_dataset("stas/c4-en-10k", split=split, streaming=True)
    
    for item in dataset:
        selector.process_text(item['text'])
    
    # Select contexts
    selected_contexts = selector.select_contexts(n_contexts)
    
    # Save if output path provided
    if output_path:
        save_contexts(selected_contexts, output_path)
    
    return selected_contexts

if __name__ == "__main__":
    # Example usage
    words_file = settings.PATH.dataset_root/"matched"/"target_words.txt"
    output_path = settings.PATH.dataset_root/"matched"/"selected_contexts.json"
    
    contexts = main(
        words_file=words_file,
        n_contexts=5,
        context_size=5,
        min_context_freq=2,
        output_path=output_path
    )
    
    # Print example contexts
    for word, word_contexts in list(contexts.items())[:3]:
        print(f"\nContexts for '{word}':")
        for context in word_contexts:
            print(f"  - {' '.join(context)}")