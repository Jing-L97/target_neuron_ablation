#!/usr/bin/env python3
"""Script to collect n-gram contexts and their frequencies from a corpus."""
import argparse
import json
import random
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from neuron_analyzer import settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="collect n-gram contexts from a corpus.")
    parser.add_argument(
        "-w","--words_file",type=Path, default="matched/cdi_childes.csv",
        help="Relative path to the target words"
        )
    parser.add_argument(
        "-o","--output_path",type=Path, default="context",
        help="Relative path to the extracted context"
        )
    parser.add_argument("-d","--dataset", type=str,default="stas/c4-en-10k", help="dataset name")
    parser.add_argument("--split", type=str,default="train", help="dataset split")
    parser.add_argument("-s","--window_size", type=int,default=5, help="context window size")
    parser.add_argument("-n","--n_contexts", type=int,default=5, help="context numbers")
    parser.add_argument("--min_log_freq", type=float,default=0.00001, help="min_freq of the context")
    parser.add_argument("--max_log_freq", type=float,default=0.1, help="min_freq of the context")
    return parser.parse_args()

@dataclass
class ContextStats:
    """Statistics for a context."""
    count: int
    frequency: float
    subsequent_words: Counter


class NGramContextCollector:
    def __init__(
        self,
        window_size: int,
        min_context_freq: float = 0.01,
        max_context_freq: float = 0.1
    ):
        """
        Initialize n-gram context collector.
        
        Args:
            window_size: Size of context window
            min_context_freq: Minimum frequency threshold for contexts
            max_context_freq: Maximum frequency threshold for contexts
        """
        self.window_size = window_size
        self.min_context_freq = min_context_freq
        self.max_context_freq = max_context_freq
        
        # Store context statistics
        self.context_stats: defaultdict[str, ContextStats] = defaultdict(
            lambda: ContextStats(0, 0.0, Counter())
        )
        self.word_counts = Counter()
        self.total_windows = 0
        
    def process_text(self, text: str) -> None:
        """Process text to collect n-gram contexts."""
        # Split text into sentences (simple split by period)
        sentences = text.strip().split('.')
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) <= self.window_size:
                continue
                
            # Process each window in the sentence
            for i in range(len(words) - self.window_size):
                context = tuple(words[i:i + self.window_size])
                next_word = words[i + self.window_size]
                
                # Update statistics
                context_key = ' '.join(context)
                self.context_stats[context_key].count += 1
                self.context_stats[context_key].subsequent_words[next_word] += 1
                self.word_counts[next_word] += 1
                self.total_windows += 1
    
    def compute_frequencies(self) -> None:
        """Compute frequencies for contexts and words."""
        for context_key, stats in self.context_stats.items():
            stats.frequency = stats.count / self.total_windows
    
    def select_contexts(
        self,
        target_words: list[str],
        n_contexts: int
    ) -> dict[str, list[dict[str, t.Any]]]:
        """
        Select contexts for target words based on frequency criteria.
        
        Args:
            target_words: List of target words to find contexts for
            n_contexts: Number of contexts to select per word
            
        Returns:
            Dictionary with selected contexts and their statistics
        """
        selected_contexts: dict[str, list[dict[str, t.Any]]] = {}
        
        for word in target_words:
            # Find contexts where word appears as subsequent word
            word_contexts = [
                {
                    'context': context_key,
                    'count': stats.count,
                    'frequency': stats.frequency,
                    'word_count': stats.subsequent_words[word]
                }
                for context_key, stats in self.context_stats.items()
                if (word in stats.subsequent_words 
                    and self.min_context_freq <= stats.frequency <= self.max_context_freq)
            ]
            
            # Sort by frequency and select random subset
            if word_contexts:
                sorted_contexts = sorted(
                    word_contexts,
                    key=lambda x: x['word_count'],
                    reverse=True
                )
                n_select = min(n_contexts, len(sorted_contexts))
                selected = random.sample(sorted_contexts, n_select)
                
                # Add word statistics
                selected_contexts[word] = {
                    'contexts': selected,
                    'word_stats': {
                        'total_count': self.word_counts[word],
                        'frequency': self.word_counts[word] / sum(self.word_counts.values())
                    }
                }
            else:
                selected_contexts[word] = {
                    'contexts': [],
                    'word_stats': {
                        'total_count': self.word_counts[word],
                        'frequency': self.word_counts[word] / sum(self.word_counts.values())
                    }
                }
                
        return selected_contexts

def load_target_words(file_path: Path) -> list[str]:
    """Load target words from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def save_contexts(
    contexts: dict[str, list[dict[str, t.Any]]],
    output_path: Path
) -> None:
    """Save contexts and statistics to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, indent=2)

def main() :
    """
    Main function to collect and select n-gram contexts.
    
    Args:
        words_file: File containing target words
        window_size: Size of context window
        n_contexts: Number of contexts to select per word
        min_count: Minimum count for contexts
        min_freq: Minimum frequency threshold
        max_freq: Maximum frequency threshold
        output_path: Optional path to save results
        split: Dataset split to use
    """
    args = parse_args()

    words_file = settings.PATH.dataset_root/args.words_file
    output_path = settings.PATH.dataset_root/args.output_path/args.dataset/f"{args.window_size}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Load target words
    target_words = load_target_words(words_file)
    print(f"Loaded {len(target_words)} target words")
    
    # Initialize collector
    collector = NGramContextCollector(
        window_size=args.window_size,
        min_context_freq=args.min_freq,
        max_context_freq=args.max_freq
    )
    
    # Load and process dataset
    dataset = load_dataset(args.dataset, split=args.split)
    print("Processing texts...")
    
    for i, item in enumerate(dataset):
        collector.process_text(item['text'])
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} texts")
    
    # Compute frequencies
    print("Computing frequencies...")
    collector.compute_frequencies()
    
    # Select contexts
    print("Selecting contexts...")
    selected_contexts = collector.select_contexts(
        target_words,
        args.n_contexts,
    )
    
    # Save if output path provided
    save_contexts(selected_contexts,output_path)
    print(f"Saved contexts to {output_path}")
    
    return selected_contexts

if __name__ == "__main__":
    
    contexts = main()
    
    # Print summary
    print("\nContext Collection Summary:")
    for word, data in list(contexts.items())[:3]:
        print(f"\nWord: {word}")
        print(f"Total occurrences: {data['word_stats']['total_count']}")
        print(f"Word frequency: {data['word_stats']['frequency']:.6f}")
        print("Sample contexts:")
        for ctx in data['contexts'][:2]:
            print(f"  - {ctx['context']} (count: {ctx['count']}, freq: {ctx['frequency']:.6f})")