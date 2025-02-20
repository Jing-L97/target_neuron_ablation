import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import spacy
from datasets import load_dataset

from neuron_analyzer import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ContextStats:
    """Statistics for a context."""
    count: int
    frequency: float
    log_freq_per_million: float
    subsequent_words: Counter

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
        self.nlp.add_pipe("sentencizer")

    def process_text(self, text: str) -> List[List[str]]:
        doc = self.nlp(text)
        return [sentence.text.strip().split() for sentence in doc.sents]

    def load_and_process_dataset(self, dataset_name: str, split: str, batch_size: int = 1000):
        dataset = load_dataset(dataset_name, split=split)
        if 'text' not in dataset.column_names:
            raise ValueError(f"'text' column not found in the dataset. Available columns: {dataset.column_names}")
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]['text']
            for text in batch:
                yield self.process_text(text)

class NGramStatisticsComputer:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.context_stats: Dict[str, ContextStats] = defaultdict(
            lambda: ContextStats(0, 0.0, float("-inf"), Counter())
        )
        self.word_counts = Counter()
        self.total_windows = 0

    def compute_stats(self, processed_sentences: List[List[str]]):
        for sentence in processed_sentences:
            if len(sentence) <= self.window_size:
                continue
            for i in range(len(sentence) - self.window_size):
                context = tuple(sentence[i : i + self.window_size])
                next_word = sentence[i + self.window_size]
                context_key = " ".join(context)
                self.context_stats[context_key].count += 1
                self.context_stats[context_key].subsequent_words[next_word] += 1
                self.word_counts[next_word] += 1
                self.total_windows += 1

    def compute_frequencies(self):
        total_words = sum(self.word_counts.values())
        for stats in self.context_stats.values():
            stats.frequency = stats.count / self.total_windows
            stats.log_freq_per_million = self.compute_log_freq_per_million(stats.count, self.total_windows)

    @staticmethod
    def compute_log_freq_per_million(count: int, total: int) -> float:
        freq_per_million = (count / total) * 1_000_000
        return math.log10(freq_per_million) if freq_per_million > 0 else float("-inf")

    def get_all_ngram_stats(self) -> Dict[str, Any]:
        total_words = sum(self.word_counts.values())
        return {
            "context_stats": {
                context: {
                    "count": stats.count,
                    "frequency": stats.frequency,
                    "log_freq_per_million": stats.log_freq_per_million,
                    "subsequent_words": {
                        word: {
                            "count": count,
                            "frequency": count / stats.count,
                            "log_freq_per_million": self.compute_log_freq_per_million(count, stats.count)
                        }
                        for word, count in stats.subsequent_words.items()
                    }
                }
                for context, stats in self.context_stats.items()
            },
            "word_stats": {
                word: {
                    "count": count,
                    "frequency": count / total_words,
                    "log_freq_per_million": self.compute_log_freq_per_million(count, total_words)
                }
                for word, count in self.word_counts.items()
            },
            "metadata": {
                "corpus_stats": {
                    "total_windows": self.total_windows,
                    "total_words": total_words,
                    "unique_contexts": len(self.context_stats),
                    "unique_words": len(self.word_counts),
                    "window_size": self.window_size
                }
            }
        }

class NGramContextCollector:
    def __init__(self, window_size: int):
        self.preprocessor = TextPreprocessor()
        self.computer = NGramStatisticsComputer(window_size)

    def collect_stats(self, dataset_name: str, split: str, batch_size: int = 1000):
        for processed_batch in self.preprocessor.load_and_process_dataset(dataset_name, split, batch_size):
            self.computer.compute_stats(processed_batch)
        self.computer.compute_frequencies()

    def get_all_ngram_stats(self):
        return self.computer.get_all_ngram_stats()

    @staticmethod
    def filter_contexts(
        ngram_stats: Dict[str, Any], 
        target_words: List[str], 
        n_contexts: int,
        mode: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        selected_data: Dict[str, List[Dict[str, Any]]] = {}

        for word in target_words:
            word_contexts = []
            for context_key, stats in ngram_stats["context_stats"].items():
                if (word in stats["subsequent_words"]):
                    word_count = stats["subsequent_words"][word]["count"]
                    word_freq = stats["subsequent_words"][word]["frequency"]
                    word_log_freq = stats["subsequent_words"][word]["log_freq_per_million"]
                    word_contexts.append({
                        "context": context_key,
                        "word_in_context_stats": {
                            "count": word_count,
                            "frequency": word_freq,
                            "log_freq_per_million": word_log_freq,
                        }
                    })

            sorted_contexts = sorted(word_contexts, key=lambda x: x["word_in_context_stats"]["count"], reverse=True)
            n_select = min(n_contexts, len(sorted_contexts))
            selected = random.sample(sorted_contexts, n_select) if mode == "random" else sorted_contexts[:n_contexts]
            selected_data[word] = selected

        return selected_data

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="collect n-gram contexts from a corpus.")
    parser.add_argument(
        "-w", "--words_file", type=Path, default="matched/cdi_childes.csv", help="Relative path to the target words"
    )
    parser.add_argument(
        "-o", "--output_path", type=Path, default="context", help="Relative path to the extracted context"
    )
    parser.add_argument("-d", "--dataset", type=str, default="stas/c4-en-10k", help="dataset name")
    parser.add_argument("--split", type=str, default="train", help="dataset split")
    parser.add_argument("-s", "--window_size", type=int, default=5, help="context window size")
    parser.add_argument("-n", "--n_contexts", type=int, default=5, help="context numbers")
    parser.add_argument("-m", "--mode", type=str, choices=["random", "topk"], default=5, help="topk")
    return parser.parse_args()



def load_target_words(file_path: Path) -> List[str]:
    if file_path.suffix==".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]
    if file_path.suffix==".csv":
        data = pd.read_csv(file_path)
        print("csv file has been loded")
        return data["word"].to_list()
        


def save_data(data: Dict[str, Any], file_path: Path) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_data(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()

    words_file = settings.PATH.dataset_root / args.words_file
    output_dir = settings.PATH.dataset_root / args.output_path / args.dataset / str(args.window_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    ngram_stats_file = output_dir / "ngram_stats.json"
    selected_contexts_file = output_dir / f"{words_file.stem}.json"

    target_words = load_target_words(words_file)
    logger.info(f"Loaded {len(target_words)} target words")


    if ngram_stats_file.exists():
        print(f"Loading existing n-gram statistics from {ngram_stats_file}")
        ngram_stats = load_data(ngram_stats_file)
    else:
        print("Computing n-gram statistics...")
        collector = NGramContextCollector(window_size=args.window_size)
        
        try:
            collector.collect_stats(args.dataset, args.split)
        except ValueError as e:
            print(f"Error processing dataset: {e}")
            return

        ngram_stats = collector.get_all_ngram_stats()
        save_data(ngram_stats, ngram_stats_file)
        print(f"Saved n-gram statistics to {ngram_stats_file}")

    target_words = load_target_words(words_file)
    
    selected_contexts = NGramContextCollector.filter_contexts(
        ngram_stats,
        target_words,
        args.n_contexts,
        args.mode
    )

    save_data(selected_contexts, selected_contexts_file)
    print(f"Saved selected contexts to {selected_contexts_file}")




if __name__ == "__main__":
    main()