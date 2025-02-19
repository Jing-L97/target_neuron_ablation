#!/usr/bin/env python3
"""Script to collect n-gram contexts and their frequencies from a corpus."""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from neuron_analyzer import settings
from neuron_analyzer.preprocess import NGramContextCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    parser.add_argument("-m", "--mode", type=str, default=5, help="topk")
    parser.add_argument("--min_log_freq", type=float, default=1, help="min_freq of the context")
    parser.add_argument("--max_log_freq", type=float, default=10, help="max_freq of the context")
    return parser.parse_args()



def load_target_words(file_path: Path) -> list[str]:
    """Load target words from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def save_data(data: dict, stats_path: Path) -> None:
    """Save both complete n-gram stats and selected contexts."""
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_ngram_stats(stats_path: Path) -> dict:
    """Load n-gram statistics from file."""
    with open(stats_path, 'r', encoding='utf-8') as f:
        return json.load(f)





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
        self.context_stats = defaultdict(lambda: ContextStats(0, 0.0, float("-inf"), Counter()))
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
        # ... (same as before)

    def get_all_ngram_stats(self):
        # ... (same as before)

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
    def filter_contexts(ngram_stats, target_words, n_contexts, min_context_freq, max_context_freq, mode):
        # ... (same as before)






def main() -> tuple[dict, dict]:
    args = parse_args()

    words_file = settings.PATH.dataset_root / args.words_file
    output_dir = settings.PATH.dataset_root / args.output_path / args.dataset / str(args.window_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    ngram_stats_file = output_dir / "ngram_stats.json"
    selected_data_file = output_dir / f"{words_file.stem}.json"

    target_words = load_target_words(words_file)
    logger.info(f"Loaded {len(target_words)} target words")

    collector = NGramContextCollector(window_size=args.window_size)

    if ngram_stats_file.is_file():
        logger.info(f"Loading pre-computed n-gram statistics from {ngram_stats_file}")
        ngram_stats = load_ngram_stats(ngram_stats_file)
    else:
        logger.info("Computing n-gram statistics...")
        dataset = load_dataset(args.dataset, split=args.split)

        for i, item in enumerate(dataset):
            collector.process_text(item['text'])

        logger.info("Computing frequencies...")
        collector.compute_frequencies()

        logger.info("Collecting n-gram statistics...")
        ngram_stats = collector.get_all_ngram_stats()
        logger.info(f"Saving n-gram statistics to {ngram_stats_file}")
        save_data(ngram_stats, ngram_stats_file)


    logger.info("Selecting contexts...")
    selected_data = collector.filter_contexts(
        ngram_stats, 
        target_words, 
        args.n_contexts,
        args.min_log_freq, 
        args.max_log_freq,
        args.mode
        ) 
    logger.info(f"Saving selected data to {selected_data_file}")
    save_data(selected_data, selected_data_file)
    
    

if __name__ == "__main__":
    main()