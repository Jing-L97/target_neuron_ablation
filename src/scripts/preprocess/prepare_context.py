import argparse
import logging
from pathlib import Path

import pandas as pd

from neuron_analyzer import settings
from neuron_analyzer.load_util import load_json, save_json
from neuron_analyzer.preprocess.preprocess import NGramContextCollector, 

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="collect n-gram contexts from a corpus.")
    parser.add_argument(
        "-w",
        "--words_file",
        type=Path,
        default="matched/oxford-understand.csv",
        help="Relative path to the target words",
    )
    parser.add_argument(
        "-o", "--output_path", type=Path, default="context", help="Relative path to the extracted context"
    )
    parser.add_argument("-d", "--dataset", type=str, default="stas/c4-en-10k", help="dataset name")
    parser.add_argument("--split", type=str, default="train", help="dataset split")
    parser.add_argument("-s", "--window_size", type=int, default=5, help="min context window size")
    parser.add_argument("-n", "--n_contexts", type=int, default=20, help="context numbers")
    parser.add_argument("-m", "--mode", type=str, choices=["random", "topk"], default=5, help="topk")
    return parser.parse_args()


def load_target_words(file_path: Path) -> list[str]:
    if file_path.suffix == ".txt":
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f]
    if file_path.suffix == ".csv":
        data = pd.read_csv(file_path)
        print("csv file has been loded")
        return data["word"].to_list()


def main():
    args = parse_args()

    words_file = settings.PATH.dataset_root / args.words_file
    output_dir = settings.PATH.dataset_root / args.output_path / args.dataset / str(args.window_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    ngram_stats_file = output_dir / "ngram_stats.json"

    selected_contexts_file = output_dir / f"{Path(args.words_file).stem}.json"
    print(selected_contexts_file)
    target_words = load_target_words(words_file)
    logger.info(f"Loaded {len(target_words)} target words")

    if ngram_stats_file.exists():
        print(f"Loading existing n-gram statistics from {ngram_stats_file}")
        ngram_stats = load_json(ngram_stats_file)
    else:
        print("Computing n-gram statistics...")
        collector = NGramContextCollector()

        try:
            collector.collect_stats(args.dataset, args.split)
        except ValueError as e:
            print(f"Error processing dataset: {e}")
            return

        ngram_stats = collector.get_all_ngram_stats()
        save_json(ngram_stats, ngram_stats_file)
        print(f"Saved n-gram statistics to {ngram_stats_file}")

    target_words = load_target_words(words_file)

    selected_contexts = NGramContextCollector.filter_contexts(
        ngram_stats, target_words, args.n_contexts, args.mode, args.window_size
    )

    save_json(selected_contexts, selected_contexts_file)
    print(f"Saved selected contexts to {selected_contexts_file}")


if __name__ == "__main__":
    main()
