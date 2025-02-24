import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from neuron_analyzer import settings
from neuron_analyzer.preprocess import NGramContextCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Select the .")
    parser.add_argument(
        "-w", "--words_file", type=Path, default="matched/oxford-understand.csv", help="Relative path to the target words"
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


# get freq 



# filter by freq


def main():
    args = parse_args()

    words_file = settings.PATH.dataset_root / args.words_file
    output_dir = settings.PATH.dataset_root / args.output_path / args.dataset / str(args.window_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    

if __name__ == "__main__":
    main()
