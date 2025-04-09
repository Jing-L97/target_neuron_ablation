import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.analysis.freq import UnigramAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Annotate word frequency.")
    parser.add_argument(
        "-u",
        "--unigram_file",
        type=Path,
        default="pythia-unigrams.npy",
        help="Relative path to the target words",
    )
    parser.add_argument(
        "-w", "--word_file", type=Path, default="cdi_childes.csv", help="Relative path to the extracted context"
    )
    parser.add_argument("-m", "--model_name", type=str, default="EleutherAI/pythia-410m", help="Target model name")
    return parser.parse_args()


def get_word_stat(df: pd.DataFrame, col_header: str, unigram_analyzer) -> pd.DataFrame:
    # Apply get_word_unigram_stats to each word in the column
    stats = df[col_header].apply(unigram_analyzer.get_word_unigram_stats)
    # Extract the statistics into separate columns
    df["token_len"] = stats.apply(lambda x: x["total_tokens"])
    df["count"] = stats.apply(lambda x: x["total_count"])
    df["freq"] = stats.apply(lambda x: x["avg_frequency"])
    df["freq_m"] = df["freq"] * 1_000_000
    return df


def main():
    """Main function demonstrating usage."""
    args = parse_args()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # set paths
    word_path = settings.PATH.dataset_root / "src" / "cdi" / args.word_file
    unigram_path = settings.PATH.dataset_root / "src" / "unigram" / args.unigram_file
    out_path = settings.PATH.dataset_root / "freq" / args.model_name / args.word_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # load csv file
    logger.info(f"Loading file from {word_path}")
    word_df = pd.read_csv(word_path)
    # intilize unigram analyzer
    logger.info(f"Matching unigram from {unigram_path}")
    unigram_analyzer = UnigramAnalyzer(model_name=args.model_name, device=device)
    # match word freq
    freq_df = get_word_stat(word_df, "word", unigram_analyzer)
    # save the file to the target directory
    freq_df.to_csv(out_path)
    logger.info(f"Save file to {out_path}")


if __name__ == "__main__":
    main()
