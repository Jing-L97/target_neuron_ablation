import argparse
import logging
from pathlib import Path

import pandas as pd

from neuron_analyzer import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Format file to save space.")
    parser.add_argument(
        "--neuron", choices=["zero", "random","mean"],default="zero",
        type=str, help="different abalation setting"
    )

    parser.add_argument(
        "--eval", choices=["cdi_childes", "oxford-understand"],default="cdi_childes",
        type=str, help="different eval"
    )

    return parser.parse_args()




def format_csv(file_path:Path)->pd.DataFrame:
    """Remove the target col from the csv file."""
    data = pd.read_csv(file_path)
    df = data.drop('ablated_neurons', axis=1)
    # convert differnet steps into different rows
    df_h = df[df["step"]==0]
    df_h = df_h.drop('surprisal', axis=1)
    df_grouped = df.groupby("step")
    for step, df_group, in df_grouped:
        df_h[step] = df_group['surprisal'].to_list()
    return df_h

def main() -> None:

    # loop over the different directories
    args = parse_args()

    surprisal_root = settings.PATH.result_dir / "surprisal" / args.neuron / args.eval
    surprisal_path = surprisal_root / "EleutherAI"
    out_path = surprisal_root / "formatted"
    surprisal_path.mkdir(parents=True, exist_ok=True)

    for file_path in surprisal_path.iterdir():
        logger.info(f"Load surprisal from {file_path}")
        df = format_csv(file_path)
        df.to_csv(out_path/file_path.name)
        logger.info(f"Save the formatted file to {out_path/file_path.name}")

if __name__ == "__main__":
    main()
