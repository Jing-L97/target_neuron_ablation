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
    df = pd.read_csv(file_path)
    if 'ablated_neurons' in df.columns:
        df = df.drop('ablated_neurons', axis=1)
    # convert differnet steps into different rows
    df_h = df[df["step"]==0]
    df_h = df_h.drop('surprisal', axis=1)
    df_grouped = df.groupby("step")
    for step, df_group in df_grouped:
        df_group = select_rows(df_group,df_h,["target_word","context_id","context"])
        df_h[step] = df_group['surprisal'].to_list()
        logger.info(f"Finished appending step {step}")
    return df_h


def select_rows(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,   # the ref df that we want to match
    match_columns: list[str]
) -> pd.DataFrame:
    """Select rows from df_a that match with df_b on specified columns."""
    # Validate that match_columns exist in both DataFrames
    for col in match_columns:
        if col not in df_a.columns or col not in df_b.columns:
            raise ValueError(f"Column '{col}' not found in both DataFrames")
    # remove duplicated rows
    cleaned_df = df_a.drop_duplicates(subset=["target_word","context_id","context"], keep="first")
    logger.info(f"Remove {df_a.shape[0] - cleaned_df.shape[0]} duplicated rows")
    result = pd.merge(
        df_b,
        cleaned_df,
        on=match_columns,
        how='left',
        indicator=True
    )
    if '_merge' in result.columns:
        result = result.drop(columns=['_merge'])
    return result




def main() -> None:

    # loop over the different directories
    args = parse_args()

    surprisal_root = settings.PATH.result_dir / "surprisal" / args.neuron / args.eval
    surprisal_path = surprisal_root / "EleutherAI"
    out_path = surprisal_root / "formatted"
    out_path.mkdir(parents=True, exist_ok=True)

    for file_path in surprisal_path.iterdir():
        out_file = out_path/file_path.name
        # look for whether the target file already exists
        if out_file.is_file():
            print(f"There exists {out_file}, skip")
        else:
            logger.info(f"Load surprisal from {file_path}")
            df = format_csv(file_path)
            df.to_csv(out_file)
            logger.info(f"Save the formatted file to {out_file}")

if __name__ == "__main__":
    main()
