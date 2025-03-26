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
    parser.add_argument("--input_path", default="longtail/mean", type=str, help="different abalation setting")
    parser.add_argument("--operation", default="segment", type=str, help="which operation")
    return parser.parse_args()


class CsvMerger:
    """Class for merging and saving CSV files from different datasets."""

    def __init__(
        self,
        file_path: Path,
        model_lst: list[str] = ["70m", "410m"],
        neuron_lst: list[int] = [10, 50, 500],
    ):
        """Initialize the CSV merger."""
        self.file_path = Path(file_path)
        self.model_lst = model_lst
        self.neuron_lst = neuron_lst
        self.out_path = self.file_path / "merged" / "EleutherAI"

    def merge_csv(self, cdi_data: pd.DataFrame, ox_data: pd.DataFrame) -> pd.DataFrame:
        # Find words in ox_data that aren't in cdi_data
        words = set(ox_data["target_word"]) - set(cdi_data["target_word"])

        # Select the additional words
        sel_df = ox_data[ox_data["target_word"].isin(words)]

        # Concatenate the rows
        merged = pd.concat([cdi_data, sel_df])
        return merged

    def process_files(self) -> None:
        # Create output directory
        self.out_path.mkdir(parents=True, exist_ok=True)

        for model in self.model_lst:
            for neuron in self.neuron_lst:
                try:
                    # Load input files
                    cdi_path = self.file_path / f"cdi_childes/EleutherAI/pythia-{model}-deduped_{neuron}.csv"
                    ox_path = self.file_path / f"oxford-understand/EleutherAI/pythia-{model}-deduped_{neuron}.csv"

                    cdi_data = pd.read_csv(cdi_path)
                    ox_data = pd.read_csv(ox_path)

                    # Merge the data
                    merged = self.merge_csv(cdi_data, ox_data)

                    # Save to output file
                    output_file = self.out_path / f"pythia-{model}-deduped_{neuron}.csv"
                    merged.to_csv(output_file, index=False)

                    logger.info(f"Saved the merged file to {output_file}")
                except FileNotFoundError as e:
                    logger.error(f"Error processing {model}/{neuron}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing {model}/{neuron}: {e}")


class EvaluationSelector:
    """Class for selecting word subsets from evaluation files."""

    def __init__(
        self,
        result_dir: Path | str,
        model_lst: list[str] = ["70m", "410m"],
        neuron_lst: list[int] = [10, 50, 500],
    ):
        self.result_dir = Path(result_dir)
        self.model_lst = model_lst
        self.neuron_lst = neuron_lst

    def sel_eval(self, results_df: pd.DataFrame, eval_path: Path | str, result_dir: Path | str, filename: str) -> None:
        # Load eval file
        eval_file = settings.PATH.dataset_root / eval_path
        result_file = Path(result_dir) / eval_file.stem / filename
        result_file.parent.mkdir(parents=True, exist_ok=True)

        eval_frame = pd.read_csv(eval_file)

        # Select target words
        results_df_sel = results_df[results_df["target_word"].isin(eval_frame["word"])]
        results_df_sel.to_csv(result_file, index=False)

        logger.info(f"Eval set saved to: {result_file}")


    def process_files(self) -> None:

        file_path = self.result_dir/"merged/EleutherAI"
        eval_path = "freq/EleutherAI/pythia-410m"
        eval_lst = ["cdi_childes.csv", "oxford-understand.csv"]

        for eval_filename in eval_lst:
            eval_file = settings.PATH.dataset_root / eval_path / eval_filename

            for model in self.model_lst:
                for neuron in self.neuron_lst:
                    try:
                        filename = f"pythia-{model}-deduped_{neuron}.csv"
                        results_df = pd.read_csv(file_path / filename)

                        self.sel_eval(
                            results_df=results_df, eval_path=eval_file, result_dir=self.result_dir, filename=filename
                        )
                    except FileNotFoundError as e:
                        logger.error(f"File not found for {model}/{neuron}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing {model}/{neuron}: {e}")


def main() -> None:
    # loop over the different directories
    args = parse_args()
    model_lst = ["70m", "410m"]
    neuron_lst = [10, 50, 500]
    file_path = settings.PATH.result_dir / "surprisal" / args.input_path

    if args.operation == "merge":
        merger = CsvMerger(file_path=file_path, model_lst=model_lst, neuron_lst=neuron_lst)
        # Process all files
        merger.process_files()

    if args.operation == "segment":
        selector = EvaluationSelector(
        result_dir=file_path,
        model_lst=model_lst,
        neuron_lst=neuron_lst
        )
        # Process files
        selector.process_files()

if __name__ == "__main__":
    main()
