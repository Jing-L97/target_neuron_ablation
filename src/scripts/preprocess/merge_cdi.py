import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.preprocess.preprocess import load_data, save_data

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="collect n-gram contexts from a corpus.")
    parser.add_argument("--file1_path", type=Path, default="stas/c4-en-10k/5/cdi_childes.json")

    parser.add_argument("--file2_path", type=Path, default="stas/c4-en-10k/5/oxford-understand.json")

    parser.add_argument("-o", "--output_path", type=Path, default="stas/c4-en-10k/5/merged.json")
    return parser.parse_args()


def merge_json_files(file1_path: Path, file2_path: Path, output_path: Path) -> None:
    """Merge two JSON files with similar structure, combining word entries."""
    # Load the two JSON files
    data1 = load_data(file1_path)
    logger.info(f"Loading file from {file1_path}")
    data2 = load_data(file2_path)
    logger.info(f"Loading file from {file2_path}")
    # Create the merged data structure
    merged_data = {}

    # Process the first file
    for word, contexts in data1.items():
        merged_data[word] = contexts.copy()

    # Process the second file, adding new words and contexts
    for word, contexts in data2.items():
        if word in merged_data:
            # Word exists in both files, add any new contexts
            existing_contexts = {context_item["context"] for context_item in merged_data[word]}

            for context_item in contexts:
                if context_item["context"] not in existing_contexts:
                    merged_data[word].append(context_item)
        else:
            # Word only exists in the second file
            merged_data[word] = contexts.copy()

    # Write the merged data to the output file
    save_data(merged_data, output_path)
    logger.info(f"Save file to {output_path}")


def main():
    args = parse_args()

    file1_path = settings.PATH.context_dir / args.file1_path
    file2_path = settings.PATH.context_dir / args.file2_path
    output_path = settings.PATH.context_dir / args.output_path
    merge_json_files(file1_path, file2_path, output_path)


if __name__ == "__main__":
    main()
