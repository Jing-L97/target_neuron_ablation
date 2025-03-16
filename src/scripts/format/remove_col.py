import logging
from pathlib import Path


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def remove_col(file_path:Path)->pd.DataFrame:
    """Remove the target col from the csv file."""


def main(base_dir: str | Path, dry_run: bool = False) -> None:
    """Main function to fix directory structure.

    Args:
        base_dir: Base directory to process
        dry_run: If True, only print actions without executing
    """
    setup_logging()
    base_path = Path(base_dir).resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    logging.info(f"Processing directory: {base_path}")
    fix_nested_structure(base_path, dry_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix nested k10.feather files structure")

    parser.add_argument(
        "--directory",
        default="/scratch2/jliu/Generative_replay/neuron/results/ablations/unigram/EleutherAI/pythia-410m-deduped",
        type=str,
        help="Base directory to process",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them")

    args = parser.parse_args()
    main(args.directory, args.dry_run)
