from pathlib import Path
import shutil
import logging
import typing as t


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fix_nested_structure(base_dir: Path, dry_run: bool = False) -> None:
    """Fix k10.feather nested structure by moving files and cleaning directories.

    Args:
        base_dir: Base directory to process
        dry_run: If True, only print actions without executing
    """
    # Create temporary directory for moving files
    temp_dir = base_dir / "temp_feather"
    if not dry_run:
        temp_dir.mkdir(exist_ok=True)

    try:
        # First step: Move all nested k10.feather files to temp directory
        for path in base_dir.glob("**/k10.feather/k10.feather"):
            try:
                step_dir = path.parent.parent.parent  # Get the step directory (e.g., '0', '1000')
                temp_path = temp_dir / f"k10_step{step_dir.name}.feather"

                if dry_run:
                    logging.info(f"Would move {path} -> {temp_path}")
                    continue

                shutil.move(str(path), str(temp_path))
                logging.info(f"Moved {path} -> {temp_path}")

            except Exception as e:
                logging.error(f"Error moving file {path}: {e}")

        # Second step: Remove empty k10.feather directories
        for dir_path in base_dir.glob("**/k10.feather"):
            if dir_path.is_dir():
                try:
                    if dry_run:
                        logging.info(f"Would remove directory: {dir_path}")
                        continue

                    shutil.rmtree(dir_path)
                    logging.info(f"Removed directory: {dir_path}")

                except Exception as e:
                    logging.error(f"Error removing directory {dir_path}: {e}")

        # Third step: Move files back to their correct locations
        if not dry_run:
            for temp_file in temp_dir.glob("*.feather"):
                try:
                    step_num = temp_file.stem.split("step")[1]  # Extract step number
                    target_path = base_dir / step_num / "500" / "k10.feather"

                    shutil.move(str(temp_file), str(target_path))
                    logging.info(f"Moved {temp_file} -> {target_path}")

                except Exception as e:
                    logging.error(f"Error moving file back {temp_file}: {e}")

            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            logging.info(f"Removed temporary directory: {temp_dir}")

    except Exception as e:
        logging.error(f"Error during processing: {e}")


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
