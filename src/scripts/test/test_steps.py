import argparse
import logging

from neuron_analyzer.model_util import StepConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument("--interval", type=int, default=20, help="Checkpoint intervals")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(resume=args.resume, debug=args.debug, file_path=None, interval=args.interval)
    print(steps_config.steps)


if __name__ == "__main__":
    main()
