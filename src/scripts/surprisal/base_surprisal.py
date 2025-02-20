#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.surprisal import StepConfig, StepSurprisalExtractor, load_eval

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument(
        "-w","--word_path",type=Path,
        default="context/stas/c4-en-10k/5/cdi_childes.json",
        help="Relative path to the target words"
        )

    parser.add_argument(
        "-m","--model_name", type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="Target model name"
        )

    parser.add_argument("--use_bos_only", action="store_true",
        help="use_bos_only if enabled"
        )

    parser.add_argument("--debug", action="store_true",
        help="Compute the first few 5 lines if enabled"
        )

    return parser.parse_args()





def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # Load target words
    target_words, contexts = load_eval(settings.PATH.dataset_root/args.word_path)
    logger.info(f"{len(target_words)} target words have been loaded.")

    if args.debug:
        target_words, contexts = target_words[:5], contexts[:5]
        logger.info("Entering debugging mode. Loading first 5 words")

    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig()

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir/args.model_name
    extractor = StepSurprisalExtractor(
        steps_config,
        model_name = args.model_name,
        model_cache_dir = model_cache_dir,   # note here we use the relative path
        )

    try:
        # Analyze steps with BOS only
        results_df = extractor.analyze_steps(contexts=contexts, target_words=target_words, use_bos_only=args.use_bos_only)
        # Save results even if some checkpoints failed
        if not results_df.empty:
            result_folder = settings.PATH.result_dir/"surprisal"/"base"
            if args.debug:
                out_path =result_folder/f"{args.model_name}.debug"
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = result_folder/f"{args.model_name}.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_path, index=False)
            logger.info(f"Results saved to: {out_path}")
            logger.info(f"Processed {len(results_df['step'].unique())} checkpoints successfully")
        else:
            logger.warning("No results were generated")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise



if __name__ == "__main__":
    main()
