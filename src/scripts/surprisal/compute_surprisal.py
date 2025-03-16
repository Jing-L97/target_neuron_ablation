import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.surprisal import StepConfig, StepSurprisalExtractor, load_eval, load_neuron_dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument(
        "-w",
        "--word_path",
        type=Path,
        default="context/stas/c4-en-10k/5/oxford-understand.json",
        help="Relative path to the target words",
    )

    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument(
        "-n", "--neuron_file", type=str, default="500_10.csv", 
        help="Target model name"
    )
    parser.add_argument(
        "-a","--ablate", type=str, default="base", 
        choices=["base", "zero", "random","mean"],
        help="Neuron options for computing surprisal"
        )
    parser.add_argument("--use_bos_only", action="store_true", help="use_bos_only if enabled")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()



def get_count(filename:str)->str:
    return filename.split(".")[0].split("_")[1]


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    ###################################
    # load materials and paths
    ###################################

    # Load target words
    target_words, contexts = load_eval(settings.PATH.dataset_root / args.word_path)
    if args.debug:
        target_words, contexts = target_words[:5], contexts[:5]
        logger.info("Entering debugging mode. Loading first 5 words")
    else:
        logger.info(f"{len(target_words)} target words have been loaded.")

    # load neuron indices
    if args.ablate != "base":
        random_base = True if args.ablate == "random" else False
        step_ablations, layer_num = load_neuron_dict(
            settings.PATH.result_dir / "token_freq" / args.model_name / args.neuron_file,
            key_col = "step",
            value_col = "top_neurons",
            random_base = random_base
            )
        neuron_count = get_count(args.neuron_file)
        filename = f"{args.model_name}_{neuron_count}.debug" if args.debug else f"{args.model_name}_{neuron_count}.csv"
    elif args.ablate == "base":
        step_ablations = None
        filename = f"{args.model_name}.debug" if args.debug else f"{args.model_name}.csv"
        layer_num = None
    logger.info(f"Compute {args.ablate} surprisal")



    ###################################
    # Initialize classes
    ###################################

    result_file = settings.PATH.result_dir / "surprisal" / args.ablate / Path(args.word_path).stem/filename
    resume_file = settings.PATH.result_dir / "surprisal" / args.ablate / Path(args.word_path).stem/"resume"/filename
    result_file.parent.mkdir(parents=True, exist_ok=True)
    resume_file.parent.mkdir(parents=True, exist_ok=True)
    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(args.resume, file_path = resume_file )

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir / args.model_name
    extractor = StepSurprisalExtractor(
        steps_config,
        model_name=args.model_name,
        model_cache_dir=model_cache_dir,  # note here we use the relative path
        layer_num=layer_num,
        step_ablations = step_ablations,
    )

    ###################################
    # Save the target results
    ###################################
    try:
        results_df = extractor.analyze_steps(
            contexts=contexts, 
            target_words=target_words, 
            use_bos_only=args.use_bos_only,
            resume_path=resume_file
        )
        # Save results even if some checkpoints failed
        if not results_df.empty:
            results_df.to_csv(result_file, index=False)
            logger.info(
                f"Results saved to: {result_file}"
                f"Processed {len(results_df['step'].unique())} checkpoints successfully"
                )
        else:
            logger.warning("No results were generated")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
