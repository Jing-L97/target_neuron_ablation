import argparse
import logging
from pathlib import Path

import torch

from neuron_analyzer import settings
from neuron_analyzer.freq import load_unigram
from neuron_analyzer.surprisal import StepConfig, StepSurprisalExtractor, load_eval, load_neuron_dict, sel_eval

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
        default="context/stas/c4-en-10k/5/merged.json",
        help="Relative path to the target words",
    )

    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument(
        "-n", "--neuron_file", type=str, default="500_10.csv", 
        help="Target model name"
    )
    parser.add_argument("--effect", type=str, choices=["boost", "suppress"],
        default="supress", help="boost or supress long-tail"
        )
    parser.add_argument(
        "--vector", type=str, default="longtail",
        choices=["mean", "longtail"],
        help="Differnt ablation model for freq vectors"
        )
    parser.add_argument(
        "-a","--ablation_mode", type=str, default="base", 
        choices=["base", "zero", "random","mean","scaled","full"],
        help="Neuron options for computing surprisal"
        )

    parser.add_argument(
        "--eval_lst", type=list, help="eval file list",
        default=[
            "freq/EleutherAI/pythia-410m/cdi_childes.csv",
            "freq/EleutherAI/pythia-410m/oxford-understand.csv"
            ]
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    if args.ablation_mode != "base":
        random_base = True if args.ablation_mode == "random" else False
        step_ablations, layer_num = load_neuron_dict(
            settings.PATH.result_dir / "token_freq" /args.effect / args.vector / args.model_name / args.neuron_file,
            key_col = "step",
            value_col = "top_neurons",
            random_base = random_base
            )
        neuron_count = get_count(args.neuron_file)
        filename = f"{args.model_name}_{neuron_count}.debug" if args.debug else f"{args.model_name}_{neuron_count}.csv"
    elif args.ablation_mode == "base":
        step_ablations = None
        filename = f"{args.model_name}.debug" if args.debug else f"{args.model_name}.csv"
        layer_num = None
    logger.info(f"Compute {args.ablation_mode} surprisal")

    # load unigram freq file
    token_frequencies = load_unigram(args.model_name,device)


    ###################################
    # Initialize classes
    ###################################
    result_dir = settings.PATH.surprisal_dir / args.effect / args.vector / args.ablation_mode
    result_file = result_dir / Path(args.word_path).stem/filename
    result_file.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_file = result_dir / Path(args.word_path).stem/"resume"/filename
        resume_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        resume_file = None
    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(resume=args.resume,debug= args.debug,file_path = resume_file)

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir / args.model_name
    extractor = StepSurprisalExtractor(
        config=steps_config,
        model_name=args.model_name,
        model_cache_dir=model_cache_dir,  # note here we use the relative path
        layer_num=layer_num,
        step_ablations = step_ablations,
        device=device,
        ablation_mode = args.ablation_mode,
        token_frequencies = token_frequencies
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
                f"Results saved to: {result_file}\n"
                f"Processed {len([col for col in results_df.columns if str(col).isdigit()])} checkpoints successfully"
            )
            if "merged" in str(args.word_path):
                # save the eval file
                for eval_path in args.eval_lst:
                    sel_eval(results_df,eval_path,result_dir,filename)
        else:
            logger.warning("No results were generated")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise



if __name__ == "__main__":
    main()
