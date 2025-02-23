import argparse
import ast
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from neuron_analyzer import settings
from neuron_analyzer.surprisal import StepConfig, StepSurprisalExtractor, load_eval

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
        default="context/stas/c4-en-10k/5/cdi_childes.json",
        help="Relative path to the target words",
    )

    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument(
        "-n", "--neuron_file", type=str, default="500_10.csv", 
        help="Target model name"
    )
    parser.add_argument("--use_bos_only", action="store_true", help="use_bos_only if enabled")

    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")

    return parser.parse_args()



def load_neuron_dict(
    file_path: Path,
    key_col: str = "step",
    value_col: str = "top_neurons",
) -> dict[int, list[int]]:
    """
    Load a DataFrame and convert neuron values to integers.
    Converts '5.2021' format to 2021.
    """
    df = pd.read_csv(file_path)

    result = {}
    for _, row in df.iterrows():
        try:
            # Parse the string to list of floats
            float_neurons = ast.literal_eval(row[value_col])
            # Extract the decimal part as integer
            neurons = [int(str(float(x)).split('.')[1]) for x in float_neurons]
            result[row[key_col]] = neurons
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing neuron list for step {row[key_col]}: {e}")
            result[row[key_col]] = []

    return result

# TODO: rely on the abalation anasis code to add the mena-abalation effect
@dataclass
class AblationConfig:
    """Configuration for neuron ablation."""

    layer_name: str = "gpt_neox.layers.5.mlp.dense_h_to_4h"  # Last MLP layer
    neurons: list[str] = None  # List of neuron indices to ablate
    k: int = 10  # Number of iterations for ablation analysis


class NeuronAblator:
    """Handles neuron ablation in transformer models."""

    def __init__(self, model: GPTNeoXForCausalLM, config: AblationConfig) -> None:
        """Initialize ablator with model and config."""
        self.model = model
        self.config = config
        self.is_ablated = False
        self.hook_handle = None
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron ablation."""

        def ablation_hook(module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            """Forward hook to zero out specified neurons."""
            if self.is_ablated and self.config.neurons:
                # Zero out activations for specified neurons
                for neuron_idx in self.config.neurons:
                    output[:, :, int(neuron_idx)] = 0
            return output

        # Get the MLP layer
        layer = dict(self.model.named_modules())[self.config.layer_name]

        # Register the forward hook
        self.hook_handle = layer.register_forward_hook(ablation_hook)

    def enable_ablation(self) -> None:
        """Enable neuron ablation."""
        self.is_ablated = True

    def disable_ablation(self) -> None:
        """Disable neuron ablation."""
        self.is_ablated = False

    def cleanup(self) -> None:
        """Remove the forward hook."""
        if self.hook_handle:
            self.hook_handle.remove()


class StepSurprisalExtractor:
    """Extracts word surprisal across different training steps with neuron ablation."""

    def __init__(
        self,
        config: StepConfig,
        model_name: str,
        model_cache_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        step_ablations: dict[int, list] | None = None,
    ) -> None:
        """Initialize the surprisal extractor."""
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.config = config
        self.device = device
        self.step_ablations = step_ablations
        self.ablator = None
        self.current_step = None
        logger.info(f"Using device: {self.device}")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.steps:
            raise ValueError("No steps provided for analysis")

        if isinstance(self.model_cache_dir, str):
            self.model_cache_dir = Path(self.model_cache_dir)

        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {self.device}. Use 'cpu' or 'cuda'")

        if self.step_ablations:
            invalid_steps = set(self.step_ablations.keys()) - set(self.config.steps)
            if invalid_steps:
                raise ValueError(f"Ablation config contains invalid steps: {invalid_steps}")

    def _setup_ablator(self, model: GPTNeoXForCausalLM, step: int) -> None:
        """Setup ablator for current step with specified neurons.

        Args:
            model: Model to apply ablation to
            step: Current training step
        """
        # Clean up existing ablator
        if self.ablator:
            self.ablator.cleanup()
            self.ablator = None
            logger.info(f"Activated ablation!")
        
        if step in self.step_ablations:
            logger.info(f"Found step {step} in the step_ablations")

        # Create new ablator if neurons are specified for this step
        if self.step_ablations and step in self.step_ablations:
            config = AblationConfig(neurons=self.step_ablations[step])
            self.ablator = NeuronAblator(model, config)
            logger.info(f"Created ablator for step {step} with neurons {self.step_ablations[step]}")

    def load_model_for_step(self, step: int) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
        """Load model and tokenizer for a specific step."""
        cache_dir = self.model_cache_dir / f"step{step}"
        logger.info(f"Loading model for step {step} on {self.device}")

        try:
            model = GPTNeoXForCausalLM.from_pretrained(self.model_name, revision=f"step{step}", cache_dir=cache_dir)
            model = model.to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=f"step{step}", cache_dir=cache_dir)
            model.eval()

            # Setup ablator for this step
            self._setup_ablator(model, step)
            self.current_step = step

            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model for step {step}: {str(e)}")
            raise

    def compute_surprisal(
        self,
        model: GPTNeoXForCausalLM,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
        ablated: bool = False,
    ) -> float:
        """Compute surprisal for a target word given a context."""
        try:
            # Handle ablation if configured
            if self.ablator and ablated:
                self.ablator.enable_ablation()
            elif self.ablator:
                self.ablator.disable_ablation()
            # Rest of the compute_surprisal implementation remains the same
            if use_bos_only:
                bos_token = tokenizer.bos_token
                input_text = bos_token + target_word
                bos_tokens = tokenizer(bos_token, return_tensors="pt").to(self.device)
                context_length = bos_tokens.input_ids.shape[1]
            else:
                input_text = context + target_word
                context_tokens = tokenizer(context, return_tensors="pt").to(self.device)
                context_length = context_tokens.input_ids.shape[1]

            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)

            if context_length >= inputs.input_ids.shape[1]:
                context_length = inputs.input_ids.shape[1] - 1

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                target_logits = logits[0, context_length - 1 : context_length]

                if context_length < inputs.input_ids.shape[1]:
                    target_token_id = inputs.input_ids[0, context_length].item()
                    log_prob = torch.log_softmax(target_logits, dim=-1)[0, target_token_id]
                    surprisal = -log_prob.item()
                else:
                    logger.error("Cannot compute surprisal: unable to identify target token")
                    surprisal = float("nan")

            return surprisal

        except Exception as e:
            logger.error(f"Error in compute_surprisal: {str(e)}")
            return float("nan")

    def analyze_steps(
        self,
        contexts: list[list[str]],
        target_words: list[str],
        use_bos_only: bool = True,
    ) -> pd.DataFrame:
        """Analyze surprisal across steps with optional neuron ablation."""
        results = []

        for step in self.config.steps:
            logger.info(f"Processing step {step}")
            try:
                model, tokenizer = self.load_model_for_step(step)

                for word_contexts, target_word in zip(contexts, target_words):
                    for context_idx, context in enumerate(word_contexts):
                        # If we have neurons to ablate for this step, compute only ablated surprisal
                        if self.ablator and step in self.step_ablations:
                            surprisal = self.compute_surprisal(
                                model, tokenizer, context, target_word,
                                use_bos_only=use_bos_only,
                                ablated=True
                            )
                            result = {
                                "step": step,
                                "target_word": target_word,
                                "context_id": context_idx,
                                "context": "BOS_ONLY" if use_bos_only else context,
                                "surprisal": surprisal,
                                "ablated_neurons": str(self.step_ablations[step]),
                            }
                        else:
                            # If no neurons to ablate, compute normal surprisal
                            surprisal = self.compute_surprisal(
                                model, tokenizer, context, target_word,
                                use_bos_only=use_bos_only,
                                ablated=False
                            )
                            result = {
                                "step": step,
                                "target_word": target_word,
                                "context_id": context_idx,
                                "context": "BOS_ONLY" if use_bos_only else context,
                                "surprisal": surprisal,
                            }
                        results.append(result)

                # Cleanup
                del model
                del tokenizer
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing step {step}: {str(e)}")
                continue

        return pd.DataFrame(results)



    def cleanup(self) -> None:
        """Clean up resources."""
        if self.ablator:
            self.ablator.cleanup()


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # Load target words
    target_words, contexts = load_eval(settings.PATH.dataset_root / args.word_path)
    logger.info(f"{len(target_words)} target words have been loaded.")
    # load neuron indices
    step_ablations = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / args.model_name / args.neuron_file,
        key_col = "step",
        value_col = "top_neurons"
        )
    if args.debug:
        target_words, contexts = target_words[:5], contexts[:5]
        logger.info("Entering debugging mode. Loading first 5 words")

    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig()

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir / args.model_name

    extractor = StepSurprisalExtractor(
        steps_config,
        model_name=args.model_name,
        model_cache_dir=model_cache_dir,  # note here we use the relative path
        step_ablations = step_ablations
    )

    try:
        # Analyze steps with BOS only
        results_df = extractor.analyze_steps(
            contexts=contexts, target_words=target_words, use_bos_only=args.use_bos_only
        )
        # Save results even if some checkpoints failed
        if not results_df.empty:
            result_folder = settings.PATH.result_dir / "surprisal" / "neuron"
            if args.debug:
                out_path = result_folder / f"{args.model_name}.debug"
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = result_folder / f"{args.model_name}.csv"
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
