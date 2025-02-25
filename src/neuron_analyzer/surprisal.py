#!/usr/bin/env python
import ast
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)

#######################################################
# Extract different steps

def generate_pythia_checkpoints() -> list[int]:
    """Generate complete list of Pythia checkpoint steps."""
    # Initial checkpoint
    checkpoints = [0]
    # Log-spaced checkpoints (2^0 to 2^9)
    log_spaced = [2**i for i in range(10)]  # 1, 2, 4, ..., 512

    # Evenly-spaced checkpoints from 1000 to 143000
    step_size = (143000 - 1000) // 142  # Calculate step size for even spacing
    linear_spaced = list(range(1000, 143001, step_size))

    # Combine all checkpoints
    checkpoints.extend(log_spaced)
    checkpoints.extend(linear_spaced)
    return sorted(list(set(checkpoints)))  # Remove duplicates and sort


@dataclass
class StepConfig:
    def __post_init__(self):
        """Initialize steps after instance creation."""
        self.steps = generate_pythia_checkpoints()
        logger.info(f"Generated {len(self.steps)} checkpoint steps")




#######################################################
# Neuron manipulation

class AblationConfig:
    """Configuration for neuron ablation."""

    def __init__(self, layer_num: str, neurons: list[int] | None = None, k: int = 10) -> None:
        """Initialize ablation configuration."""
        self.layer_name: str = f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
        self.neurons: list[int] | None = neurons
        self.k: int = k

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


#######################################################
# Extract surprisal by different steps

class StepSurprisalExtractor:
    """Extracts word surprisal across different training steps with neuron ablation."""

    def __init__(
        self,
        config: StepConfig,
        model_name: str,
        model_cache_dir: Path,
        layer_num:int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        step_ablations: dict[int, list[int]] | None = None,
    ) -> None:
        """Initialize the surprisal extractor."""
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.layer_num = layer_num
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
        """Setup ablator for current step with specified neurons."""
        # Clean up existing ablator
        if self.ablator:
            self.ablator.cleanup()
            self.ablator = None

        # Only proceed with ablation setup if step_ablations exists and contains the step
        if self.step_ablations is not None and step in self.step_ablations:
            # Create ablation config with the correct layer number (as string) and neurons list
            layer_num:str
            config = AblationConfig(
                layer_num=self.layer_num,
                neurons=self.step_ablations[step]
            )
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



#######################################################
# Util func to load prompt

def load_eval(
    word_path,
    word_header: str = "word",
    BOS_only: bool = True,
    prompt_header = None
    ) -> tuple[list[str], list[list[str]]]:
    """Load word and context lists from a JSON file."""
    try:
        with word_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {word_path}: {str(e)}")
    # Extract words
    target_words = list(data.keys())
    words = []
    contexts = []
    for word in target_words:
        word_contexts = []
        # Handle different JSON structures
        word_data = data[word]
        if len(word_data) == 0:
            continue
        else:
            words.append(word)
            for context_data in word_data:
                word_contexts.append(context_data["context"])
        contexts.append(word_contexts)
    logger.info(f"{len(target_words)-len(words)} words have no context!")
    return words, contexts



def load_neuron_dict(
    file_path: Path,
    key_col: str = "step",
    value_col: str = "top_neurons",
    random_base: bool= False
) -> dict[int, list[int]]:
    """Load a DataFrame and convert neuron values to integers."""
    df = pd.read_csv(file_path)

    result = {}
    for _, row in df.iterrows():
        try:
            # Parse the string to list of floats
            float_neurons = ast.literal_eval(row[value_col])
            # Extract the decimal part as integer; Converts '5.2021' format to 2021.
            neurons = [int(str(float(x)).split('.')[1]) for x in float_neurons]
            # generate the random indices excluded the 
            if random_base:
                neurons = generate_nerons(neurons)
            result[row[key_col]] = neurons
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing neuron list for step {row[key_col]}: {e}")
            result[row[key_col]] = []
    layer_num = [int(str(float(x)).split('.')[0]) for x in float_neurons][0]
    return result, layer_num



def generate_nerons(exclude_list: list[str], min_val: int = 1, max_val: int = 2048) -> list[int]:
    """Generate a list of non-repeating random integers with the same length as the input list."""
    # Convert all strings to integers for comparison
    excluded_ints = set(int(x) for x in exclude_list)
    # Calculate how many numbers we need
    count_needed = len(exclude_list)
    # Ensure the range is large enough to generate required unique numbers
    available_range = max_val - min_val + 1 - len(excluded_ints)
    if available_range < count_needed:
        max_val = min_val + count_needed + len(excluded_ints) - 1
    # Generate the list of non-repeating random integers
    result = []
    while len(result) < count_needed:
        rand_int = random.randint(min_val, max_val)
        if rand_int not in excluded_ints and rand_int not in result:
            result.append(str(rand_int))
    return result
