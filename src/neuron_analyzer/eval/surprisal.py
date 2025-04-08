#!/usr/bin/env python
import ast
import json
import logging
import random
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from neuron_analyzer import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")

#######################################################
# Util func to set up steps

class StepConfig:
    """Configuration for step-wise analysis of model checkpoints."""

    def __init__(
        self, resume: bool = False, debug: bool = False, file_path: Path | None = None, interval: int = 1
    ) -> None:
        """Initialize step configuration."""
        # Generate the complete list of steps
        self.steps = self.generate_pythia_checkpoints()

        # Apply interval sampling while preserving start and end steps
        if interval > 1:
            self.steps = self._apply_interval_sampling(self.steps, interval)
            logger.info(f"Applied interval sampling with n={interval}, resulting in {len(self.steps)} steps")

        if debug:
            self.steps = self.steps[:5]
            logger.info("Entering debugging mode, select first 5 steps.")

        # If resuming, filter out already processed steps
        if resume and file_path is not None:
            self.steps = self.recover_steps(file_path)

        logger.info(f"Generated {len(self.steps)} checkpoint steps")

    def _apply_interval_sampling(self, steps: list[int], interval: int) -> list[int]:
        """Sample steps at every nth interval while preserving start and end steps."""
        if len(steps) <= 2:
            return steps

        # Always keep first and last steps
        first_step = steps[0]
        last_step = steps[-1]

        # Sample the middle steps at the specified interval
        middle_steps = steps[1:-1][::interval]

        # Combine and return
        result = [first_step] + middle_steps + [last_step]

        # Ensure no duplicates if interval sampling results in last step being included twice
        return sorted(list(set(result)))

    def generate_pythia_checkpoints(self) -> list[int]:
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

        # Remove duplicates and sort
        return sorted(list(set(checkpoints)))

    def recover_steps(self, file_path: Path) -> list[int]:
        """Filter out steps that have already been processed based on column names."""
        if not file_path.is_file():
            return self.steps

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract completed steps from column headers (only consider fully numeric columns)
        completed_steps = set()
        for col in df.columns:
            if col.isdigit():
                completed_steps.add(int(col))

        return [step for step in self.steps if step not in completed_steps]


#######################################################
# Util func to compute surprisal


class AblationConfig:
    """Configuration for neuron ablation."""

    def __init__(
        self,
        layer_num: str,
        neurons: list[int] | None = None,
        ablation_mode: str = "base",
        k: int = 10,
        scaling_factor: float = 1.5,
        top_k_percent: float = 0.05,
        variance_threshold: float = 0.95,
        token_frequencies: torch.Tensor = None,
        model_name: str = "pythia-410m",
    ):
        """Initialize ablation configuration."""
        self.layer_name: str = f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
        self.neurons = neurons
        self.ablation_mode = ablation_mode
        self.scaling_factor = scaling_factor
        self.top_k_percent = top_k_percent
        self.variance_threshold = variance_threshold
        self.token_frequencies = token_frequencies
        self.model_name = model_name
        self.k: int = k


class NeuronAblator:
    """Handles neuron ablation in transformer models."""

    def __init__(
        self, model: GPTNeoXForCausalLM, config: AblationConfig, token_frequencies: t.Optional[torch.Tensor] = None
    ) -> None:
        """Initialize ablator with model and config."""
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Load token frequencies if needed
        if token_frequencies is None and config.ablation_mode == "scaled":
            self.token_frequencies = self.config.token_frequencies
        else:
            self.token_frequencies = token_frequencies
        self.hook_handle = None
        self.ablation_enabled = False
        self.null_space_basis = None
        self.projection_matrix = None

        # Compute null space if token frequencies are provided and scaled_activation mode is used
        if self.token_frequencies is not None and config.ablation_mode == "scaled":
            self._compute_token_null_space()

        self._setup_hooks()

    def _get_unembedding_matrix(self) -> torch.Tensor:
        """Access the unembedding matrix from the model."""
        try:
            return self.model.embed_out.weight
        except AttributeError:
            pass

        try:
            return self.model.lm_head.weight
        except AttributeError:
            pass

        # Try other common names
        for name, param in self.model.named_parameters():
            if any(term in name.lower() for term in ["unembed", "lm_head", "embed_out", "output_embedding"]):
                logger.info(f"Found unembedding matrix: {name}")
                return param

        # If none found, raise an error
        raise ValueError("Cannot access the unembedding matrix from the model")

    def _compute_token_null_space(self) -> None:
        """Compute the null space of the most frequent tokens based on token frequencies"""
        if self.token_frequencies is None:
            logger.warning("No token frequencies provided. Cannot compute token null space.")
            return

        # Get the unembedding matrix
        W_U = self._get_unembedding_matrix()

        # Get the most frequent tokens
        vocab_size = self.token_frequencies.shape[0]
        top_k = int(vocab_size * self.config.top_k_percent)
        top_indices = torch.argsort(self.token_frequencies, descending=True)[:top_k]

        # Extract the embeddings of top frequent tokens
        frequent_token_embeddings = W_U[top_indices].to(self.device)

        # Get the embedding dimension and hidden size (for neuron activations)
        embedding_dim = frequent_token_embeddings.shape[1]
        hidden_size = None

        # Try to get the hidden size from the model configuration
        try:
            if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
                hidden_size = self.model.config.hidden_size
            else:
                # For models where we can't directly get hidden_size from config
                # Find the activation dimensions by looking at the MLP layer
                layer_path = self.config.layer_name.split(".")
                module = self.model
                for part in layer_path:
                    if hasattr(module, part):
                        module = getattr(module, part)

                # Check the output dimension of the module
                if hasattr(module, "out_features"):
                    hidden_size = module.out_features
                elif hasattr(module, "weight") and hasattr(module.weight, "shape"):
                    # For typical linear layers, weight shape is [out_features, in_features]
                    hidden_size = module.weight.shape[0]
                else:
                    # If we can't determine it, use the first activation dimension
                    # This is a fallback and might not be correct for all models
                    sample_input = torch.zeros(1, 1, embedding_dim, device=self.device)
                    with torch.no_grad():
                        try:
                            sample_output = module(sample_input)
                            hidden_size = sample_output.shape[-1]
                        except:
                            # Fallback to embedding dimension if all else fails
                            logger.warning(f"Could not determine hidden size, using embedding dim: {embedding_dim}")
                            hidden_size = embedding_dim
        except Exception as e:
            logger.warning(f"Error determining hidden size: {e}. Using embedding dim: {embedding_dim}")
            hidden_size = embedding_dim

        # Convert to numpy for SVD calculation
        embeddings_np = frequent_token_embeddings.detach().cpu().numpy()

        # Perform SVD to find the principal components
        U, S, Vh = np.linalg.svd(embeddings_np, full_matrices=True)

        # Calculate cumulative explained variance
        explained_variance_ratio = S**2 / np.sum(S**2)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find how many components to keep based on threshold
        k = np.argmax(cumulative_variance >= self.config.variance_threshold) + 1
        logger.info(f"Using {k} principal components to span the space of top {top_k} tokens (out of {vocab_size})")

        # The null space is the remaining basis vectors
        null_space_basis = Vh[k:].T

        # Convert to PyTorch tensor
        null_space_basis_tensor = torch.tensor(null_space_basis, dtype=torch.float32, device=self.device)

        # Now we need to adapt the null space basis to match the hidden size
        if embedding_dim != hidden_size:
            # We'll create a simple linear mapping from embedding space to hidden space
            try:
                # Try to find a model component that maps between these spaces
                for name, param in self.model.named_parameters():
                    if ("embed" in name.lower() or "proj" in name.lower()) and param.shape == (
                        hidden_size,
                        embedding_dim,
                    ):
                        logger.info(f"Using existing projection from model: {name}")
                        projection = param.detach()
                        # Transform the null space basis
                        adapted_basis = torch.mm(projection, null_space_basis_tensor.T).T
                        break
                else:
                    # If no suitable projection is found, use a simple dimension adaptation
                    if embedding_dim > hidden_size:
                        # Reduce dimensions by taking the most significant ones
                        logger.info("Reducing null space dimensions")
                        adapted_basis = null_space_basis_tensor[:, :hidden_size]
                    else:
                        # Increase dimensions by padding with zeros
                        logger.info("Expanding null space dimensions with zero padding")
                        padding = torch.zeros(
                            null_space_basis_tensor.shape[0], hidden_size - embedding_dim, device=self.device
                        )
                        adapted_basis = torch.cat([null_space_basis_tensor, padding], dim=1)
            except Exception as e:
                logger.warning(f"Error in adaptation: {e}. Using dimension truncation/padding.")
                if embedding_dim > hidden_size:
                    adapted_basis = null_space_basis_tensor[:, :hidden_size]
                else:
                    padding = torch.zeros(
                        null_space_basis_tensor.shape[0], hidden_size - embedding_dim, device=self.device
                    )
                    adapted_basis = torch.cat([null_space_basis_tensor, padding], dim=1)
        else:
            # Dimensions already match, no adaptation needed
            adapted_basis = null_space_basis_tensor

        # Store the adapted null space basis
        self.null_space_basis = adapted_basis

        # Compute the projection matrix for the adapted basis
        self.projection_matrix = torch.mm(adapted_basis, adapted_basis.T)

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron ablation."""

        def ablation_hook(module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            """Forward hook to ablate specified neurons using zero, mean, or scaled activation."""
            if not self.ablation_enabled:
                return output

            # Clone the output to avoid modifying the original tensor in-place
            modified_output = output.clone()

            # If we reach here, ablation is enabled and we have neurons to ablate
            if self.config.ablation_mode in ["zero", "random"]:
                # Zero activation - set activations to 0
                for neuron_idx in self.config.neurons:
                    modified_output[:, :, int(neuron_idx)] = 0

            elif self.config.ablation_mode == "full":
                # Zero activation - set activations to 0
                for neuron_idx in self.config.neurons:
                    modified_output[:, :, int(neuron_idx)] = 1

            elif self.config.ablation_mode == "mean":
                # Calculate mean activation across all neurons (dimension 2)
                mean_activations = torch.mean(output, dim=2)
                # Replace each specified neuron's activation with the mean
                for neuron_idx in self.config.neurons:
                    for b in range(output.shape[0]):
                        for s in range(output.shape[1]):
                            modified_output[b, s, int(neuron_idx)] = mean_activations[b, s]

            elif self.config.ablation_mode == "scaled":
                # Verify that we have computed the null space basis
                if self.null_space_basis is None:
                    logger.warning("Null space basis not computed. Skipping scaled activation.")
                    return modified_output

                # Apply scaling in the null space for specified neurons
                for neuron_idx in self.config.neurons:
                    # Get dimensions
                    batch_size, seq_length = output.shape[0], output.shape[1]

                    # Process each position individually
                    for b in range(batch_size):
                        for s in range(seq_length):
                            # Get the activation value for this position
                            activation = output[b, s, int(neuron_idx)].item()

                            # Create a vector representation for this activation
                            # This maps it into the space where our null space basis exists
                            activation_vector = (
                                torch.ones(self.null_space_basis.shape[0], device=self.device) * activation
                            )

                            # Project the activation onto the null space
                            # The projection matrix is self.projection_matrix
                            null_space_component = torch.mv(self.projection_matrix, activation_vector)

                            # Calculate the magnitude of projection in the null space
                            projection_magnitude = torch.sum(null_space_component)

                            # Scale only the null space component, then add back to original activation
                            scaled_activation = activation + (self.config.scaling_factor - 1.0) * projection_magnitude

                            # Update the activation
                            modified_output[b, s, int(neuron_idx)] = scaled_activation

                    logger.debug(f"Applied null space scaling to neuron {neuron_idx}")

            return modified_output

        # Get the MLP layer
        try:
            layer = dict(self.model.named_modules())[self.config.layer_name]
            # Register the forward hook
            self.hook_handle = layer.register_forward_hook(ablation_hook)
        except KeyError:
            logger.error(f"Layer {self.config.layer_name} not found in model")
            raise ValueError(f"Layer {self.config.layer_name} not found in model")

    def enable_ablation(self) -> None:
        """Enable neuron ablation."""
        self.ablation_enabled = True

    def disable_ablation(self) -> None:
        """Disable neuron ablation."""
        self.ablation_enabled = False

    def cleanup(self) -> None:
        """Remove the hook and clean up resources."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        # Clear tensors
        self.null_space_basis = None
        self.projection_matrix = None


#######################################################
# Util func to set up steps


class StepSurprisalExtractor:
    """Extracts word surprisal across different training steps with neuron ablation."""

    def __init__(
        self,
        config: StepConfig,
        model_name: str,
        model_cache_dir: Path,
        device: str,
        ablation_mode: str = "base",
        layer_num: int = None,
        step_ablations: dict[int, list[int]] | None = None,
        token_frequencies: torch.Tensor = None,
        scaling_factor: float = 1.5,
        top_k_percent: float = 0.05,
        variance_threshold: float = 0.95,
    ) -> None:
        """Initialize the surprisal extractor."""

        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.layer_num = str(layer_num)
        self.config = config
        self.device = device
        self.step_ablations = step_ablations
        self.ablation_mode = ablation_mode
        self.ablator = None
        self.current_step = None
        self.token_frequencies = token_frequencies
        self.scaling_factor = scaling_factor
        self.top_k_percent = top_k_percent
        self.variance_threshold = variance_threshold
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

        # Additional validation for scaled_activation mode
        if self.ablation_mode == "scaled" and self.token_frequencies is None:
            logger.warning("scaled_activation mode requires token_frequency file, but none provided")

    def _setup_ablator(self, model: GPTNeoXForCausalLM, step: int) -> None:
        """Setup ablator for current step with specified neurons."""
        # Clean up existing ablator
        if self.ablator:
            self.ablator.cleanup()
            self.ablator = None

        # Only proceed with ablation setup if step_ablations exists and contains the step
        if self.step_ablations is not None and step in self.step_ablations and self.step_ablations[step]:
            # Create ablation config with the correct layer, neurons, and ablation mode
            ablation_config = AblationConfig(
                layer_num=self.layer_num,
                neurons=self.step_ablations[step],
                ablation_mode=self.ablation_mode,
                scaling_factor=self.scaling_factor,
                top_k_percent=self.top_k_percent,
                variance_threshold=self.variance_threshold,
                token_frequencies=self.token_frequencies,
                model_name=self.model_name,
            )
            # Create the ablator with the model and config
            self.ablator = NeuronAblator(model, ablation_config)
            logger.info(
                f"Created {self.ablation_mode} ablator for step {step} with {len(self.step_ablations[step])} neurons."
            )
        else:
            # logger.info(f"No ablation configured for step {step}")
            pass

    def load_model_for_step(self, step: int) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
        """Load model and tokenizer for a specific step."""
        cache_dir = self.model_cache_dir / f"step{step}"

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

            # Rest of the compute_surprisal implementation
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

            if context_length >= inputs.input_ids.shape[1]:  # TODO: figure out why do we -1
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
        self, contexts: list[list[str]], target_words: list[str], use_bos_only: bool = True, resume_path: Path = None
    ) -> pd.DataFrame:
        """Analyze surprisal across steps with optional neuron ablation."""

        surprisal_frame = (
            load_df(resume_path, "target_word") if resume_path and resume_path.is_file() else pd.DataFrame()
        )
        results = []

        for step in self.config.steps:
            try:
                model, tokenizer = self.load_model_for_step(step)
                surprisal_lst = []

                for word_contexts, target_word in zip(contexts, target_words):
                    for context_idx, context in enumerate(word_contexts):
                        # Determine if ablation is needed
                        ablated = self.step_ablations is not None and step in self.step_ablations

                        # Compute surprisal with appropriate ablation setting
                        surprisal = self.compute_surprisal(
                            model, tokenizer, context, target_word, use_bos_only=use_bos_only, ablated=ablated
                        )

                        if surprisal_frame.empty:
                            results.append(
                                {
                                    "target_word": target_word,
                                    "context_id": context_idx,
                                    "context": "BOS_ONLY" if use_bos_only else context,
                                    step: surprisal,
                                }
                            )
                        else:
                            surprisal_lst.append(surprisal)

                # Save intermediate results
                if surprisal_frame.empty and results:
                    surprisal_frame = pd.DataFrame(results)
                elif surprisal_lst:
                    surprisal_frame[step] = surprisal_lst

                if resume_path:
                    surprisal_frame.to_csv(resume_path, index=False)
                # Cleanup resources
                del model, tokenizer
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing step {step}: {str(e)}")
                continue

        return surprisal_frame

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.ablator:
            self.ablator.cleanup()


#######################################################
# Util func to load prompt
def load_eval(
    word_path, word_header: str = "word", BOS_only: bool = True, prompt_header=None
) -> tuple[list[str], list[list[str]]]:
    """Load word and context lists from a JSON file."""
    try:
        with word_path.open("r", encoding="utf-8") as f:
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
    logger.info(f"{len(target_words) - len(words)} words have no context!")
    return words, contexts


def load_df(file_path: Path, col_header: str) -> pd.DataFrame:
    "Load df from the given column."
    df = pd.read_csv(file_path)
    start_idx = df.columns.tolist().index(col_header)
    return df[start_idx:]


def load_neuron_dict(
    file_path: Path, key_col: str = "step", value_col: str = "top_neurons", random_base: bool = False, top_n=0
) -> dict[int, list[int]]:
    """Load a DataFrame and convert neuron values to integers."""
    df = pd.read_csv(file_path)

    result = {}
    for _, row in df.iterrows():
        try:
            # Parse the string to list of floats
            float_neurons = ast.literal_eval(row[value_col])
            # Extract the decimal part as integer; Converts '5.2021' format to 2021.
            neurons = [int(str(float(x)).split(".")[1]) for x in float_neurons]
            # generate the random indices excluded the
            if random_base:
                neurons = generate_neurons(neurons)
            if top_n!=0:
                neurons = neurons[:top_n]
            result[row[key_col]] = neurons
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing neuron list for step {row[key_col]}: {e}")
            result[row[key_col]] = []
    layer_num = [int(str(float(x)).split(".")[0]) for x in float_neurons][0]
    logger.info(f"Computing on {top_n} neurons")
    return result, layer_num


def generate_neurons(exclude_list: list[str], min_val: int = 1, max_val: int = 2047) -> list[int]:
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


def sel_eval(results_df: pd.DataFrame, eval_path: Path, result_dir: Path, filename):
    """Select the word subset from the eval file."""
    # load eval file
    eval_file = settings.PATH.dataset_root / eval_path
    result_file = result_dir / eval_file.stem / filename
    result_file.parent.mkdir(parents=True, exist_ok=True)
    eval_frame = pd.read_csv(eval_file)
    # select target words
    results_df_sel = results_df[results_df["target_word"].isin(eval_frame["word"])]
    results_df_sel.to_csv(result_file, index=False)
    logger.info(f"Eval set saved to: {result_file}")
