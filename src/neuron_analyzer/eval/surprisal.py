#!/usr/bin/env python
import logging
import random
import typing as t
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from neuron_analyzer.ablation.intervention import AblationConfig, NeuronAblator
from neuron_analyzer.load_util import load_df
from neuron_analyzer.model_util import StepConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")

######################################################
# Extract surprisal with different interventions


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
        """
        if not self.config.steps:
            raise ValueError("No steps provided for analysis")
        """

        if isinstance(self.model_cache_dir, str):
            self.model_cache_dir = Path(self.model_cache_dir)

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
            logger.error(f"Error loading model for step {step}: {e!s}")
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
            logger.error(f"Error in compute_surprisal: {e!s}")
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

                for word_contexts, target_word in zip(contexts, target_words, strict=False):
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
                logger.error(f"Error processing step {step}: {e!s}")
                continue

        return surprisal_frame

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.ablator:
            self.ablator.cleanup()
