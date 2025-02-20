#!/usr/bin/env python

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


#######################################################
# Extract surprisal by different steps
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


class StepSurprisalExtractor:
    """Extracts word surprisal across different training steps."""

    def __init__(
        self,
        config: StepConfig,
        model_name: str,
        model_cache_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize the surprisal extractor."""
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.config = config
        self.device = device
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

    def load_model_for_step(self, step: int) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
        """Load model and tokenizer for a specific step."""
        cache_dir = self.model_cache_dir / f"step{step}"
        logger.info(f"Loading model for step {step} on {self.device}")

        try:
            model = GPTNeoXForCausalLM.from_pretrained(self.model_name, revision=f"step{step}", cache_dir=cache_dir)
            model = model.to(self.device)  # Move model to specified device

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=f"step{step}", cache_dir=cache_dir)

            model.eval()
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
    ) -> float:
        """Compute surprisal for a target word given a context."""
        if use_bos_only:
            bos_token = tokenizer.bos_token
            input_text = bos_token + target_word
            context_length = 1  # BOS token length
        else:
            input_text = context + target_word
            context_tokens = tokenizer(context, return_tensors="pt").to(self.device)  # Move to device
            context_length = context_tokens.input_ids.shape[1]

        # Tokenize input and move to correct device
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            target_logits = logits[0, context_length - 1 : context_length]
            target_token_id = inputs.input_ids[0, context_length]

            log_prob = torch.log_softmax(target_logits, dim=-1)[0, target_token_id]
            surprisal = -log_prob.item()

        return surprisal

    def analyze_steps(
        self,
        contexts: list[list[str]],
        target_words: list[str],
        use_bos_only: bool = True,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """Analyze surprisal across all steps for given words and their contexts."""

        results = []

        for step in self.config.steps:
            logger.info(f"Processing step {step}")
            try:
                model, tokenizer = self.load_model_for_step(step)

                # Process each word and its contexts
                for word_contexts, target_word in zip(contexts, target_words):
                    # Process each context for the current word
                    for context_idx, context in enumerate(word_contexts):
                        surprisal = self.compute_surprisal_for_context(
                            model, tokenizer, context, target_word, use_bos_only=use_bos_only
                        )

                        results.append(
                            {
                                "step": step,
                                "target_word": target_word,
                                "context_id": context_idx,
                                "context": "BOS_ONLY" if use_bos_only else context,
                                "surprisal": surprisal,
                            }
                        )

                # Clean up GPU memory
                del model
                del tokenizer
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing step {step}: {str(e)}")
                continue

        return pd.DataFrame(results)


#######################################################
# Util func to load prompt


def load_eval(
    word_path: Path | str, word_header: str = "word", BOS_only: bool = True, prompt_header: str | None = None
) -> tuple[list[str], list[str]]:
    """Load word and context lists from a JSON file."""

    with word_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract words and contexts
    target_words = list(data.keys())

    if BOS_only:
        contexts = [""] * len(target_words)
    else:
        contexts = [context["context"] for word in data.values() for context in word]
    return target_words, contexts
