#!/usr/bin/env python
import json
import logging
import random
import typing as t
from pathlib import Path

import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.ablation.abl_util import get_pile_unigram_distribution

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")

#######################################################
# load and select eval set


def load_eval(
    word_path, word_header: str = "word", BOS_only: bool = True, prompt_header=None
) -> tuple[list[str], list[list[str]]]:
    """Load word and context lists from a JSON file."""
    try:
        with word_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {word_path}: {e!s}")
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
        words.append(word)
        for context_data in word_data:
            word_contexts.append(context_data["context"])
        contexts.append(word_contexts)
    logger.info(f"{len(target_words) - len(words)} words have no context!")
    return words, contexts


def load_df(file_path: Path, col_header: str) -> pd.DataFrame:
    """Load df from the given column."""
    df = pd.read_csv(file_path)
    start_idx = df.columns.tolist().index(col_header)
    return df[start_idx:]


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


#######################################################
# load unigram file


def load_unigram(model_name, device) -> torch.Tensor:
    """Load unigram distribution based on model type."""
    # Load unigram distribution
    if "pythia" in model_name:
        file_path = settings.PATH.unigram_dir / "pythia-unigrams.npy"
        unigram_distrib = get_pile_unigram_distribution(device=device, pad_to_match_W_U=True, file_path=file_path)
        logger.info(f"Loaded unigram freq from {file_path}")
    elif "gpt" in model_name:
        file_path = settings.PATH.unigram_dir / "gpt2-small-unigrams_openwebtext-2M_rows_500000.npy"
        unigram_distrib = get_pile_unigram_distribution(device=device, pad_to_match_W_U=False, file_path=file_path)
        logger.info(f"Loading unigram freq from {file_path}")
    else:
        raise Exception(f"No unigram distribution for {model_name}")

    return unigram_distrib
