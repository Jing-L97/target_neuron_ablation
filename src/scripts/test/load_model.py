# %%
import sys

sys.path.append("../")
import logging
import typing as t
from warnings import simplefilter

import pandas as pd
import transformer_lens
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM

from neuron_analyzer import settings

T = t.TypeVar("T")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


def tl_name_to_hf_name(model_name):
    hf_model_name = transformer_lens.loading_from_pretrained.get_official_model_name(model_name)
    return hf_model_name


def load_model_from_tl_name_single(model_name, device="cuda", cache_dir=None, hf_token=None):
    """Load single models from transformer lens."""
    hf_model_name = tl_name_to_hf_name(model_name)

    # loading tokenizer
    if "qwen" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            pad_token="<|extra_0|>",
            eos_token="<|endoftext|>",
            cache_dir=cache_dir,
        )
        # following the example given in their github repo: https://github.com/QwenLM/Qwen
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, trust_remote_code=True, cache_dir=cache_dir, token=hf_token
        )

    # loading model
    if "llama" in model_name.lower() or "gemma" in model_name.lower() or "mistral" in model_name.lower():
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=hf_token, cache_dir=cache_dir)
        model = HookedTransformer.from_pretrained(
            model_name=model_name, hf_model=hf_model, tokenizer=tokenizer, device=device, cache_dir=cache_dir
        )
    else:
        model = HookedTransformer.from_pretrained(model_name, device=device, cache_dir=cache_dir, token=hf_token)

    return model, tokenizer


def load_pythia_steps(
    model_name, device="cuda", step=None, cache_dir=None, hf_token=None
) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for a specific step."""
    if "pythia" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=cache_dir / model_name / f"step{step}"
        )

        hf_model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=cache_dir / model_name / f"step{step}"
        )
        logger.info(f"import HF model from {cache_dir}/{model_name}/step{step}")
        model = HookedTransformer.from_pretrained(
            model_name=model_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            cache_dir=cache_dir / model_name / f"step{step}",
        )
        logger.info(f"import hooked model from {cache_dir}/{model_name}/step{step}")
        return model, tokenizer
    return None


def load_model_from_tl_name(model_name, device="cuda", step=None, cache_dir=None, hf_token=None):
    """Load models for for ablation experiments."""
    if "pythia" in model_name.lower():
        logger.info
        return load_pythia_steps(model_name, device=device, step=step, cache_dir=cache_dir, hf_token=hf_token)
    logger.info("Load non-pythia model")
    return load_model_from_tl_name_single(model_name, device=device, cache_dir=cache_dir, hf_token=hf_token)


model_name = "gpt2"

load_model_from_tl_name(model_name, device="cuda", cache_dir=None, hf_token=None, step=None)


class ModelHandler:
    def load_model_and_tokenizer(
        self,
        model_name: str,
        hf_token_path: str,
        device: str,
        step=None,
    ):
        """Load model and tokenizer for processing."""
        # Load HF token

        model, self.tokenizer = load_model_from_tl_name(
            model_name, device, step=step, cache_dir=settings.PATH.model_dir, hf_token=None
        )
        self.model = model.to(device)
        self.model.eval()
        return self.model, self.tokenizer


model_handler = ModelHandler()


# Load model and tokenizer for specific step
model, tokenizer = model_handler.load_model_and_tokenizer(
    step=None,
    model_name=model_name,
    hf_token_path=settings.PATH.unigram_dir / "hf_token.txt",
    device="cuda",
)
logger.info("######################")
logger.info("Model has been loaded")
