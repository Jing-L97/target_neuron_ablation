import sys
sys.path.append('../')
import logging
from warnings import simplefilter

import pandas as pd
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from neuron_analyzer import settings
from neuron_analyzer.ablations import load_model_from_tl_name

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

def check_model_dimensions(model: GPTNeoXForCausalLM):
    """Check dimensions of a HuggingFace GPTNeoX model."""
    config = model.config
    logger.info(f"Model dimensions:")
    logger.info(f"Number of layers: {config.num_hidden_layers}")
    logger.info(f"Layer dimension (d_mlp): {config.intermediate_size}")
    logger.info(f"Hidden dimension (d_model): {config.hidden_size}")
    logger.info(f"Number of attention heads: {config.num_attention_heads}")
    # For reference in neuron indexing
    logger.info(f"Maximum valid neuron index would be: {config.intermediate_size - 1}")

def verify_mlp_dimensions(model: GPTNeoXForCausalLM):
    """Verify MLP dimensions in GPTNeoX."""
    # Get the last layer MLP
    last_mlp = model.gpt_neox.layers[-1].mlp
    
    logger.info("MLP Component Dimensions:")
    # Input → 4x hidden (equivalent to W_in)
    logger.info(f"dense_h_to_4h shape: {last_mlp.dense_h_to_4h.weight.shape}")
    # 4x hidden → output (equivalent to W_out)
    logger.info(f"dense_4h_to_h shape: {last_mlp.dense_4h_to_h.weight.shape}")
    
    # Check if dimensions match what we expect
    h_to_4h = last_mlp.dense_h_to_4h.weight.shape
    _4h_to_h = last_mlp.dense_4h_to_h.weight.shape
    
    logger.info(f"\nInput dimension: {h_to_4h[1]}")        # Should be 512
    logger.info(f"Hidden dimension: {h_to_4h[0]}")         # Should be 2048
    logger.info(f"Output dimension: {_4h_to_h[0]}")        # Should be 512


model_name = "EleutherAI/pythia-70m-deduped"
device="cuda"
step=78000
cache_dir=settings.PATH.model_dir

tokenizer = AutoTokenizer.from_pretrained(
    model_name, revision=f"step{step}", cache_dir=cache_dir/model_name/f"step{step}"
        )
logger.info(f"Tokenizer has been loaded")


hf_model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=cache_dir/model_name/f"step{step}"
        )
logger.info(f"hf model has been loaded")

verify_mlp_dimensions(hf_model)



print("##################################")
model = HookedTransformer.from_pretrained(model_name=model_name, hf_model=hf_model, 
    tokenizer=tokenizer, device=device, cache_dir=cache_dir/model_name/f"step{step}")

logger.info(f"import hooked model from {cache_dir}/{model_name}/step{step}")
logger.info(f"Hooked model has been loaded!")
verify_mlp_dimensions(model)


