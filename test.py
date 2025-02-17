
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from neuron_analyzer import settings

def load_model_from_tl_name(
    model_name, device='cuda', step=None,cache_dir=None, hf_token=None
    ) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for a specific step."""
    if "pythia" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=cache_dir/model_name/f"step{step}"
        )

        hf_model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=cache_dir/model_name/f"step{step}"
        )
        print("HF model has been loaded")
        print("#################################")
        model = HookedTransformer.from_pretrained(model_name=model_name, hf_model=hf_model, 
            tokenizer=tokenizer, device=device, cache_dir=cache_dir/model_name/f"step{step}")

        return model, tokenizer
    return None


model_name = "EleutherAI/pythia-70m-deduped"
step = 0
cache_dir=settings.PATH.model_dir/model_name/f"step{step}"

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=cache_dir
        )

hf_model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=cache_dir
        )
print("Model has been laoded")


model = HookedTransformer.from_pretrained(model_name=model_name, hf_model=hf_model, 
            tokenizer=tokenizer, device=device, cache_dir=cache_dir)

            

print("#################################")
print("The hooked model has been loaded")