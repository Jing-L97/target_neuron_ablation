# %%
import sys

sys.path.append("../")
import glob
import logging
import os
import typing as t
from warnings import simplefilter

import pandas as pd
import torch
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from hf_olmo.convert_olmo_to_hf import convert_checkpoint, maybe_unshard
from transformer_lens import HookedTransformer
from transformers import AutoConfig

from neuron_analyzer import settings

T = t.TypeVar("T")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


out_dir = "/scratch2/jliu/Generative_replay/neuron/models/allenai/OLMo-1B/step20000/converted"

maybe_unshard(out_dir)

convert_checkpoint(out_dir)


cache_dir = settings.PATH.model_dir
model_name = "allenai/OLMo-1B"
step = 20000


tokenizer = OLMoTokenizerFast.from_pretrained(model_name)


olmo = OLMoForCausalLM.from_pretrained(
    model_name, revision="step20000-tokens84B", cache_dir=cache_dir / model_name / f"step{step}"
)


olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B")

model = HookedTransformer.from_pretrained(
    model_name="allenai/OLMo-1B-hf",
    hf_model=olmo,
    tokenizer=tokenizer,
    device="cuda",
    # cache_dir=cache_dir / model_name / f"step{step}",
)


# convert the olmo ckpts into hf format

convert.maybe_unshard(local_checkpoint_dir)


def convert_hf_to_olmo_native(hf_model_dir, output_dir):
    """Convert Hugging Face model to OLMo native format"""
    os.makedirs(output_dir, exist_ok=True)

    # Load HF model
    hf_model = OLMoForCausalLM.from_pretrained(hf_model_dir)

    # Extract OLMo model and save in native format
    olmo_model = hf_model.model

    # Create a state dict with proper prefix
    state_dict = olmo_model.state_dict()
    fixed_state_dict = {}
    for key, val in state_dict.items():
        # Remove the 'transformer.' prefix if present
        if key.startswith("transformer."):
            fixed_key = key
        else:
            fixed_key = f"transformer.{key}"
        fixed_state_dict[fixed_key] = val

    # Save the model.pt file
    torch.save(fixed_state_dict, os.path.join(output_dir, "model.pt"))

    # Extract config and save as YAML
    config_dict = olmo_model.config.to_dict()
    config = {"model": config_dict}

    # Save config.yaml
    import yaml

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    return output_dir


def convert_hf_to_olmo_native(hf_model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Search for model.safetensors in the directory and its subdirectories
    def find_model_file(start_dir, target_files=["model.safetensors", "pytorch_model.bin"]):
        """Find the first occurrence of any target file in the directory tree"""
        for root, _, files in os.walk(start_dir):
            for file in files:
                if file in target_files:
                    return os.path.join(root, file)
        return None

    # Find model.safetensors or pytorch_model.bin
    model_file_path = find_model_file(hf_model_dir)

    if model_file_path:
        # Use parent directory of the found file
        actual_model_dir = os.path.dirname(model_file_path)
        print(f"Found model file at: {model_file_path}")
        print(f"Using model directory: {actual_model_dir}")
    else:
        # Fall back to original directory if no model file found
        actual_model_dir = hf_model_dir
        print(f"No model file found, using original directory: {actual_model_dir}")

    # Check if config.json exists in the model directory
    config_path = os.path.join(actual_model_dir, "config.json")
    if not os.path.exists(config_path):
        # Try to find config.json elsewhere
        config_files = glob.glob(os.path.join(hf_model_dir, "**/config.json"), recursive=True)
        if config_files:
            # Copy the first found config.json to the model directory
            import shutil

            shutil.copy(config_files[0], config_path)
            print(f"Copied config.json from {config_files[0]} to {config_path}")

    # Load HF model
    try:
        from hf_olmo import OLMoForCausalLM

        hf_model = OLMoForCausalLM.from_pretrained(actual_model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try alternate loading method
        print("Trying alternate loading method...")
        config = AutoConfig.from_pretrained(actual_model_dir)
        from hf_olmo import OLMoForCausalLM

        hf_model = OLMoForCausalLM.from_config(config)

        # Load state dict manually if needed
        if model_file_path.endswith(".safetensors"):
            from safetensors import safe_open

            state_dict = {}
            with safe_open(model_file_path, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            hf_model.load_state_dict(state_dict, strict=False)
        elif model_file_path.endswith(".bin"):
            state_dict = torch.load(model_file_path, map_location="cpu")
            hf_model.load_state_dict(state_dict, strict=False)

    # Extract OLMo model and save in native format
    olmo_model = hf_model.model

    # Create a state dict with proper prefix
    state_dict = olmo_model.state_dict()
    fixed_state_dict = {}
    for key, val in state_dict.items():
        # Ensure the proper prefix structure
        if key.startswith("transformer."):
            fixed_key = key
        else:
            fixed_key = f"transformer.{key}"
        fixed_state_dict[fixed_key] = val

    # Save the model.pt file
    torch.save(fixed_state_dict, os.path.join(output_dir, "model.pt"))

    # Create config dictionary for config.yaml - Using ModelConfig.asdict() instead of to_dict()
    try:
        # Try using asdict() method if it exists
        config_dict = olmo_model.config.asdict()
    except AttributeError:
        # If asdict() doesn't exist, directly create a dict from attributes
        print("ModelConfig doesn't have asdict() method, creating dict manually...")
        config_dict = {}
        # Try to get all the attributes from the config object
        for attr in dir(olmo_model.config):
            # Skip private/special attributes
            if not attr.startswith("_") and not callable(getattr(olmo_model.config, attr)):
                config_dict[attr] = getattr(olmo_model.config, attr)

    # Create the full config structure
    config = {"model": config_dict}

    # Save config.yaml
    try:
        import yaml

        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
    except Exception as e:
        print(f"Error saving config: {e}")
        # Alternative method to save config
        import json

        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            json.dump(config, f, indent=2)
        print("Saved config using JSON format")

    print(f"Model converted and saved to: {output_dir}")
    return output_dir


# Convert HF model to OLMo native format
olmo_root = "/scratch2/jliu/Generative_replay/neuron/models/allenai/OLMo-1B/step20000"
hf_model_path = f"{olmo_root}/models--allenai--OLMo-1B"
olmo_native_dir = f"{olmo_root}/converted"

output_dir = convert_hf_to_olmo_native(hf_model_path, olmo_native_dir)


def write_tokenizer(checkpoint_dir: str):
    """Write tokenizer files to the checkpoint directory, handling missing tokenizer config"""
    # First, check if the HF tokenizer files already exist
    if os.path.exists(os.path.join(checkpoint_dir, "tokenizer.json")):
        print("Tokenizer files already exist, loading directly...")
        tokenizer = OLMoTokenizerFast.from_pretrained(checkpoint_dir)
        return

    # If not, we need to create them
    try:
        # Try the original method first
        from olmo.tokenizer import Tokenizer

        tokenizer_raw = Tokenizer.from_checkpoint(checkpoint_dir)

        tokenizer = OLMoTokenizerFast(
            tokenizer_object=tokenizer_raw.base_tokenizer,
            truncation=tokenizer_raw.truncate_direction,
            max_length=tokenizer_raw.truncate_to,
            eos_token=tokenizer_raw.decode([tokenizer_raw.eos_token_id], skip_special_tokens=False),
        )
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        tokenizer.pad_token_id = tokenizer_raw.pad_token_id
        tokenizer.eos_token_id = tokenizer_raw.eos_token_id

    except Exception as e:
        print(f"Error loading tokenizer from checkpoint: {e}")
        print("Falling back to using the default OLMo tokenizer...")

        # Create a default OLMo tokenizer
        # Use a known good tokenizer ID for OLMo models
        tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B-hf")

        # Try to get model config to extract eos_token_id
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if "model" in config and "eos_token_id" in config["model"]:
                    tokenizer.eos_token_id = config["model"]["eos_token_id"]
            except Exception:
                print("Could not load eos_token_id from config.yaml")

    # Save the tokenizer
    print(f"Saving tokenizer to {checkpoint_dir}")
    tokenizer.save_pretrained(checkpoint_dir)

    # For completeness, add a tokenizer section to the config.yaml file
    try:
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if os.path.exists(config_path):
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Add tokenizer section if it doesn't exist
            if "tokenizer" not in config:
                config["tokenizer"] = {
                    "identifier": "allenai/OLMo-1B-hf",
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "truncate_direction": "right",
                    "truncate_to": 2048,
                }

                with open(config_path, "w") as f:
                    yaml.dump(config, f)
                print("Added tokenizer section to config.yaml")
    except Exception as e:
        print(f"Could not update config.yaml with tokenizer info: {e}")
