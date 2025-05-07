#!/usr/bin/env python3
"""Fine-tuning script for decoder-only transformer language models.
Loads text files as training data and fine-tunes a pre-trained model.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier (defaults to model_name_or_path)"},
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use one of the fast tokenizer implementations"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Floating-point format in which the model weights should be initialized and trained"},
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and evaluation."""

    train_file: str = field(default=None, metadata={"help": "Path to training file (.txt)"})
    train_dir: str = field(default=None, metadata={"help": "Directory containing training files (.txt)"})
    validation_file: str = field(default=None, metadata={"help": "Path to validation file (.txt)"})
    validation_dir: str = field(default=None, metadata={"help": "Directory containing validation files (.txt)"})
    max_seq_length: int = field(default=1024, metadata={"help": "Maximum sequence length that the model might handle"})
    preprocessing_num_workers: int = field(default=None, metadata={"help": "Number of processes for preprocessing"})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_only: bool = field(
        default=False, metadata={"help": "Only run the preprocessing script to be cached for future use"}
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when processing input files"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Custom training arguments."""

    report_to: str = field(
        default="tensorboard", metadata={"help": "The integration to report the results and logs to."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to save memory at the expense of slower backward pass."},
    )


class TextDataset(Dataset):
    """Dataset for loading text files for causal language modeling."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        file_paths: list[str | Path],
        block_size: int,
        keep_linebreaks: bool = True,
    ):
        self.tokenizer = tokenizer
        self.file_paths = [Path(path) for path in file_paths]
        self.block_size = block_size
        self.keep_linebreaks = keep_linebreaks

        # Load and tokenize all the texts
        logger.info(f"Loading and tokenizing {len(file_paths)} text files...")
        self.examples = self._load_and_tokenize()
        logger.info(f"Created {len(self.examples)} training examples of size {block_size}")

    def _load_and_tokenize(self) -> list[torch.Tensor]:
        """Load all text files and tokenize them into block-sized chunks."""
        tokenized_examples = []

        for file_path in self.file_paths:
            try:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                # Optional processing
                if not self.keep_linebreaks:
                    text = text.replace("\n", " ").replace("  ", " ")

                # Tokenize the text
                tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
                input_ids = tokenized_text.input_ids[0]

                # Create examples of block_size
                for i in range(0, len(input_ids) - self.block_size + 1, self.block_size):
                    tokenized_examples.append(input_ids[i : i + self.block_size])

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")

        return tokenized_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return {"input_ids": self.examples[idx], "labels": self.examples[idx]}


def get_file_paths(path: str | Path) -> list[Path]:
    """Get all .txt file paths from a file or directory."""
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() == ".txt":
            return [path]
        logger.warning(f"Ignoring non-txt file: {path}")
        return []
    if path.is_dir():
        return list(path.glob("**/*.txt"))
    logger.warning(f"Path does not exist: {path}")
    return []


def setup_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    """Setup tokenizer for training by ensuring it has padding token, etc."""
    special_tokens_dict = {}

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            special_tokens_dict["pad_token"] = "[PAD]"

    # Add special tokens if needed
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def finetune_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    """Fine-tune the model with the given arguments."""
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer = setup_tokenizer(tokenizer)

    # Determine the torch dtype
    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", "float16", "float32"] else "auto"

    # Load model
    logger.info(f"Loading pretrained model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    # Collect text files
    train_files = []
    if data_args.train_file:
        train_files.extend(get_file_paths(data_args.train_file))
    if data_args.train_dir:
        train_files.extend(get_file_paths(data_args.train_dir))

    validation_files = []
    if data_args.validation_file:
        validation_files.extend(get_file_paths(data_args.validation_file))
    if data_args.validation_dir:
        validation_files.extend(get_file_paths(data_args.validation_dir))

    if not train_files:
        raise ValueError("No training files found. Please specify --train_file or --train_dir")

    logger.info(f"Found {len(train_files)} training files and {len(validation_files)} validation files")

    # Create datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_paths=train_files,
        block_size=data_args.max_seq_length,
        keep_linebreaks=data_args.keep_linebreaks,
    )

    validation_dataset = None
    if validation_files:
        validation_dataset = TextDataset(
            tokenizer=tokenizer,
            file_paths=validation_files,
            block_size=data_args.max_seq_length,
            keep_linebreaks=data_args.keep_linebreaks,
        )

    # Exit if we only want to preprocess the data
    if data_args.preprocessing_only:
        logger.info("Preprocessing completed. Exiting.")
        return

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    train_result = trainer.train()

    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    # Log and save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training completed!")


def test_function() -> None:
    """Example showing how to use the script."""
    # This is a simple test to show how to use the script
    # In practice, you would call this from the command line

    # For testing in a Python script:
    os.environ["WANDB_DISABLED"] = "true"

    model_args = ModelArguments(
        model_name_or_path="gpt2-small",
    )

    data_args = DataArguments(
        train_dir="./data/train_texts",
        validation_file="./data/valid/sample.txt",
        max_seq_length=128,  # Small for testing
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
    )

    finetune_model(model_args, data_args, training_args)
    print("Test fine-tuning completed!")


def main() -> None:
    """Parse arguments and run fine-tuning."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Validate arguments
    if not data_args.train_file and not data_args.train_dir:
        raise ValueError("Need either a training file or directory")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    logger.setLevel(logging.INFO if training_args.local_rank <= 0 else logging.WARNING)
    logger.info(f"Training/evaluation parameters: {training_args}")

    # Fine-tune model
    finetune_model(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

"""
# Example command:

python finetune_lm.py \
    --model_name_or_path /scratch2/jliu/lm_feedback/ckpts_ppo/best_reward \
    --train_dir /scratch2/jliu/BabyLM/babylm-train/babylm_data/babylm_100M/train.txt \
    --validation_file /scratch2/jliu/BabyLM/babylm-train/babylm_data/babylm_dev/dev.txt \
    --output_dir /scratch2/jliu/lm_feedback/ckpts_ppo/finetuned \
    --num_train_epochs 4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --save_steps 500 \
    --save_total_limit 2
    
    
"""
