"""Extract word surprisal across different training steps."""

from pathlib import Path
import typing as t
from dataclasses import dataclass
import logging
from datetime import datetime

import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    """Configuration for model steps."""
    model_name: str = "EleutherAI/pythia-70m-deduped"
    base_cache_dir: Path = Path("../../models/pythia-70m-deduped")
    
    def __post_init__(self):
        """Initialize steps after instance creation."""
        self.steps = generate_pythia_checkpoints()
        logger.info(f"Generated {len(self.steps)} checkpoint steps")

class StepSurprisalExtractor:
    """Extracts word surprisal across different training steps."""

    def __init__(self, config: StepConfig) -> None:
        """Initialize the step surprisal extractor.
        
        Args:
            config: Configuration for model steps
        """
        self.config = config

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.steps:
            raise ValueError("No steps provided for analysis")
        
        if isinstance(self.config.base_cache_dir, str):
            self.config.base_cache_dir = Path(self.config.base_cache_dir)

    def load_model_for_step(self, step: int) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
        """Load model and tokenizer for a specific step.
        
        Args:
            step: Training step number
            
        Returns:
            Tuple of (model, tokenizer)
        """
        cache_dir = self.config.base_cache_dir / f"step{step}"
        logger.info(f"Loading model for step {step}")
        
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                self.config.model_name,
                revision=f"step{step}",
                cache_dir=cache_dir
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                revision=f"step{step}",
                cache_dir=cache_dir
            )
            
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
        use_bos_only: bool = True
    ) -> float:
        """Compute surprisal for a target word given a context."""
        if use_bos_only:
            # Get the BOS token and create input with just BOS + target word
            bos_token = tokenizer.bos_token
            input_text = bos_token + target_word
            context_length = 1  # BOS token length
        else:
            # Use provided context
            input_text = context + target_word
            context_tokens = tokenizer(context, return_tensors="pt")
            context_length = context_tokens.input_ids.shape[1]

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get logits for the position after BOS or context
            target_logits = logits[0, context_length-1:context_length]
            
            # Get the actual token ID for the target word
            target_token_id = inputs.input_ids[0, context_length]
            
            # Compute log probability
            log_prob = torch.log_softmax(target_logits, dim=-1)[0, target_token_id]
            surprisal = -log_prob.item()
            
        return surprisal

    def analyze_steps(
        self, 
        contexts: list[str], 
        target_words: list[str],
        use_bos_only: bool = True  # New parameter
    ) -> pd.DataFrame:
        """Analyze surprisal across all steps for given words."""
        results = []
        
        for step in self.config.steps:
            logger.info(f"Processing step {step}")
            
            try:
                model, tokenizer = self.load_model_for_step(step)
                
                for context, target in zip(contexts, target_words):
                    surprisal = self.compute_surprisal(
                        model, 
                        tokenizer, 
                        context, 
                        target,
                        use_bos_only=use_bos_only
                    )
                    
                    results.append({
                        'step': step,
                        'context': 'BOS_ONLY' if use_bos_only else context,
                        'target_word': target,
                        'surprisal': surprisal
                    })
                
                # Clean up
                del model
                del tokenizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing step {step}: {str(e)}")
                continue
        
        return pd.DataFrame(results)

def main() -> None:
    """Main function demonstrating usage."""
    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig()
    
    # Load target words
    CDI_path = "/scratch1/projects/lexical-benchmark/v2/datasets/wordstats/matched/EN/1/cdi_childes.csv"
    target_frame = pd.read_csv(CDI_path)
    target_words = target_frame["word"].to_list()
    contexts = [""] * len(target_words)
    
    # Initialize extractor
    extractor = StepSurprisalExtractor(steps_config)
    
    try:
        # Analyze steps with BOS only
        results_df = extractor.analyze_steps(
            contexts=contexts,
            target_words=target_words,
            use_bos_only=True
        )
        
        # Save results even if some checkpoints failed
        if not results_df.empty:
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"pythia_surprisal_analysis_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            
            # Save summary statistics
            summary_file = output_dir / f"pythia_surprisal_summary_{timestamp}.csv"
            summary = results_df.groupby(['step', 'target_word'])['surprisal'].describe()
            summary.to_csv(summary_file)
            
            logger.info(f"Results saved to: {output_file}")
            logger.info(f"Processed {len(results_df['step'].unique())} checkpoints successfully")
        else:
            logger.warning("No results were generated")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()