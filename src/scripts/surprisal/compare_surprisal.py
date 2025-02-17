"""Extract word surprisal with comparative neuron ablation analysis."""

from pathlib import Path
import typing as t
from dataclasses import dataclass

import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import pandas as pd


@dataclass
class AblationConfig:
    """Configuration for neuron ablation."""
    layer_name: str = "gpt_neox.layers.5.mlp"  # Last MLP layer
    neurons: list[str] = None  # List of neuron indices to ablate
    k: int = 10  # Number of iterations for ablation analysis


class NeuronAblator:
    """Handles neuron ablation in transformer models."""

    def __init__(self, model: GPTNeoXForCausalLM, config: AblationConfig) -> None:
        """Initialize ablator with model and config.
        Args:
            model: The transformer model
            config: Ablation configuration
        """
        self.model = model
        self.config = config
        self.is_ablated = False
        self.hook_handle = None
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron ablation."""
        def ablation_hook(module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            """Forward hook to zero out specified neurons.
            Args:
                module: The PyTorch module
                input_tensor: Input tensor tuple
                output: Output tensor
            
            Returns:
                Modified output tensor with ablated neurons
            """
            if self.is_ablated and self.config.neurons:
                # Zero out activations for specified neurons
                for neuron_idx in self.config.neurons:
                    output[:, :, int(neuron_idx)] = 0
            return output

        # Get the MLP layer
        layer = dict(self.model.named_modules())[self.config.layer_name]
        
        # Register the forward hook
        self.hook_handle = layer.register_forward_hook(ablation_hook)

    def enable_ablation(self) -> None:
        """Enable neuron ablation."""
        self.is_ablated = True

    def disable_ablation(self) -> None:
        """Disable neuron ablation."""
        self.is_ablated = False

    def cleanup(self) -> None:
        """Remove the forward hook."""
        if self.hook_handle:
            self.hook_handle.remove()


class SurprisalExtractor:
    """Extracts word surprisal with comparative neuron ablation support."""

    def __init__(
        self, 
        model: GPTNeoXForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
        ablation_config: AblationConfig | None = None
    ) -> None:
        """Initialize the surprisal extractor.
        
        Args:
            model: Pre-trained transformer model
            tokenizer: Associated tokenizer
            ablation_config: Configuration for neuron ablation
        """
        self.model = model or self._load_default_model()
        self.tokenizer = tokenizer or self._load_default_tokenizer()
        
        if ablation_config:
            self.ablator = NeuronAblator(self.model, ablation_config)
        else:
            self.ablator = None
        
        self.model.eval()  # Set to evaluation mode
        
    def _load_default_model(self) -> GPTNeoXForCausalLM:
        """Load the default model if none provided."""
        cache_dir = Path("../../models/pythia-70m-deduped/step3000")
        return GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir=cache_dir
        )

    def _load_default_tokenizer(self) -> AutoTokenizer:
        """Load the default tokenizer if none provided."""
        cache_dir = Path("../../models/pythia-70m-deduped/step3000")
        return AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir=cache_dir
        )

    def compute_surprisal(self, context: str, target_word: str, ablated: bool = False) -> float:
        """Compute surprisal for a target word given a context.
        
        Args:
            context: The context string
            target_word: The target word to compute surprisal for
            ablated: Whether to compute with neuron ablation
            
        Returns:
            Computed surprisal value
        """
        # Set ablation state if ablator exists
        if self.ablator:
            if ablated:
                self.ablator.enable_ablation()
            else:
                self.ablator.disable_ablation()

        # Tokenize input
        inputs = self.tokenizer(context + target_word, return_tensors="pt")
        context_tokens = self.tokenizer(context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get the logits for the position after context
            context_length = context_tokens.input_ids.shape[1]
            target_logits = logits[0, context_length-1:context_length]
            
            # Get the actual token ID for the target word
            target_token_id = inputs.input_ids[0, context_length]
            
            # Compute log probability
            log_prob = torch.log_softmax(target_logits, dim=-1)[0, target_token_id]
            surprisal = -log_prob.item()
            
        return surprisal

    def compute_comparative_surprisal(
        self, 
        context: str, 
        target_word: str
    ) -> dict[str, float]:
        """Compute surprisal values before and after ablation.
        
        Args:
            context: The context string
            target_word: The target word to compute surprisal for
            
        Returns:
            Dictionary containing original and ablated surprisal values
        """
        # Compute original surprisal
        original_surprisal = self.compute_surprisal(context, target_word, ablated=False)
        
        # Compute ablated surprisal if ablator exists
        if self.ablator:
            ablated_surprisal = self.compute_surprisal(context, target_word, ablated=True)
        else:
            ablated_surprisal = None
            
        return {
            'original_surprisal': original_surprisal,
            'ablated_surprisal': ablated_surprisal,
            'surprisal_difference': ablated_surprisal - original_surprisal if ablated_surprisal is not None else None
        }

    def batch_comparative_extraction(
        self, 
        contexts: list[str], 
        target_words: list[str]
    ) -> pd.DataFrame:
        """Extract comparative surprisal values for multiple context-word pairs.
        
        Args:
            contexts: List of context strings
            target_words: List of target words
            
        Returns:
            DataFrame containing the comparative results
        """
        results = []
        for context, target in zip(contexts, target_words):
            surprisal_values = self.compute_comparative_surprisal(context, target)
            results.append({
                'context': context,
                'target_word': target,
                **surprisal_values
            })
        
        return pd.DataFrame(results)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.ablator:
            self.ablator.cleanup()


def main() -> None:
    """Main function demonstrating usage."""
    # TODO: integrate the freq extraction
    config = AblationConfig(
        neurons=['207', '427', '509', '130', '452'] 
    )
    
    # Initialize surprisal extractor with ablation
    extractor = SurprisalExtractor(ablation_config=config)
    
    try:
        # Example usage
        contexts = [
            "The cat sat on the",
            "She opened the book and started",
            "In the morning, he prepared",
        ]
        target_words = [
            "mat",
            "reading",
            "breakfast"
        ]
        
        # Extract comparative surprisal values
        results_df = extractor.batch_comparative_extraction(contexts, target_words)
        print("\nComparative Surprisal Analysis:")
        print(results_df)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(results_df[['original_surprisal', 'ablated_surprisal', 'surprisal_difference']].describe())
        
    finally:
        # Clean up resources
        extractor.cleanup()


if __name__ == "__main__":
    main()