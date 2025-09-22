import math


class ModelConfig:
    def __init__(self):
        # Manual step mode config
        self.step_dict = {
            "gpt2-medium": "single",
            "gpt2-large": "single",
            "gpt2-xl": "single",
            "EleutherAI/pythia-1B-deduped": "multi",
            "EleutherAI/pythia-1.4B-deduped": "multi",
            "EleutherAI/pythia-2.8B-deduped": "multi",
        }

        # Manual regime boundary annotation
        self.neuron_dict = {
            "gpt2-medium": 90,
            "gpt2-large": 35,
            "gpt2-xl": 30,
            "EleutherAI/pythia-1B-deduped": 50,
            "EleutherAI/pythia-1.4B-deduped": 90,
            "EleutherAI/pythia-2.8B-deduped": 50,
        }

        self.regime_dict = {}

        # Manual vector annotation
        self.vector_dict = {
            "gpt2-medium": "longtail_0_50",
            "gpt2-large": "longtail_50",
            "gpt2-xl": "longtail_50",
            "EleutherAI/pythia-1B-deduped": "longtail_0_50",
            "EleutherAI/pythia-1.4B-deduped": "longtail_0_50",
            "EleutherAI/pythia-2.8B-deduped": "longtail_0_50",
        }

    def config_args(self, args, top_n=True):
        """Apply all config rules to args."""
        args = self.get_step(args)
        args = self.get_vector(args)
        if top_n:
            args = self.get_plateau(args)
        return args

    def get_step(self, args):
        """Get step mode from manual labels."""
        if args.model in self.step_dict:
            args.step_mode = self.step_dict[args.model]
        return args

    def get_plateau(self, args):
        """Get plateau from manual labels."""
        if args.model in self.neuron_dict:
            args.top_n = self.neuron_dict[args.model]
        if getattr(args, "regime", None) and args.model in self.regime_dict:
            args.top_n = self.regime_dict[args.model]
            args.top_n = int(math.exp(args.top_n))
        return args

    def get_vector(self, args):
        """Get vector from manual labels."""
        if args.model in self.vector_dict:
            args.vector = self.vector_dict[args.model]
        return args
