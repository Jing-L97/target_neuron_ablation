import math

#######################################################
# Manual regime boundary annotation

neuron_dict = {
    "gpt2-medium": 90,
    "gpt2-large": 35,
    "gpt2-xl": 30,
    "pythia-1B": 50,
    "pythia-1.4B": 90,
    "pythia-2.8B": 50,
}

regime_dict = {}


def get_plateau(args: dict, neuron_dict=neuron_dict, regime_dict=regime_dict) -> dict:
    """Get plateua from manual labels."""
    args.top_n = neuron_dict[args.model]
    if args.regime:
        args.top_n = regime_dict[args.model]
        args.top_n = int(math.exp(args.top_n))
    return args
