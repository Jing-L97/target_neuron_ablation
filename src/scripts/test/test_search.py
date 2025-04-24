import argparse
import logging

from neuron_analyzer.selection.group_backup import NeuronGroupSearch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument("--interval", type=int, default=20, help="Checkpoint intervals")
    parser.add_argument("--start", type=int, default=0, help="Start index of step range")
    parser.add_argument("--end", type=int, default=141, help="End index of step range")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


def test_neuron_search():
    """Test the NeuronGroupSearch class with a simple example."""

    # Define a simple evaluation function
    def evaluate_neurons(neurons: list[int]) -> float:
        # This is a dummy evaluation function
        # In a real case, this would compute the delta loss
        # Here we'll just return a value based on the sum of neuron IDs
        return sum(neurons) / 100

    # load neuron indices and selta loss from the feather file
    neurons = list(range(100))
    # load from the
    individual_delta_loss = list(range(100))
    # Initialize the search
    search = NeuronGroupSearch(neurons=neurons, evaluation_fn=evaluate_neurons, target_size=10)

    # Get the best result using all methods
    best_method, result = search.get_best_result()

    print(f"Best method: {best_method}")
    print(f"Best neurons: {result.neurons}")
    print(f"Delta loss: {result.delta_loss}")

    # You can also run individual methods
    beam_result = search.progressive_beam_search(beam_width=5)
    print(f"\nBeam search result: {beam_result}")


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    # Initialize configuration with all Pythia checkpoints
    test_neuron_search()


if __name__ == "__main__":
    main()
