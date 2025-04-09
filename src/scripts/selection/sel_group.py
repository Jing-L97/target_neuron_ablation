#!/usr/bin/env python
import argparse
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from neuron_analyzer import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", choices=["mean", "longtail"], default="longtail")
    parser.add_argument(
        "--effect", type=str, choices=["boost", "suppress"], default="suppress", help="boost or suppress long-tail prob"
    )
    parser.add_argument("--top_n", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


# TODO: cache the group activation as we want to save computation


@dataclass
class SearchResult:
    """Result of a neuron group search."""

    neurons: List[int]
    mediation: float
    kl_divergence: float

    def __repr__(self) -> str:
        return f"SearchResult(neurons={self.neurons}, mediation={self.mediation:.4f}, kl={self.kl_divergence:.4f})"


class NeuronGroupSearch:
    """Class for finding optimal neuron groups using various search strategies."""

    def __init__(
        self,
        neurons: List[int],
        evaluation_fn: Callable[[List[int]], Tuple[float, float]],
        target_size: int,
        individual_mediation: Optional[List[float]] = None,
        individual_kl: Optional[List[float]] = None,
    ):
        """Initialize the neuron group search."""
        self.neurons = neurons
        self.evaluation_fn = evaluation_fn
        self.target_size = min(target_size, len(neurons))

        # Compute individual scores if not provided
        if individual_mediation is None or individual_kl is None:
            self._compute_individual_scores()
        else:
            self.individual_mediation = individual_mediation
            self.individual_kl = individual_kl

        # Derived information
        self.composite_scores = [
            self.individual_mediation[i] + 0.01 * self.individual_kl[i] for i in range(len(self.neurons))
        ]

    def _compute_individual_scores(self) -> None:
        """Compute mediation and KL scores for individual neurons."""
        self.individual_mediation = []
        self.individual_kl = []

        for i, neuron in enumerate(self.neurons):
            mediation, kl = self.evaluation_fn([neuron])
            self.individual_mediation.append(mediation)
            self.individual_kl.append(kl)

    def _evaluate_group(self, group: List[int]) -> Tuple[float, float]:
        """Evaluate a neuron group using the evaluation function."""
        return self.evaluation_fn(group)

    def _get_sorted_neurons(self) -> List[Tuple[int, float]]:
        """Get neurons sorted by their composite scores (mediation prioritized)."""
        scores = [(i, self.composite_scores[i]) for i in range(len(self.neurons))]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def progressive_beam_search(self, beam_width: int = 20) -> SearchResult:
        """Find the best neuron group using progressive beam search."""
        # Get individual neuron scores
        sorted_indices = self._get_sorted_neurons()

        # Initialize beam with top individual neurons
        current_beam = []
        for i in range(min(beam_width, len(sorted_indices))):
            idx, _ = sorted_indices[i]
            neuron = self.neurons[idx]
            mediation, kl = self.individual_mediation[idx], self.individual_kl[idx]
            current_beam.append(({neuron}, mediation, kl))

        # Expand to larger groups
        for size in range(2, self.target_size + 1):
            candidates = []

            for group, _, _ in current_beam:
                # Try adding each neuron not already in the group
                for idx, _ in sorted_indices:
                    neuron = self.neurons[idx]
                    if neuron not in group:
                        new_group = group.union({neuron})

                        # Skip if already evaluated
                        if any(g == new_group for g, _, _ in candidates):
                            continue

                        # Evaluate the new group
                        mediation, kl = self._evaluate_group(list(new_group))
                        candidates.append((new_group, mediation, kl))

            # Sort candidates by mediation first, then KL
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

            # Keep top beam_width candidates
            current_beam = candidates[:beam_width]

        # Return the best group of the target size
        best_group = max(current_beam, key=lambda x: (x[1], x[2]))
        return SearchResult(neurons=list(best_group[0]), mediation=best_group[1], kl_divergence=best_group[2])

    def hierarchical_cluster_search(self, n_clusters: int = 5, expansion_factor: int = 3) -> SearchResult:
        """Find the best neuron group using hierarchical clustering-based search."""
        # Create feature vectors for clustering using both heuristics
        features = np.array([(self.individual_mediation[i], self.individual_kl[i]) for i in range(len(self.neurons))])

        # Normalize features
        if features.std(axis=0).min() > 0:  # Avoid division by zero
            features = (features - features.mean(axis=0)) / features.std(axis=0)

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(self.neurons)))
        cluster_labels = clustering.fit_predict(features)

        # Group neurons by cluster
        clusters = defaultdict(list)
        for i, neuron_idx in enumerate(range(len(self.neurons))):
            clusters[cluster_labels[i]].append(neuron_idx)

        # Find representatives from each cluster (highest individual scores)
        representatives = []
        for cluster_id, cluster_neurons in clusters.items():
            # Sort neurons in this cluster by mediation and KL
            sorted_cluster = sorted(
                [
                    (self.neurons[idx], self.individual_mediation[idx], self.individual_kl[idx])
                    for idx in cluster_neurons
                ],
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )

            # Take top representatives from each cluster
            representatives.extend([n for n, _, _ in sorted_cluster[:expansion_factor]])

        # If we have fewer representatives than target_size, use all representatives
        if len(representatives) <= self.target_size:
            mediation, kl = self._evaluate_group(representatives)
            return SearchResult(neurons=representatives, mediation=mediation, kl_divergence=kl)

        # Try various combinations to find the best group
        # Start with greedy selection based on individual scores
        sorted_reps = sorted(
            [
                (n, self.individual_mediation[self.neurons.index(n)], self.individual_kl[self.neurons.index(n)])
                for n in representatives
            ],
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )

        # Start with top scoring neurons
        current_group = [n for n, _, _ in sorted_reps[: min(5, self.target_size)]]

        # Complete the group
        while len(current_group) < self.target_size:
            best_candidate = None
            best_candidate_scores = (-float("inf"), -float("inf"))

            for n, _, _ in [(n, m, k) for n, m, k in sorted_reps if n not in current_group]:
                test_group = current_group + [n]
                mediation, kl = self._evaluate_group(test_group)

                if (mediation, kl) > best_candidate_scores:
                    best_candidate = n
                    best_candidate_scores = (mediation, kl)

            if best_candidate is not None:
                current_group.append(best_candidate)
            else:
                break

        # Evaluate the final group
        mediation, kl = self._evaluate_group(current_group)
        best_group = current_group
        best_scores = (mediation, kl)

        # Refine the group with local search
        for _ in range(min(10, len(best_group))):
            for i, n in enumerate(best_group):
                # Try replacing each neuron in the group
                for candidate in [neuron for neuron in self.neurons if neuron not in best_group]:
                    test_group = [x for x in best_group if x != n] + [candidate]
                    mediation, kl = self._evaluate_group(test_group)

                    if (mediation, kl) > best_scores:
                        best_group = test_group
                        best_scores = (mediation, kl)

        return SearchResult(neurons=best_group, mediation=best_scores[0], kl_divergence=best_scores[1])

    def iterative_pruning(self) -> SearchResult:
        """Find the best neuron group using iterative pruning."""
        # Start with all neurons
        current_group = self.neurons.copy()

        # Iteratively remove neurons until we reach target_size
        while len(current_group) > self.target_size:
            worst_neuron = None
            best_remaining_scores = (-float("inf"), -float("inf"))

            # Try removing each neuron and keep the best resulting group
            for n in current_group:
                test_group = [x for x in current_group if x != n]
                mediation, kl = self._evaluate_group(test_group)

                # Determine if this is the best group after removal
                if (mediation, kl) > best_remaining_scores:
                    best_remaining_scores = (mediation, kl)
                    worst_neuron = n

            # Remove the worst neuron
            if worst_neuron is not None:
                current_group.remove(worst_neuron)
            else:
                # This shouldn't happen, but break to avoid infinite loop
                break

        # Evaluate final group
        mediation, kl = self._evaluate_group(current_group)

        return SearchResult(neurons=current_group, mediation=mediation, kl_divergence=kl)

    def importance_weighted_sampling(self, n_iterations: int = 100, learning_rate: float = 0.1) -> SearchResult:
        """Find the best neuron group using importance weighted sampling."""
        # Initialize weights from composite scores
        weights = np.array(self.composite_scores)

        # Ensure all weights are positive
        min_weight = min(weights)
        weights = weights - min_weight + 0.000001

        # Normalize to probabilities
        weights = weights / np.sum(weights)

        best_group = None
        best_scores = (-float("inf"), -float("inf"))

        for _ in range(n_iterations):
            # Sample neurons according to current weights
            if len(self.neurons) <= self.target_size:
                sampled_indices = np.arange(len(self.neurons))
            else:
                sampled_indices = np.random.choice(len(self.neurons), size=self.target_size, replace=False, p=weights)

            sampled_group = [self.neurons[idx] for idx in sampled_indices]

            # Evaluate group
            mediation, kl = self._evaluate_group(sampled_group)

            # Update best group
            if (mediation, kl) > best_scores:
                best_group = sampled_group
                best_scores = (mediation, kl)

                # Update weights for neurons in successful groups
                for idx in sampled_indices:
                    weights[idx] *= 1 + learning_rate

                # Re-normalize
                weights = weights / np.sum(weights)

        return SearchResult(neurons=best_group, mediation=best_scores[0], kl_divergence=best_scores[1])

    def hybrid_search(self, n_init: int = 3, n_refinement: int = 50) -> SearchResult:
        """ Find the best neuron group using a hybrid approach. """
        candidate_groups = []

        # 1. Start with top individual neurons (greedy)
        sorted_indices = self._get_sorted_neurons()
        top_neurons = [self.neurons[idx] for idx, _ in sorted_indices[: self.target_size]]
        candidate_groups.append(top_neurons)

        # 2. Start with random initialization (n_init different starts)
        for _ in range(n_init):
            random_group = random.sample(self.neurons, min(len(self.neurons), self.target_size))
            candidate_groups.append(random_group)

        # 3. Start with biased sampling based on individual scores
        for _ in range(n_init):
            # Create probability distribution based on composite scores
            probs = np.array(self.composite_scores)
            min_prob = min(probs)
            probs = probs - min_prob + 0.000001
            probs = probs / np.sum(probs)

            # Sample based on probabilities
            sampled_indices = np.random.choice(
                len(self.neurons), size=min(len(self.neurons), self.target_size), replace=False, p=probs
            )
            sampled_group = [self.neurons[idx] for idx in sampled_indices]
            candidate_groups.append(sampled_group)

        # Evaluate all candidate groups
        best_group = None
        best_scores = (-float("inf"), -float("inf"))

        for group in candidate_groups:
            mediation, kl = self._evaluate_group(group)
            if (mediation, kl) > best_scores:
                best_group = group
                best_scores = (mediation, kl)

        # Refine the best group
        current_group = best_group.copy()
        current_scores = best_scores

        for _ in range(n_refinement):
            improved = False

            # Try replacing each neuron in the group
            for i, current_neuron in enumerate(current_group):
                for candidate_neuron in [n for n in self.neurons if n not in current_group]:
                    # Create a new group with one neuron replaced
                    test_group = current_group.copy()
                    test_group[i] = candidate_neuron

                    # Evaluate the new group
                    mediation, kl = self._evaluate_group(test_group)

                    # Update if better
                    if (mediation, kl) > current_scores:
                        current_group = test_group
                        current_scores = (mediation, kl)
                        improved = True

            # Early stopping if no improvement
            if not improved:
                break

        return SearchResult(neurons=current_group, mediation=current_scores[0], kl_divergence=current_scores[1])

    def run_all_methods(self) -> Dict[str, SearchResult]:
        """Run all search methods and return the results."""
        results = {}

        # Run all methods
        results["progressive_beam"] = self.progressive_beam_search()
        results["hierarchical_cluster"] = self.hierarchical_cluster_search()
        results["iterative_pruning"] = self.iterative_pruning()
        results["importance_weighted"] = self.importance_weighted_sampling()
        results["hybrid"] = self.hybrid_search()

        return results

    def get_best_result(self) -> Tuple[str, SearchResult]:
        """Run all methods and return the best result."""
        results = self.run_all_methods()

        # Find the best method (prioritizing mediation, then KL)
        best_method = max(results.items(), key=lambda x: (x[1].mediation, x[1].kl_divergence))

        return best_method


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path = (
        settings.PATH.result_dir
        / "token_freq"
        / args.effect
        / args.vector
        / args.model
        / f"{args.data_range_end}_{args.top_n}.csv"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.is_file():
        logger.info(f"{save_path} already exists, skip!")
    else:
        neuron_df = pd.DataFrame()
        # check whether the target file has been created
        for step in abl_path.iterdir():
            feather_path = abl_path / str(step) / str(args.data_range_end) / f"k{args.k}.feather"
            frame = select_top_token_frequency_neurons(feather_path, args.top_n, step.name, args.effect)
            neuron_df = pd.concat([neuron_df, frame])
        # assign col headers
        neuron_df.to_csv(save_path)
        logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
