import math
import random
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List

import spacy

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
nlp.add_pipe("sentencizer")


def compute_log_freq_per_million(count: int, total: int) -> float:
    """Compute log10 frequency per million tokens."""
    freq_per_million = (count / total) * 1_000_000
    return math.log10(freq_per_million) if freq_per_million > 0 else float("-inf")


@dataclass
class ContextStats:
    """Statistics for a context."""

    count: int
    frequency: float
    log_freq_per_million: float
    subsequent_words: Counter


class NGramContextCollector:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.context_stats: defaultdict[str, ContextStats] = defaultdict(
            lambda: ContextStats(0, 0.0, float("-inf"), Counter())
        )
        self.word_counts = Counter()
        self.total_windows = 0

    def load_dataset():
        dataset = load_dataset(args.dataset, split=args.split)
    
        if 'text' in dataset.column_names:
            batch_size = 1000  # Adjust based on available memory
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]['text']
                batch_stats = None
                for text in batch:
                    text_stats = collector.process_text(text)
                    if batch_stats is None:
                        batch_stats = text_stats
                    else:
                        collector.merge_stats(text_stats)
                collector.merge_stats(batch_stats)
        else:
            print(f"Warning: 'text' column not found in the dataset. Available columns: {dataset.column_names}")



    def process_text(self, text: str) -> None:
        """Process text to collect n-gram contexts."""
        doc = nlp(text)
        sentences: List[str] = [sent.text.strip() for sent in doc.sents]

        for sentence in sentences:
            words = sentence.split()
            if len(words) <= self.window_size:
                continue

            for i in range(len(words) - self.window_size):
                context = tuple(words[i : i + self.window_size])
                next_word = words[i + self.window_size]
                context_key = " ".join(context)

                self.context_stats[context_key].count += 1
                self.context_stats[context_key].subsequent_words[next_word] += 1
                self.word_counts[next_word] += 1
                self.total_windows += 1

    def compute_frequencies(self) -> None:
        """Compute frequencies and log frequencies per million."""
        total_words = sum(self.word_counts.values())

        for stats in self.context_stats.values():
            stats.frequency = stats.count / self.total_windows
            stats.log_freq_per_million = compute_log_freq_per_million(stats.count, self.total_windows)

    def get_word_stats(self, word: str) -> dict[str, float]:
        """Get comprehensive statistics for a word."""
        total_words = sum(self.word_counts.values())
        count = self.word_counts[word]
        freq = count / total_words
        log_freq = compute_log_freq_per_million(count, total_words)

        return {"count": count, "frequency": freq, "log_freq_per_million": log_freq}

    def get_context_stats(self, context: str) -> dict[str, t.Any]:
        """Get comprehensive statistics for a context."""
        stats = self.context_stats[context]
        return {
            "count": stats.count,
            "frequency": stats.frequency,
            "log_freq_per_million": stats.log_freq_per_million,
            "subsequent_words": dict(stats.subsequent_words),
        }

    def get_all_ngram_stats(self) -> dict[str, dict]:
        """Get complete statistics for all n-grams."""
        total_words = sum(self.word_counts.values())

        ngram_stats = {
            "context_stats": {
                context: {
                    "count": stats.count,
                    "frequency": stats.frequency,
                    "log_freq_per_million": stats.log_freq_per_million,
                    "subsequent_words": {
                        word: {
                            "count": count,
                            "frequency": count / stats.count,
                            "log_freq_per_million": compute_log_freq_per_million(count, stats.count)
                        }
                        for word, count in stats.subsequent_words.items()
                    }
                }
                for context, stats in self.context_stats.items()
            },
            "word_stats": {
                word: {
                    "count": count,
                    "frequency": count / total_words,
                    "log_freq_per_million": compute_log_freq_per_million(count, total_words)
                }
                for word, count in self.word_counts.items()
            },
            "metadata": {
                "corpus_stats": {
                    "total_windows": self.total_windows,
                    "total_words": total_words,
                    "unique_contexts": len(self.context_stats),
                    "unique_words": len(self.word_counts),
                    "window_size": self.window_size
                }
            }
        }
        return ngram_stats

    @staticmethod
    def filter_contexts(
        ngram_stats: dict, 
        target_words: list[str], 
        n_contexts: int,
        min_context_freq: float, 
        max_context_freq: float,
        mode:str  # random ot topk
    ) -> dict[str, list[dict[str, t.Any]]]:
        """Filter contexts for target words from pre-computed n-gram statistics."""
        selected_data: dict[str, list[dict[str, t.Any]]] = {}

        for word in target_words:
            word_contexts = []
            for context_key, stats in ngram_stats["context_stats"].items():
                if (
                    word in stats["subsequent_words"]
                    and min_context_freq <= stats["log_freq_per_million"] <= max_context_freq
                ):
                    word_count = stats["subsequent_words"][word]["count"]
                    word_freq = stats["subsequent_words"][word]["frequency"]
                    word_log_freq = stats["subsequent_words"][word]["log_freq_per_million"]
                    word_contexts.append({
                        "context": context_key,
                        "word_in_context_stats": {
                            "count": word_count,
                            "frequency": word_freq,
                            "log_freq_per_million": word_log_freq,
                        }
                    })

            # Sort contexts by frequency (descending order) and select top n_contexts
            sorted_contexts = sorted(word_contexts, key=lambda x: x["word_in_context_stats"]["count"], reverse=True)
            n_select = min(n_contexts, len(sorted_contexts))
            selected = random.sample(sorted_contexts, n_select) if mode == "random" else sorted_contexts[:n_contexts]
            selected_data[word] = selected

        return selected_data

