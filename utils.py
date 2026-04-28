import math
import random

DNA_ALPHABET = ("A", "C", "G", "T")
NEG_INF = -1e18  # used as a stand-in for impossible log-probabilities


def normalize_probs(probs):
    # Scale probabilities so they sum to 1
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Probability dictionary must sum to a positive value.")
    return {k: v / total for k, v in probs.items()}


CODON_TABLE_STOP = {"TAA", "TAG", "TGA"}
START_CODONS = {"ATG"}


def weighted_choice(items, probs, rng):
    # Sample one item based on cumulative probabilities
    r = rng.random()
    cumulative = 0.0
    for item, p in zip(items, probs):  # pairs each item with its probability
        cumulative += p
        if r <= cumulative:
            return item
    return items[-1]  # fallback in case of rounding issues


def sample_from_dict(prob_dict, rng):
    # Sample a key from a probability dictionary
    prob_dict = normalize_probs(prob_dict)
    items = list(prob_dict.keys())
    probs = list(prob_dict.values())
    return weighted_choice(items, probs, rng)


def sample_base(dist, rng):
    # Convenience wrapper for sampling a DNA base
    return sample_from_dict(dist, rng)


def argmax(items):
    # Return the (score, state) pair with the largest score
    best_score, best_state = items[0]
    for score, state in items[1:]:
        if score > best_score:
            best_score, best_state = score, state
    return best_score, best_state


def state_requires_pattern(pattern):
    # Returns a checker function that tests whether a pattern ends at a position
    def checker(position, sequence):
        if position < len(pattern) - 1:
            return False
        start = position - (len(pattern) - 1)
        return sequence[start : position + 1] == pattern  # slice includes end position
    return checker


def start_state_checker(position, sequence):
    # Check if the 3-base window ending here is a start codon
    if position < 2:
        return False
    return sequence[position - 2 : position + 1] == "ATG"


def stop_state_checker(position, sequence):
    # Check if the 3-base window ending here is a stop codon
    if position < 2:
        return False
    return sequence[position - 2 : position + 1] in CODON_TABLE_STOP