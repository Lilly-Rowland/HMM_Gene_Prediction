from hmm import HMMModel
from data_generation import INTERGENIC_DIST, CODING_DIST, INTRON_DIST, START_DIST, STOP_DIST
from utils import state_requires_pattern, start_state_checker, stop_state_checker


def build_model1():
    # Simplest model: just coding vs noncoding
    states = ["N", "C"]
    return HMMModel(
        name="Model 1: 2-state coding/noncoding",
        states=states,
        start_probs={"N": 0.7, "C": 0.3},
        transition_probs={
            "N": {"N": 0.96, "C": 0.04},
            "C": {"C": 0.96, "N": 0.04},
        },
        emission_probs={
            "N": INTERGENIC_DIST,
            "C": CODING_DIST,
        },
    )


def build_model2():
    # Adds explicit start/exon/stop states
    states = ["N", "START", "EXON", "STOP"]
    return HMMModel(
        name="Model 2: explicit start/stop boundaries",
        states=states,
        start_probs={"N": 0.96, "START": 0.01, "EXON": 0.02, "STOP": 0.01},
        transition_probs={
            "N": {"N": 0.965, "START": 0.03, "EXON": 0.003, "STOP": 0.002},
            "START": {"N": 0.002, "START": 0.60, "EXON": 0.395, "STOP": 0.003},
            "EXON": {"N": 0.002, "START": 0.001, "EXON": 0.975, "STOP": 0.022},
            "STOP": {"N": 0.93, "START": 0.01, "EXON": 0.002, "STOP": 0.058},
        },
        emission_probs={
            "N": INTERGENIC_DIST,
            "START": START_DIST,
            "EXON": CODING_DIST,
            "STOP": STOP_DIST,
        },
        allowed_state_for_base={
            # Restrict these states to valid sequence patterns
            "START": start_state_checker,
            "STOP": stop_state_checker,
        },
    )


def build_model3():
    # Splice-aware model with exon/intron structure
    states = ["I", "START", "EXON", "DONOR", "INTRON", "ACCEPTOR", "STOP"]
    return HMMModel(
        name="Model 3: splice-aware exon-intron HMM",
        states=states,
        start_probs={
            "I": 0.93,
            "START": 0.01,
            "EXON": 0.02,
            "DONOR": 0.01,
            "INTRON": 0.01,
            "ACCEPTOR": 0.01,
            "STOP": 0.01,
        },
        transition_probs={
            "I": {"I": 0.955, "START": 0.038, "EXON": 0.002, "DONOR": 0.001, "INTRON": 0.001, "ACCEPTOR": 0.001, "STOP": 0.002},
            "START": {"I": 0.001, "START": 0.55, "EXON": 0.438, "DONOR": 0.002, "INTRON": 0.001, "ACCEPTOR": 0.001, "STOP": 0.007},
            "EXON": {"I": 0.001, "START": 0.001, "EXON": 0.955, "DONOR": 0.025, "INTRON": 0.001, "ACCEPTOR": 0.001, "STOP": 0.016},
            "DONOR": {"I": 0.001, "START": 0.001, "EXON": 0.001, "DONOR": 0.12, "INTRON": 0.87, "ACCEPTOR": 0.003, "STOP": 0.004},
            "INTRON": {"I": 0.001, "START": 0.001, "EXON": 0.001, "DONOR": 0.001, "INTRON": 0.955, "ACCEPTOR": 0.04, "STOP": 0.001},
            "ACCEPTOR": {"I": 0.001, "START": 0.001, "EXON": 0.98, "DONOR": 0.001, "INTRON": 0.001, "ACCEPTOR": 0.01, "STOP": 0.006},
            "STOP": {"I": 0.96, "START": 0.005, "EXON": 0.002, "DONOR": 0.001, "INTRON": 0.001, "ACCEPTOR": 0.001, "STOP": 0.03},
        },
        emission_probs={
            "I": INTERGENIC_DIST,
            "START": START_DIST,
            "EXON": CODING_DIST,
            "DONOR": {"A": 0.10, "C": 0.05, "G": 0.70, "T": 0.15},
            "INTRON": INTRON_DIST,
            "ACCEPTOR": {"A": 0.65, "C": 0.05, "G": 0.20, "T": 0.10},
            "STOP": STOP_DIST,
        },
        allowed_state_for_base={
            # Enforce biological sequence patterns for special states
            "START": start_state_checker,
            "DONOR": state_requires_pattern("GT"),
            "ACCEPTOR": state_requires_pattern("AG"),
            "STOP": stop_state_checker,
        },
    )


def build_model4():
    # Coding model that tracks codon position (C1, C2, C3)
    states = ["I", "START", "C1", "C2", "C3", "STOP"]
    return HMMModel(
        name="Model 4: codon-aware periodic HMM",
        states=states,
        start_probs={"I": 0.95, "START": 0.01, "C1": 0.015, "C2": 0.01, "C3": 0.005, "STOP": 0.01},
        transition_probs={
            "I": {"I": 0.968, "START": 0.026, "C1": 0.002, "C2": 0.001, "C3": 0.001, "STOP": 0.002},
            "START": {"I": 0.001, "START": 0.60, "C1": 0.395, "C2": 0.001, "C3": 0.001, "STOP": 0.002},
            "C1": {"I": 0.001, "START": 0.001, "C1": 0.001, "C2": 0.985, "C3": 0.001, "STOP": 0.011},
            "C2": {"I": 0.001, "START": 0.001, "C1": 0.001, "C2": 0.001, "C3": 0.995, "STOP": 0.001},
            "C3": {"I": 0.001, "START": 0.001, "C1": 0.985, "C2": 0.001, "C3": 0.001, "STOP": 0.011},
            "STOP": {"I": 0.95, "START": 0.005, "C1": 0.002, "C2": 0.001, "C3": 0.001, "STOP": 0.041},
        },
        emission_probs={
            "I": INTERGENIC_DIST,
            "START": START_DIST,
            "C1": {"A": 0.15, "C": 0.35, "G": 0.30, "T": 0.20},
            "C2": {"A": 0.20, "C": 0.25, "G": 0.35, "T": 0.20},
            "C3": {"A": 0.25, "C": 0.30, "G": 0.20, "T": 0.25},
            "STOP": STOP_DIST,
        },
        allowed_state_for_base={
            "START": start_state_checker,
            "STOP": stop_state_checker,
        },
    )