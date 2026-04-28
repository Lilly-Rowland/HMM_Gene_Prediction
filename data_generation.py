import random
from utils import sample_base, CODON_TABLE_STOP, START_CODONS


class SequenceExample:
    def __init__(self, sequence, fine_labels, coarse_labels, organism_type):
        self.sequence = sequence
        self.fine_labels = fine_labels
        self.coarse_labels = coarse_labels
        self.organism_type = organism_type


# Base distributions used for different sequence regions
INTERGENIC_DIST = {"A": 0.30, "C": 0.20, "G": 0.20, "T": 0.30}
CODING_DIST = {"A": 0.20, "C": 0.30, "G": 0.30, "T": 0.20}
INTRON_DIST = {"A": 0.35, "C": 0.15, "G": 0.15, "T": 0.35}
START_DIST = {"A": 0.33, "C": 0.17, "G": 0.17, "T": 0.33}
STOP_DIST = {"A": 0.33, "C": 0.17, "G": 0.17, "T": 0.33}


def random_coding_codon(rng):
    # Keep generating codons until one is neither a stop nor start codon
    while True:
        codon = "".join(sample_base(CODING_DIST, rng) for _ in range(3))
        if codon not in CODON_TABLE_STOP and codon not in START_CODONS:
            return codon


def generate_prokaryote_sequence(
    length=300,
    intergenic_before=60,
    coding_codons=40,
    intergenic_after=60,
    rng=None,
):
    # Use provided RNG or create a new one
    rng = rng or random.Random()

    sequence_parts = []
    labels = []

    # Add intergenic region before the gene
    for _ in range(intergenic_before):
        sequence_parts.append(sample_base(INTERGENIC_DIST, rng))
        labels.append("I")

    start = "ATG"
    sequence_parts.extend(list(start))
    labels.extend(["START", "START", "START"])

    # Add coding region one codon at a time
    for _ in range(coding_codons):
        codon = random_coding_codon(rng)
        sequence_parts.extend(list(codon))
        labels.extend(["EXON", "EXON", "EXON"])

    # Pick a valid stop codon
    stop = rng.choice(sorted(CODON_TABLE_STOP))
    sequence_parts.extend(list(stop))
    labels.extend(["STOP", "STOP", "STOP"])

    # Add intergenic region after the gene
    for _ in range(intergenic_after):
        sequence_parts.append(sample_base(INTERGENIC_DIST, rng))
        labels.append("I")

    seq = "".join(sequence_parts)
    # Coarse labels collapse coding-related states into C, everything else into N
    coarse = ["C" if lab in {"START", "EXON", "STOP"} else "N" for lab in labels]
    return SequenceExample(seq, labels, coarse, organism_type="prokaryote")


def generate_eukaryote_sequence(
    intergenic_before=50,
    exon1_codons=12,
    intron_len=45,
    exon2_codons=15,
    intergenic_after=50,
    rng=None,
):
    # Use provided RNG or create a new one
    rng = rng or random.Random()

    seq = []
    labels = []

    # Add upstream intergenic region
    for _ in range(intergenic_before):
        seq.append(sample_base(INTERGENIC_DIST, rng))
        labels.append("I")

    start = "ATG"
    seq.extend(list(start))
    labels.extend(["START", "START", "START"])

    # First exon
    for _ in range(exon1_codons):
        codon = random_coding_codon(rng)
        seq.extend(list(codon))
        labels.extend(["EXON", "EXON", "EXON"])

    donor = "GT"
    seq.extend(list(donor))
    labels.extend(["DONOR", "DONOR"])

    # Subtract donor/acceptor bases so total intron length stays close to intron_len
    middle_intron = max(0, intron_len - 4)
    for _ in range(middle_intron):
        seq.append(sample_base(INTRON_DIST, rng))
        labels.append("INTRON")

    acceptor = "AG"
    seq.extend(list(acceptor))
    labels.extend(["ACCEPTOR", "ACCEPTOR"])

    # Second exon
    for _ in range(exon2_codons):
        codon = random_coding_codon(rng)
        seq.extend(list(codon))
        labels.extend(["EXON", "EXON", "EXON"])

    stop = rng.choice(sorted(CODON_TABLE_STOP))
    seq.extend(list(stop))
    labels.extend(["STOP", "STOP", "STOP"])

    # Add downstream intergenic region
    for _ in range(intergenic_after):
        seq.append(sample_base(INTERGENIC_DIST, rng))
        labels.append("I")

    sequence = "".join(seq)
    # Only coding positions are marked as C in coarse labels
    coarse = ["C" if lab in {"START", "EXON", "STOP"} else "N" for lab in labels]
    return SequenceExample(sequence, labels, coarse, organism_type="eukaryote")


def generate_dataset(n_prokaryote=25, n_eukaryote=25, seed=7):
    # Seeded RNG makes dataset generation reproducible
    rng = random.Random(seed)
    data = []

    for _ in range(n_prokaryote):
        data.append(
            generate_prokaryote_sequence(
                intergenic_before=rng.randint(30, 80),
                coding_codons=rng.randint(20, 60),
                intergenic_after=rng.randint(30, 80),
                rng=rng,
            )
        )

    for _ in range(n_eukaryote):
        data.append(
            generate_eukaryote_sequence(
                intergenic_before=rng.randint(25, 70),
                exon1_codons=rng.randint(8, 18),
                intron_len=rng.randint(25, 80),
                exon2_codons=rng.randint(8, 20),
                intergenic_after=rng.randint(25, 70),
                rng=rng,
            )
        )

    # Shuffle so organism types are mixed together
    rng.shuffle(data)
    return data