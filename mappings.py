def map_fine_to_model1(fine_labels):
    # Collapse all coding-related labels into "C", everything else -> "N"
    return ["C" if lab in {"START", "EXON", "STOP"} else "N" for lab in fine_labels]


def map_fine_to_model2(fine_labels):
    mapped = []
    for lab in fine_labels:
        # Non-coding regions all mapped to "N"
        if lab == "I" or lab in {"DONOR", "INTRON", "ACCEPTOR"}:
            mapped.append("N")
        elif lab == "START":
            mapped.append("START")
        elif lab == "EXON":
            mapped.append("EXON")
        elif lab == "STOP":
            mapped.append("STOP")
        else:
            mapped.append("N")  # fallback for anything unexpected
    return mapped


def map_fine_to_model3(fine_labels):
    mapped = []
    for lab in fine_labels:
        # Keep more detailed structure (explicit states)
        if lab == "I":
            mapped.append("I")
        elif lab == "START":
            mapped.append("START")
        elif lab == "EXON":
            mapped.append("EXON")
        elif lab == "DONOR":
            mapped.append("DONOR")
        elif lab == "INTRON":
            mapped.append("INTRON")
        elif lab == "ACCEPTOR":
            mapped.append("ACCEPTOR")
        elif lab == "STOP":
            mapped.append("STOP")
        else:
            mapped.append("I")  # default to intergenic
    return mapped


def map_fine_to_model4(fine_labels):
    mapped = []
    frame = 1  # keeps track of codon position (1,2,3)

    for lab in fine_labels:
        # Reset frame whenever leaving coding region
        if lab == "I" or lab in {"DONOR", "INTRON", "ACCEPTOR"}:
            mapped.append("I")
            frame = 1
        elif lab == "START":
            mapped.append("START")
            frame = 1
        elif lab == "EXON":
            # Label exon positions with reading frame (C1, C2, C3)
            mapped.append(f"C{frame}")
            frame = 1 if frame == 3 else frame + 1 
        elif lab == "STOP":
            mapped.append("STOP")
            frame = 1
        else:
            mapped.append("I")
            frame = 1

    return mapped