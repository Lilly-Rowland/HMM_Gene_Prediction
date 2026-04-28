import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib import patheffects
import numpy as np

def draw_node(ax, xy, label, radius=0.34, fontsize=10, lw=1.6):
    circ = Circle(
        xy,
        radius=radius,
        facecolor="white",
        edgecolor="black",
        linewidth=lw
    )
    ax.add_patch(circ)

    txt = ax.text(
        xy[0], xy[1], label,
        ha="center", va="center",
        fontsize=fontsize
    )
    txt.set_path_effects([
        patheffects.withStroke(linewidth=3, foreground="white")
    ])


def edge_points(p1, p2, r):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    d = np.linalg.norm(v)

    if d == 0:
        return p1, p2

    u = v / d
    return p1 + u * r, p2 - u * r


def draw_arrow(
    ax,
    p1,
    p2,
    label=None,
    rad=0.0,
    lw=1.3,
    ms=14,
    r=0.34,
    label_offset=(0, 0),
    ls="-"
):
    start, end = edge_points(p1, p2, r)

    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color="black"
    )
    ax.add_patch(arrow)

    if label:
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = max((dx**2 + dy**2)**0.5, 1e-6)
        perp = np.array([-dy / norm, dx / norm])

        tx = mid[0] + perp[0] * rad * 0.7 + label_offset[0]
        ty = mid[1] + perp[1] * rad * 0.7 + label_offset[1]

        txt = ax.text(tx, ty, label, fontsize=8, ha="center", va="center")
        txt.set_path_effects([
            patheffects.withStroke(linewidth=3, foreground="white")
        ])


def draw_self_loop(ax, p, label=None, direction="up", r=0.34, lw=1.2):
    x, y = p

    if direction == "up":
        start = (x - r * 0.35, y + r * 0.85)
        end   = (x + r * 0.35, y + r * 0.85)
        rad = 1.7
        tx, ty = x, y + r + 0.15   #
    elif direction == "down":
        start = (x + r * 0.35, y - r * 0.85)
        end   = (x - r * 0.35, y - r * 0.85)
        rad = 1.7
        tx, ty = x, y - r - 0.15   # 
    elif direction == "left":
        start = (x - r * 0.85, y - r * 0.35)
        end   = (x - r * 0.85, y + r * 0.35)
        rad = 1.7
        tx, ty = x - r - 0.25, y   # 
    else:  # right
        start = (x + r * 0.85, y + r * 0.35)
        end   = (x + r * 0.85, y - r * 0.35)
        rad = 1.7
        tx, ty = x + r + 0.25, y 

    arrow = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle='-|>',
        mutation_scale=13,
        linewidth=lw,
        color="black"
    )
    ax.add_patch(arrow)

    if label:
        txt = ax.text(tx, ty, label, fontsize=8, ha="center", va="center")
        txt.set_path_effects([
            patheffects.withStroke(linewidth=3, foreground="white")
        ])

def add_emission_note(ax, x, y, text):
    ax.text(
        x, y, text,
        fontsize=8.5,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="black", lw=0.8)
    )


def make_hmm_figure(save_prefix="hmm_models_paper_diagram"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    # -------------------------
    # Model 1
    # -------------------------
    ax = axes[0]
    ax.set_title("Model 1: 2-state coding/noncoding", fontsize=12, pad=10)

    pos = {"N": (0, 0), "C": (2.1, 0)}

    for k, p in pos.items():
        draw_node(ax, p, k)

    draw_self_loop(ax, pos["N"], "0.96", direction="up")
    draw_self_loop(ax, pos["C"], "0.96", direction="up")
    draw_arrow(ax, pos["N"], pos["C"], "0.04", rad=0.08, label_offset=(0, 0.02))
    draw_arrow(ax, pos["C"], pos["N"], "0.04", rad=0.08, label_offset=(0, -0.05))

    add_emission_note(ax, 0, -1.0, "emit: intergenic")
    add_emission_note(ax, 2.1, -1.0, "emit: coding")

    ax.set_xlim(-1.0, 3.1)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    # -------------------------
    # Model 2
    # -------------------------
    ax = axes[1]
    ax.set_title("Model 2: explicit start/stop boundaries", fontsize=12, pad=10)

    pos = {
        "N": (0, 0),
        "START": (1.6, 0),
        "EXON": (3.5, 0),
        "STOP": (5.4, 0)
    }

    for k, p in pos.items():
        draw_node(ax, p, k, fontsize=9)

    draw_self_loop(ax, pos["N"], "0.965", direction="up")
    draw_self_loop(ax, pos["START"], "0.60", direction="up")
    draw_self_loop(ax, pos["EXON"], "0.975", direction="up")
    draw_self_loop(ax, pos["STOP"], "0.058", direction="up")

    draw_arrow(ax, pos["N"], pos["START"], "0.03")
    draw_arrow(ax, pos["START"], pos["EXON"], "0.395")
    draw_arrow(ax, pos["EXON"], pos["STOP"], "0.022")
    draw_arrow(ax, pos["STOP"], pos["N"], "0.93", rad=0.35, label_offset=(0, 1))

    add_emission_note(ax, 0, -1.0, "intergenic")
    add_emission_note(ax, 1.6, -1.0, "START motif")
    add_emission_note(ax, 3.5, -1.0, "coding")
    add_emission_note(ax, 5.4, -1.0, "STOP motif")

    ax.text(3.0, 1.12, "Only dominant transitions shown", fontsize=8, ha="center")

    ax.set_xlim(-0.9, 6.3)
    ax.set_ylim(-1.5, 1.6)
    ax.axis("off")

    # -------------------------
    # Model 3
    # -------------------------
    ax = axes[2]
    ax.set_title("Model 3: splice-aware exon–intron HMM", fontsize=12, pad=10)

    pos = {
        "I": (0, 0),
        "START": (1.5, 0.9),
        "EXON": (3.2, 0.9),
        "DONOR": (4.9, 0.9),
        "INTRON": (4.9, -0.9),
        "ACCEPTOR": (3.2, -0.9),
        "STOP": (1.5, -0.9),
    }

    for k, p in pos.items():
        draw_node(ax, p, k, fontsize=8.8)

    draw_self_loop(ax, pos["I"], "0.955", direction="left")
    draw_self_loop(ax, pos["START"], "0.55", direction="up")
    draw_self_loop(ax, pos["EXON"], "0.955", direction="up")
    draw_self_loop(ax, pos["DONOR"], "0.12", direction="right")
    draw_self_loop(ax, pos["INTRON"], "0.955", direction="down")
    draw_self_loop(ax, pos["ACCEPTOR"], "0.01", direction="down")
    draw_self_loop(ax, pos["STOP"], "0.03", direction="down")

    draw_arrow(ax, pos["I"], pos["START"], "0.038")
    draw_arrow(ax, pos["START"], pos["EXON"], "0.438")
    draw_arrow(ax, pos["EXON"], pos["DONOR"], "0.025")
    draw_arrow(ax, pos["DONOR"], pos["INTRON"], "0.87", label_offset=(0.38, 0.0))
    draw_arrow(ax, pos["INTRON"], pos["ACCEPTOR"], "0.04")
    draw_arrow(ax, pos["ACCEPTOR"], pos["EXON"], "0.98", label_offset=(-0.42, 0.0))
    draw_arrow(
        ax,
        pos["EXON"],
        pos["STOP"],
        "0.016",
        rad=-0.38,
        label_offset=(1, -0.12)
    )
    draw_arrow(ax, pos["STOP"], pos["I"], "0.96", rad=-0.25, label_offset=(0.0, -0.05))

    add_emission_note(ax, 0, 1.6, "intergenic")
    add_emission_note(ax, 2.35, 1.65, "START motif")
    add_emission_note(ax, 4.1, 1.65, "coding")
    add_emission_note(ax, 6.0, 0.9, "GT donor")
    add_emission_note(ax, 6.0, -0.9, "intron")
    add_emission_note(ax, 3.2, -1.75, "AG acceptor")
    add_emission_note(ax, 1.5, -1.75, "STOP motif")

    ax.set_xlim(-1.0, 6.6)
    ax.set_ylim(-2.1, 2.0)
    ax.axis("off")

    # -------------------------
    # Model 4
    # -------------------------
    ax = axes[3]
    ax.set_title("Model 4: codon-aware periodic HMM", fontsize=12, pad=10)

    pos = {
        "I": (0.0, 0.0),
        "START": (2.0, 0.0),
        "C1": (4.4, 1.0),
        "C2": (4.4, 0.0),
        "C3": (4.4, -1.0),
        "STOP": (6.9, 0.0),
    }

    for k, p in pos.items():
        draw_node(ax, p, k, fontsize=9)

    # self-loops
    draw_self_loop(ax, pos["I"], "0.968", direction="up")
    draw_self_loop(ax, pos["START"], "0.60", direction="up")
    draw_self_loop(ax, pos["STOP"], "0.041", direction="right")

    # main entry path
    draw_arrow(ax, pos["I"], pos["START"], "0.026", rad=0.0, label_offset=(0, 0.02))
    draw_arrow(ax, pos["START"], pos["C1"], "0.395", rad=0.0, label_offset=(0.02, 0.03))

    # vertical codon progression
    draw_arrow(ax, pos["C1"], pos["C2"], "0.985", rad=0.0, label_offset=(-0.38, 0.0))
    draw_arrow(ax, pos["C2"], pos["C3"], "0.995", rad=0.0, label_offset=(-0.38, 0.0))

    # periodic cycle: keep it tight on the right side of the codon stack
    draw_arrow(ax, pos["C3"], pos["C1"], "0.985", rad=-0.55, label_offset=(0.55, 0.0))

    # stop transitions
    draw_arrow(ax, pos["C1"], pos["STOP"], "0.011", rad=0.08, label_offset=(0.02, 0.12))
    draw_arrow(ax, pos["C3"], pos["STOP"], "0.011", rad=-0.08, label_offset=(0.02, -0.12))

    # return to intergenic: send it BELOW the whole diagram
    draw_arrow(ax, pos["STOP"], pos["I"], "0.95", rad=-0.48, label_offset=(0, -0.38))

    # emission notes
    add_emission_note(ax, pos["I"][0], -1.75, "intergenic")
    add_emission_note(ax, pos["START"][0], -1.75, "START motif")
    add_emission_note(ax, 4.4, 1.75, "codon pos. 1")
    add_emission_note(ax, 5.45, 0.0, "codon pos. 2")
    add_emission_note(ax, 4.4, -1.75, "codon pos. 3")
    add_emission_note(ax, pos["STOP"][0], -1.75, "STOP motif")

    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-2.3, 2.0)
    ax.axis("off")

    fig.suptitle(
        "Hidden Markov model architectures used for gene-structure experiments",
        fontsize=14,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    make_hmm_figure()