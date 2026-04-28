from pathlib import Path
from data_generation import generate_dataset
from eval_utils import benchmark_all_models
from io_utils import print_results_table, save_results_csv, to_fasta, labels_to_string
import matplotlib.pyplot as plt


def run_size_scaling(sizes, seed=42, outdir="outputs", warmup=True):
    """Run benchmarks for multiple dataset sizes and plot only size-scaling comparisons."""
    output_dir = Path(outdir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    aggregated = {} # store results across all dataset sizes and grouped by the model

    for size in sizes:
        print(f"Running benchmark for n_prok=n_euk={size}...")

        #generate the synthetic datasdet with equal prok and euk examples
        dataset = generate_dataset(n_prokaryote=size, n_eukaryote=size, seed=seed)

        prok = [ex for ex in dataset if ex.organism_type == "prokaryote"]
        euk = [ex for ex in dataset if ex.organism_type == "eukaryote"]


        # run benchmarking on each of the subsedts
        prok_results = benchmark_all_models(prok,warmup=warmup)
        euk_results = benchmark_all_models(euk,warmup=warmup)
        
        # initlaizie strucutre on first iteration of resuluts
        if not aggregated:
            for r in prok_results:
                aggregated[r["model_name"]] = {
                    "sizes": [],
                    "prok_coding_accuracy": [],
                    "euk_coding_accuracy": [],
                    "prok_total_runtime_seconds": [],
                    "euk_total_runtime_seconds": [],
                }
        # colelct the results
        for pr, er in zip(prok_results, euk_results):
            m = aggregated[pr["model_name"]]
            total_examples = len(prok) + len(euk)

            m["sizes"].append(total_examples)
            m["prok_coding_accuracy"].append(pr["coding_accuracy"])
            m["euk_coding_accuracy"].append(er["coding_accuracy"])
            m["prok_total_runtime_seconds"].append(pr["total_runtime_seconds"])
            m["euk_total_runtime_seconds"].append(er["total_runtime_seconds"])

    # ---- Plot coding accuracy vs dataset size ----
    plt.figure(figsize=(8, 5))
    for model_name, data in aggregated.items():
        plt.plot(
            data["sizes"],
            data["prok_coding_accuracy"],
            marker="o",
            label=f"{model_name} (prok)",
        )
        plt.plot(
            data["sizes"],
            data["euk_coding_accuracy"],
            marker="s",
            linestyle="--",
            label=f"{model_name} (euk)",
        )
    plt.xlabel("Total examples (prok + euk)")
    plt.ylabel("Coding accuracy")
    plt.title("Coding accuracy vs dataset size (prok vs euk)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "size_scaling_coding_accuracy.png", dpi=200)
    plt.close()

    # ---- Plot total runtime vs dataset size ----
    plt.figure(figsize=(8, 5))
    for model_name, data in aggregated.items():
        plt.plot(
            data["sizes"],
            data["prok_total_runtime_seconds"],
            marker="o",
            label=f"{model_name} (prok)",
        )
        plt.plot(
            data["sizes"],
            data["euk_total_runtime_seconds"],
            marker="s",
            linestyle="--",
            label=f"{model_name} (euk)",
        )
    plt.xlabel("Total examples (prok + euk)")
    plt.ylabel("Total runtime (s)")
    plt.title("Runtime vs dataset size (prok vs euk)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "size_scaling_runtime.png", dpi=200)
    plt.close()

    print(f"Saved scaling plots to: {plot_dir}")


def run(n_prokaryote=250, n_eukaryote=250, seed=42, outdir="outputs", warmup=True):
   #generate a full dataset
    dataset = generate_dataset(
        n_prokaryote=n_prokaryote,
        n_eukaryote=n_eukaryote,
        seed=seed,
    )

    #split for comparison
    prok = [ex for ex in dataset if ex.organism_type == "prokaryote"]
    euk = [ex for ex in dataset if ex.organism_type == "eukaryote"]

    #benchmark separately
    prok_results = benchmark_all_models( prok, warmup=warmup)
    euk_results = benchmark_all_models(euk, warmup=warmup)

    #benchmark together
    results = benchmark_all_models(dataset,warmup=warmup)
    print_results_table(results)

    output_dir = Path(outdir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # save csv data
    save_results_csv(results, output_dir / "benchmark_results.csv")

    print(f"\nSaved benchmark table to: {output_dir / 'benchmark_results.csv'}")
    print(f"Saved plots to: {plot_dir}")

    model_names = [r["model_name"] for r in prok_results]
    x = range(len(model_names))
    width = 0.35

    # ---- coding accuracy grouped bar plots ----
    prok_acc = [r["coding_accuracy"] for r in prok_results]
    euk_acc = [r["coding_accuracy"] for r in euk_results]

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], prok_acc, width=width, label="prokaryote")
    plt.bar([i + width / 2 for i in x], euk_acc, width=width, label="eukaryote")
    plt.xticks(list(x), model_names, rotation=20, ha="right")
    plt.ylabel("Coding accuracy")
    plt.title("Coding accuracy: prok vs euk by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "prok_vs_euk_coding_accuracy.png", dpi=200)
    plt.close()

    # ---- runtime grouped bars ----
    prok_rt = [r["total_runtime_seconds"] for r in prok_results]
    euk_rt = [r["total_runtime_seconds"] for r in euk_results]
    plt.ylabel("Total runtime (s)")
    plt.title("Total runtime: prok vs euk by model")

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], prok_rt, width=width, label="prokaryote")
    plt.bar([i + width / 2 for i in x], euk_rt, width=width, label="eukaryote")
    plt.xticks(list(x), model_names, rotation=20, ha="right")
    # plt.ylabel("Average runtime per sequence (s)")
    # plt.title("Average runtime: prok vs euk by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "prok_vs_euk_runtime.png", dpi=200)
    plt.close()

    # ---- sensitivity grouped bar plots ----
    prok_sens = [r["coding_sensitivity"] for r in prok_results]
    euk_sens = [r["coding_sensitivity"] for r in euk_results]

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], prok_sens, width=width, label="prokaryote")
    plt.bar([i + width / 2 for i in x], euk_sens, width=width, label="eukaryote")
    plt.xticks(list(x), model_names, rotation=20, ha="right")
    plt.ylabel("Coding sensitivity")
    plt.title("Coding sensitivity: prok vs euk by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "prok_vs_euk_coding_sensitivity.png", dpi=200)
    plt.close()

    # ---- specificity  ----
    prok_spec = [r["coding_specificity"] for r in prok_results]
    euk_spec = [r["coding_specificity"] for r in euk_results]

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], prok_spec, width=width, label="prokaryote")
    plt.bar([i + width / 2 for i in x], euk_spec, width=width, label="eukaryote")
    plt.xticks(list(x), model_names, rotation=20, ha="right")
    plt.ylabel("Coding specificity")
    plt.title("Coding specificity: prok vs euk by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "prok_vs_euk_coding_specificity.png", dpi=200)
    plt.close()

    # ---- eukaryote structural benchmark plot ----
    metrics_to_plot = {
        "Start recall": [r["start_recall"] for r in euk_results],
        "Stop recall": [r["stop_recall"] for r in euk_results],
        "Intron sensitivity": [r["intron_sensitivity"] for r in euk_results],
        "Splice recall": [r["splice_recall"] for r in euk_results],
        "Donor recall": [r["donor_recall"] for r in euk_results],
        "Acceptor recall": [r["acceptor_recall"] for r in euk_results],
    }

    model_names = [r["model_name"] for r in euk_results]
    x = range(len(model_names))
    width = 0.12

    plt.figure(figsize=(10, 6))

    for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
        offset = (i - (len(metrics_to_plot) - 1) / 2) * width
        plt.bar(
            [j + offset for j in x],
            values,
            width=width,
            label=metric_name,
        )

    plt.xticks(list(x), model_names, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Eukaryote structural feature detection by model")
    plt.legend(title="Benchmark")
    plt.tight_layout()
    plt.savefig(plot_dir / "euk_structural_benchmarks.png", dpi=200)
    plt.close()

    # show example of synthetic sequences as a sanity check
    print("\nExample synthetic sequences:\n")
    for i, ex in enumerate(dataset[:2], start=1):
        print(to_fasta(ex, header=f"synthetic_{i}_{ex.organism_type}").strip())
        print("Labels:")
        print(labels_to_string(ex.fine_labels[:80]), "...")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HMM benchmarking on synthetic data")

    # debugging arguemnts
    parser.add_argument("--n-prok", type=int, default=250, help="number of prokaryote examples")
    parser.add_argument("--n-euk", type=int, default=250, help="number of eukaryote examples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--outdir", type=str, default="outputs", help="output directory")

    parser.add_argument(
        "--scale-sizes",
        type=str,
        default="50,100,250",
        help="comma-separated sizes for scaling experiment",
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="disable warmup run before timing",
    )

    args = parser.parse_args()

    warmup = not args.no_warmup

    run(
        n_prokaryote=args.n_prok,
        n_eukaryote=args.n_euk,
        seed=args.seed,
        outdir=args.outdir,
        warmup=warmup,
    )

    if args.scale_sizes:
        sizes = [int(s.strip()) for s in args.scale_sizes.split(",") if s.strip()]
        run_size_scaling(
            sizes,
            seed=args.seed,
            outdir=args.outdir,
            warmup=warmup,
        )