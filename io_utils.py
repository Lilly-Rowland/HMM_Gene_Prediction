import csv
from pathlib import Path
import matplotlib.pyplot as plt


# Columns used for displaying and saving model results
RESULT_COLUMNS = [
    "model_name",
    "coding_accuracy",
    "coding_sensitivity",
    "coding_specificity",
    "boundary_recall",
    "boundary_precision",
    "start_recall",
    "stop_recall",
    "total_runtime_seconds",
    "num_states",
]


def to_fasta(example, header):
    # Format sequence as FASTA string (">header" + sequence on next line)
    fasta = f">{header}\n{example.sequence}\n"
    
    # Write FASTA to file named after header in outputs/
    with open(f"outputs/{header}", "w") as f:
        f.write(fasta)
    return fasta


def labels_to_string(labels):
    # Convert list like ["A","B","C"] -> "A B C"
    return " ".join(labels)


def print_results_table(results):
    # Initialize column widths (at least 14 chars or header length)
    widths = {c: max(len(c), 14) for c in RESULT_COLUMNS}
    
    # Update widths based on actual values in results
    for row in results:
        for c in RESULT_COLUMNS:
            val = row[c]
            # Format floats to 4 decimal places, otherwise just string
            text = f"{val:.4f}" if isinstance(val, float) else str(val)
            widths[c] = max(widths[c], len(text))

    # Build header row with left-justified columns
    header = " | ".join(c.ljust(widths[c]) for c in RESULT_COLUMNS)
    print(header)
    print("-" * len(header))  # separator line

    # Print each row with same formatting
    for row in results:
        parts = []
        for c in RESULT_COLUMNS:
            val = row[c]
            text = f"{val:.4f}" if isinstance(val, float) else str(val)
            parts.append(text.ljust(widths[c]))  # pad to column width
        print(" | ".join(parts))


def save_results_csv(results, output_path):
    # Ensure parent directory exists (creates it if not)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open CSV file for writing
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        
        # Write each row, keeping only expected columns
        for row in results:
            writer.writerow({col: row[col] for col in RESULT_COLUMNS})


def _bar_plot(model_names, values, title, ylabel, output_path):
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up figure size
    plt.figure(figsize=(10, 6))
    
    # Basic bar chart
    plt.bar(model_names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    
    # Rotate x labels slightly so they don’t overlap
    plt.xticks(rotation=20, ha="right")
    
    plt.tight_layout()  # adjust spacing to prevent clipping
    plt.savefig(output_path, dpi=200)
    plt.close()  # free memory


def save_result_plots(results, output_dir):
    # Make sure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model names for x-axis
    model_names = [row["model_name"] for row in results]

    # Each call builds a different metric plot
    _bar_plot(
        model_names,
        [row["coding_accuracy"] for row in results],  # list comprehension
        "Coding Accuracy by Model",
        "Accuracy",
        output_dir / "coding_accuracy.png",  # Path object join
    )
    _bar_plot(
        model_names,
        [row["coding_sensitivity"] for row in results],
        "Coding Sensitivity by Model",
        "Sensitivity",
        output_dir / "coding_sensitivity.png",
    )
    _bar_plot(
        model_names,
        [row["coding_specificity"] for row in results],
        "Coding Specificity by Model",
        "Specificity",
        output_dir / "coding_specificity.png",
    )
    _bar_plot(
        model_names,
        [row["boundary_recall"] for row in results],
        "Boundary Recall by Model",
        "Recall",
        output_dir / "boundary_recall.png",
    )
    _bar_plot(
        model_names,
        [row["start_recall"] for row in results],
        "Start Codon Recall by Model",
        "Recall",
        output_dir / "start_recall.png",
    )
    _bar_plot(
        model_names,
        [row["total_runtime_seconds"] for row in results],
        "Average Runtime by Model",
        "Seconds",
        output_dir / "total_runtime_seconds",
    )