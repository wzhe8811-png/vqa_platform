import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description="Plot baseline comparison bar chart.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/baseline_comparison_complex20.csv",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/baseline_comparison_complex20.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="GQA Complex Subset: Baseline Comparison (Exact vs Soft Accuracy)",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    methods = [r["method"] for r in rows]
    exact = [float(r["exact_acc"]) for r in rows]
    soft = [float(r["soft_acc"]) for r in rows]

    x = np.arange(len(methods))
    width = 0.36

    plt.figure(figsize=(11, 6))
    bars1 = plt.bar(x - width / 2, exact, width, label="Exact Acc (%)")
    bars2 = plt.bar(x + width / 2, soft, width, label="Soft Acc (%)")

    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.title(args.title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            plt.text(
                b.get_x() + b.get_width() / 2,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    out_dir = os.path.dirname(args.output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=220)
    print(f"Saved figure: {args.output_png}")


if __name__ == "__main__":
    main()
