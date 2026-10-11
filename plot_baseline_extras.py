import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_exact_only(rows, output_png: str):
    methods = [r["method"] for r in rows]
    exact = [float(r["exact_acc"]) for r in rows]

    x = np.arange(len(methods))
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(x, exact, color="#2C7FB8")
    plt.ylabel("Exact Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.title("GQA Complex Subset: Exact Accuracy by Method")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ensure_dir(output_png)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def plot_gain_vs_standard(rows, output_png: str):
    methods = [r["method"] for r in rows]
    exact = [float(r["exact_acc"]) for r in rows]
    soft = [float(r["soft_acc"]) for r in rows]

    # baseline assumed row 0 = Standard
    base_exact = exact[0]
    base_soft = soft[0]

    gain_exact = [v - base_exact for v in exact]
    gain_soft = [v - base_soft for v in soft]

    x = np.arange(len(methods))
    width = 0.36
    plt.figure(figsize=(11, 6))
    bars1 = plt.bar(x - width / 2, gain_exact, width, label="Exact Gain vs Standard")
    bars2 = plt.bar(x + width / 2, gain_soft, width, label="Soft Gain vs Standard")

    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Gain (percentage points)")
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.title("GQA Complex Subset: Accuracy Gain vs Standard")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            plt.text(
                b.get_x() + b.get_width() / 2,
                h + (0.8 if h >= 0 else -1.2),
                f"{h:+.1f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=9,
            )

    ensure_dir(output_png)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create extra paper-ready baseline figures.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/baseline_comparison_complex20.csv",
    )
    parser.add_argument(
        "--exact_png",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/baseline_exact_only_complex20.png",
    )
    parser.add_argument(
        "--gain_png",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/baseline_gain_vs_standard_complex20.png",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    if not rows:
        raise ValueError("CSV is empty.")
    if rows[0]["method"].lower() != "standard":
        raise ValueError("First row must be Standard for gain plot.")

    plot_exact_only(rows, args.exact_png)
    plot_gain_vs_standard(rows, args.gain_png)
    print(f"Saved exact-only figure: {args.exact_png}")
    print(f"Saved gain figure: {args.gain_png}")


if __name__ == "__main__":
    main()
