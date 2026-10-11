import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot robustness curves from summary csv.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/robustness_summary.csv",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/robustness_curve_exact.png",
    )
    parser.add_argument("--metric", type=str, default="exact_acc", choices=["exact_acc", "soft_acc"])
    parser.add_argument("--title", type=str, default="Robustness Curve: Accuracy vs Gaussian Noise")
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input summary csv is empty.")

    grouped = defaultdict(list)
    for r in rows:
        method = r["method"]
        lvl = float(r["noise_level"])
        val = float(r[args.metric])
        grouped[method].append((lvl, val))

    plt.figure(figsize=(10, 6))
    for method, pts in grouped.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("Gaussian Noise Level (sigma ratio)")
    plt.ylabel("Accuracy (%)")
    plt.title(args.title + f" [{args.metric}]")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_dir = os.path.dirname(args.output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=220)
    print(f"Saved robustness figure: {args.output_png}")


if __name__ == "__main__":
    main()
