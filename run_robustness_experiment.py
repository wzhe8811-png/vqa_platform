import argparse
import csv
import os
import subprocess
from typing import Dict, List


def parse_levels(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}")
    subprocess.run(["bash", "-lc", cmd], check=True)


def read_compare_csv(path: str) -> Dict[str, Dict[str, float]]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[r["method"]] = {
                "exact_acc": float(r["exact_acc"]),
                "soft_acc": float(r["soft_acc"]),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description="Run robustness experiment over Gaussian noise levels.")
    parser.add_argument(
        "--project_root",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/data/processed_gqa_robustness",
    )
    parser.add_argument("--levels", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--complex_only", action="store_true")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/robustness_summary.csv",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default="/root/miniconda3/envs/llava_env/bin/python",
    )
    args = parser.parse_args()

    levels = parse_levels(args.levels)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)

    records = []
    for level in levels:
        tag = f"gaussian_{level:.2f}"
        input_dir = os.path.join(args.data_root, tag)
        if not os.path.exists(os.path.join(input_dir, "gqa_clean_test.jsonl")):
            raise FileNotFoundError(f"Missing dataset for level {level:.2f}: {input_dir}")

        standard_out = os.path.join(args.project_root, "results", f"standard_{tag}.json")
        cot_out = os.path.join(args.project_root, "results", f"cot_{tag}.json")
        tot_out = os.path.join(args.project_root, "results", f"tot_{tag}.json")
        got_out = os.path.join(args.project_root, "results", f"got_{tag}.json")
        modular_out = os.path.join(args.project_root, "results", f"modular_{tag}.json")
        compare_csv = os.path.join(args.project_root, "results", f"compare_{tag}.csv")
        compare_md = os.path.join(args.project_root, "results", f"compare_{tag}.md")

        complex_flag = "--complex_only" if args.complex_only else ""
        py = args.python_exec

        run_cmd(
            f"\"{py}\" \"{args.project_root}/run_standard_on_gqa.py\" --input_dir \"{input_dir}\" --max_samples {args.max_samples} {complex_flag} --output_file \"{standard_out}\""
        )
        run_cmd(
            f"\"{py}\" \"{args.project_root}/run_cot_on_gqa.py\" --input_dir \"{input_dir}\" --max_samples {args.max_samples} {complex_flag} --output_file \"{cot_out}\""
        )
        run_cmd(
            f"\"{py}\" \"{args.project_root}/run_tot_on_gqa.py\" --input_dir \"{input_dir}\" --max_samples {args.max_samples} {complex_flag} --output_file \"{tot_out}\""
        )
        run_cmd(
            f"\"{py}\" \"{args.project_root}/run_got_on_gqa.py\" --input_dir \"{input_dir}\" --max_samples {args.max_samples} {complex_flag} --output_file \"{got_out}\""
        )
        run_cmd(
            f"\"{py}\" \"{args.project_root}/run_modular_reasoning_on_gqa.py\" --input_dir \"{input_dir}\" --max_samples {args.max_samples} {complex_flag} --output_file \"{modular_out}\""
        )
        run_cmd(
            (
                f"\"{py}\" \"{args.project_root}/compare_gqa_baselines.py\" "
                f"--standard_file \"{standard_out}\" --cot_file \"{cot_out}\" --tot_file \"{tot_out}\" --got_file \"{got_out}\" "
                f"--modular_file \"{modular_out}\" --output_md \"{compare_md}\" --output_csv \"{compare_csv}\""
            )
        )

        stats = read_compare_csv(compare_csv)
        for method, vals in stats.items():
            records.append(
                {
                    "noise_level": level,
                    "method": method,
                    "exact_acc": vals["exact_acc"],
                    "soft_acc": vals["soft_acc"],
                }
            )

    with open(args.summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_level", "method", "exact_acc", "soft_acc"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved robustness summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
