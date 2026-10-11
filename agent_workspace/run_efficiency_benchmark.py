import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from vqa_eval_utils import answers_equivalent


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"^[a-z]\s*:\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def soft_match(pred: str, gt: str) -> bool:
    pred_n = normalize_text(pred)
    gt_n = normalize_text(gt)
    if not pred_n or not gt_n:
        return False
    if pred_n == gt_n:
        return True
    if gt_n in pred_n:
        return True
    pred_n2 = re.sub(r"[^a-z0-9\s]", " ", pred_n)
    gt_n2 = re.sub(r"[^a-z0-9\s]", " ", gt_n)
    pred_n2 = re.sub(r"\s+", " ", pred_n2).strip()
    gt_n2 = re.sub(r"\s+", " ", gt_n2).strip()
    return bool(gt_n2) and gt_n2 in pred_n2


def compute_metrics(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    total = len(rows)
    exact = 0
    soft = 0
    for r in rows:
        pred = r.get("prediction", "")
        gt = r.get("ground_truth", r.get("answer", ""))
        ex = answers_equivalent(pred, gt)
        sf = soft_match(pred, gt)
        exact += 1 if ex else 0
        soft += 1 if sf else 0
    return {
        "total": total,
        "exact_acc": (100.0 * exact / total) if total else 0.0,
        "soft_acc": (100.0 * soft / total) if total else 0.0,
    }


def sample_gpu_memory(stop_event: threading.Event, interval_sec=1.0):
    peak = 0
    while not stop_event.is_set():
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                vals = []
                for ln in res.stdout.strip().splitlines():
                    ln = ln.strip()
                    if ln:
                        vals.append(int(ln))
                if vals:
                    peak = max(peak, max(vals))
        except Exception:
            pass
        time.sleep(interval_sec)
    return peak


def run_one(command, working_dir):
    stop_event = threading.Event()
    holder = {"peak": 0}

    def _sampler():
        holder["peak"] = sample_gpu_memory(stop_event)

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=working_dir, check=False)
    elapsed = time.perf_counter() - start
    stop_event.set()
    t.join(timeout=2.0)
    return proc.returncode, elapsed, holder["peak"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark runtime and GPU memory for VQA methods.")
    parser.add_argument(
        "--project_root",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/data/processed_gqa_robustness_50/gaussian_0.00",
    )
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--complex_only", action="store_true")
    parser.add_argument(
        "--python_exec",
        type=str,
        default="/root/miniconda3/envs/llava_env/bin/python",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/efficiency_benchmark_complex50.csv",
    )
    parser.add_argument(
        "--out_md",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/efficiency_benchmark_complex50.md",
    )
    args = parser.parse_args()

    pr = Path(args.project_root)
    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("Standard", "run_standard_on_gqa.py", "eff_standard_tmp.json"),
        ("CoT", "run_cot_on_gqa.py", "eff_cot_tmp.json"),
        ("ToT", "run_tot_on_gqa.py", "eff_tot_tmp.json"),
        ("GoT", "run_got_on_gqa.py", "eff_got_tmp.json"),
        ("Modular(4-Module)", "run_modular_reasoning_on_gqa.py", "eff_modular_tmp.json"),
    ]

    rows = []
    for method_name, script_name, tmp_name in methods:
        out_json = pr / "results" / tmp_name
        cmd = [
            args.python_exec,
            str(pr / script_name),
            "--input_dir",
            args.input_dir,
            "--max_samples",
            str(args.max_samples),
            "--output_file",
            str(out_json),
        ]
        if args.complex_only:
            cmd.append("--complex_only")

        print(f"=== Running {method_name} ===")
        code, elapsed, peak_mb = run_one(cmd, str(pr))
        if code != 0:
            rows.append(
                {
                    "method": method_name,
                    "total": 0,
                    "exact_acc": 0.0,
                    "soft_acc": 0.0,
                    "wall_time_sec": round(elapsed, 3),
                    "peak_gpu_mem_mb": peak_mb,
                    "status": f"failed({code})",
                }
            )
            continue

        m = compute_metrics(out_json)
        rows.append(
            {
                "method": method_name,
                "total": m["total"],
                "exact_acc": round(m["exact_acc"], 4),
                "soft_acc": round(m["soft_acc"], 4),
                "wall_time_sec": round(elapsed, 3),
                "peak_gpu_mem_mb": peak_mb,
                "status": "ok",
            }
        )

    # CSV
    fields = ["method", "total", "exact_acc", "soft_acc", "wall_time_sec", "peak_gpu_mem_mb", "status"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md = []
    md.append("| Method | Total | Exact Acc(%) | Soft Acc(%) | Wall Time(s) | Peak GPU Mem(MB) | Status |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(
            f"| {r['method']} | {r['total']} | {r['exact_acc']:.2f} | {r['soft_acc']:.2f} | "
            f"{r['wall_time_sec']:.2f} | {r['peak_gpu_mem_mb']} | {r['status']} |"
        )
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"SAVED: {args.out_csv}")
    print(f"SAVED: {args.out_md}")


if __name__ == "__main__":
    main()
