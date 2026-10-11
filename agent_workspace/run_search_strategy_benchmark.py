import argparse
import csv
import json
import os
import re
import sys
import subprocess
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/vision_qa_platform")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reasoning_modules import QuestionDecomposer, SearchController, StateEvaluator, ThoughtGenerator
from tot_got_integration import VisionToTextReasoner
from vqa_eval_utils import constrained_finalize_answer, is_complex_question, answers_equivalent


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


def load_gqa_questions(path: str):
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    out.append(json.loads(ln))
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported question format")


def find_questions_file(input_dir: str) -> str:
    candidates = [
        os.path.join(input_dir, "gqa_clean_test.jsonl"),
        os.path.join(input_dir, "testdev_balanced_questions.json"),
        os.path.join(input_dir, "questions", "testdev_balanced_questions.json"),
    ]
    p = next((x for x in candidates if os.path.exists(x)), None)
    if not p:
        raise FileNotFoundError(f"No question file found in {input_dir}")
    return p


def image_path(input_dir: str, image_name: str) -> str:
    if os.path.isabs(image_name):
        return image_name
    p = os.path.join(input_dir, "images", image_name)
    if os.path.exists(p):
        return p
    return os.path.join(input_dir, image_name)


def sample_peak_gpu(stop_event: threading.Event, interval=1.0):
    peak = 0
    while not stop_event.is_set():
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                vals = [int(x.strip()) for x in res.stdout.splitlines() if x.strip().isdigit()]
                if vals:
                    peak = max(peak, max(vals))
        except Exception:
            pass
        time.sleep(interval)
    return peak


def main():
    parser = argparse.ArgumentParser(description="Benchmark search strategy configs for modular reasoning.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--complex_only", action="store_true")
    parser.add_argument(
        "--configs",
        type=str,
        default="2x2,3x2,3x3,4x2,4x3",
        help="comma-separated n_subquestions x n_candidates",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/search_strategy_benchmark.csv",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/search_strategy_benchmark.md",
    )
    args = parser.parse_args()

    q_path = find_questions_file(args.input_dir)
    questions = load_gqa_questions(q_path)
    if args.complex_only:
        questions = [x for x in questions if is_complex_question(x.get("question", ""))]
    questions = questions[: args.max_samples]

    reasoner = VisionToTextReasoner(
        vision_model_path="/root/autodl-tmp/vision_qa_platform/llava-1.5-7b-hf",
        llm_model_path="/root/autodl-tmp/vision_qa_platform/text_model/qwen2_1_5b_instruct",
    )
    llm_caller = reasoner.get_llm_caller()
    decomposer = QuestionDecomposer(llm_caller)
    generator = ThoughtGenerator(llm_caller)
    evaluator = StateEvaluator(llm_caller)
    controller = SearchController(decomposer, generator, evaluator, llm_caller)

    confs = []
    for token in args.configs.split(","):
        token = token.strip().lower()
        if "x" not in token:
            continue
        a, b = token.split("x", 1)
        confs.append((int(a), int(b)))

    rows = []
    for n_subq, n_cand in confs:
        stop_event = threading.Event()
        holder = {"peak": 0}

        def _sample():
            holder["peak"] = sample_peak_gpu(stop_event)

        th = threading.Thread(target=_sample, daemon=True)
        th.start()

        t0 = time.perf_counter()
        exact = 0
        soft = 0
        done = 0
        for i, item in enumerate(questions):
            q = item.get("question", "")
            gt = item.get("answer", "")
            img = image_path(args.input_dir, item.get("image", item.get("full_image_path", "")))
            if not os.path.exists(img):
                continue
            visual_desc = reasoner.get_visual_description(img, q)
            solved = controller.solve(
                visual_description=visual_desc,
                question=q,
                n_subquestions=n_subq,
                n_candidates=n_cand,
            )
            pred = constrained_finalize_answer(
                llm_caller, visual_desc, q, solved.get("final_answer", "")
            )
            done += 1
            ex = answers_equivalent(pred, gt)
            sf = soft_match(pred, gt)
            exact += 1 if ex else 0
            soft += 1 if sf else 0
            print(f"[{n_subq}x{n_cand}] {i+1}/{len(questions)} exact={ex} soft={sf}")

        elapsed = time.perf_counter() - t0
        stop_event.set()
        th.join(timeout=2.0)

        rows.append(
            {
                "strategy": f"{n_subq}x{n_cand}",
                "total": done,
                "exact_acc": (100.0 * exact / done) if done else 0.0,
                "soft_acc": (100.0 * soft / done) if done else 0.0,
                "wall_time_sec": elapsed,
                "avg_time_per_sample_sec": (elapsed / done) if done else 0.0,
                "peak_gpu_mem_mb": holder["peak"],
            }
        )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "total",
                "exact_acc",
                "soft_acc",
                "wall_time_sec",
                "avg_time_per_sample_sec",
                "peak_gpu_mem_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = []
    md_lines.append("| Strategy(n_subq x n_cand) | Total | Exact Acc(%) | Soft Acc(%) | Wall Time(s) | Avg Time/Sample(s) | Peak GPU Mem(MB) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['strategy']} | {r['total']} | {r['exact_acc']:.2f} | {r['soft_acc']:.2f} | "
            f"{r['wall_time_sec']:.2f} | {r['avg_time_per_sample_sec']:.2f} | {r['peak_gpu_mem_mb']} |"
        )
    Path(args.output_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"SAVED: {args.output_csv}")
    print(f"SAVED: {args.output_md}")


if __name__ == "__main__":
    main()
