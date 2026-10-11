import argparse
import csv
import json
import os
import re

from vqa_eval_utils import answers_equivalent


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"^[a-z]\s*:\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize_text(s: str):
    s = re.sub(r"[^a-z0-9\s]", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split(" ")


def _canonical_token(t: str) -> str:
    t = (t or "").strip().lower()
    token_map = {
        "y": "yes",
        "n": "no",
        "yeah": "yes",
        "yep": "yes",
        "nope": "no",
        "leftside": "left",
        "rightside": "right",
    }
    t = token_map.get(t, t)

    # light stemming for common VQA nouns
    if len(t) > 3 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 2 and t.endswith("s"):
        t = t[:-1]
    return t


def soft_match(pred: str, gt: str) -> bool:
    pred_n = normalize_text(pred)
    gt_n = normalize_text(gt)
    if not pred_n or not gt_n:
        return False
    if pred_n == gt_n:
        return True

    pred_tokens = [_canonical_token(x) for x in _tokenize_text(pred_n)]
    gt_tokens = [_canonical_token(x) for x in _tokenize_text(gt_n)]
    if not pred_tokens or not gt_tokens:
        return False

    # 单词答案：仅在“短答案输出”中允许命中，避免长解释蹭词
    if len(gt_tokens) == 1:
        gt_tok = gt_tokens[0]
        pred_set = set(pred_tokens)
        if gt_tok not in pred_set:
            return False
        # yes/no 为特殊闭集
        if gt_tok in {"yes", "no"}:
            return len(pred_tokens) <= 3
        return len(pred_tokens) <= 3

    # 短语答案：要求短输出 + 短语级命中（词边界）
    pred_phrase = " ".join(pred_tokens)
    gt_phrase = " ".join(gt_tokens)
    if len(pred_tokens) > max(len(gt_tokens) + 2, 5):
        return False
    return bool(gt_phrase) and re.search(rf"\b{re.escape(gt_phrase)}\b", pred_phrase) is not None


def load_results(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Result file must be a JSON array: {path}")
    return data


def method_stats(method_name: str, rows):
    total = len(rows)
    success_rows = [r for r in rows if r.get("status", "success") == "success"]
    success = len(success_rows)
    failed = total - success

    correct_exact = 0
    correct_soft = 0
    for r in rows:
        pred = r.get("prediction", "")
        gt = r.get("ground_truth", r.get("answer", ""))
        exact = answers_equivalent(pred, gt)
        soft = soft_match(pred, gt)
        correct_exact += 1 if exact else 0
        correct_soft += 1 if soft else 0

    exact_acc = (correct_exact / total * 100.0) if total > 0 else 0.0
    soft_acc = (correct_soft / total * 100.0) if total > 0 else 0.0
    return {
        "method": method_name,
        "total": total,
        "success": success,
        "failed": failed,
        "correct_exact": correct_exact,
        "exact_acc": exact_acc,
        "correct_soft": correct_soft,
        "soft_acc": soft_acc,
    }


def to_markdown_table(stats_rows):
    lines = []
    lines.append("| Method | Total | Success | Failed | Exact Correct | Exact Acc(%) | Soft Correct | Soft Acc(%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats_rows:
        lines.append(
            f"| {s['method']} | {s['total']} | {s['success']} | {s['failed']} | "
            f"{s['correct_exact']} | {s['exact_acc']:.2f} | {s['correct_soft']} | {s['soft_acc']:.2f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Standard/CoT/ToT/GoT on GQA results.")
    parser.add_argument(
        "--standard_file",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/standard_results.json",
    )
    parser.add_argument(
        "--cot_file",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/cot_results.json",
    )
    parser.add_argument(
        "--tot_file",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/tot_results.json",
    )
    parser.add_argument(
        "--got_file",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/got_results.json",
    )
    parser.add_argument(
        "--modular_file",
        type=str,
        default="",
        help="Optional modular reasoning results json.",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/baseline_comparison.md",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/baseline_comparison.csv",
    )
    args = parser.parse_args()

    pairs = [
        ("Standard", args.standard_file),
        ("CoT", args.cot_file),
        ("ToT", args.tot_file),
        ("GoT", args.got_file),
    ]
    if args.modular_file:
        pairs.append(("Modular(4-Module)", args.modular_file))

    stats_rows = []
    for name, path in pairs:
        rows = load_results(path)
        stats_rows.append(method_stats(name, rows))

    table_md = to_markdown_table(stats_rows)
    print(table_md)

    out_dir = os.path.dirname(args.output_md)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(table_md + "\n")

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "total",
                "success",
                "failed",
                "correct_exact",
                "exact_acc",
                "correct_soft",
                "soft_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"\nSaved markdown table: {args.output_md}")
    print(f"Saved csv table: {args.output_csv}")


if __name__ == "__main__":
    main()
