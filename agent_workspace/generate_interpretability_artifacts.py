import csv
import json
from pathlib import Path
from statistics import mean


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def esc(s: str, limit=80) -> str:
    return str(s or "").replace('"', '\\"')[:limit]


def pick_cases(samples, k=8):
    # Prefer wrong predictions first (more explanatory value), then fill with correct ones.
    wrong = [x for x in samples if not x.get("correct", False)]
    right = [x for x in samples if x.get("correct", False)]
    picked = wrong[:k]
    if len(picked) < k:
        picked.extend(right[: k - len(picked)])
    return picked


def trace_depth(trace):
    return len(trace or [])


def branch_factor(trace):
    if not trace:
        return 0.0
    return mean(len(t.get("candidates", [])) for t in trace)


def confidence(trace):
    scores = []
    for t in trace or []:
        for s in t.get("scores", []):
            scores.append(safe_float(s))
    return mean(scores) if scores else 0.0


def selected_quality(selected_sub_answers):
    if not selected_sub_answers:
        return {"empty_ratio": 1.0, "avg_len": 0.0}
    answers = [str(x.get("selected_answer", "")).strip() for x in selected_sub_answers]
    empties = sum(1 for a in answers if not a)
    avg_len = mean(len(a.split()) for a in answers) if answers else 0.0
    return {"empty_ratio": empties / len(answers), "avg_len": avg_len}


def make_mermaid_for_case(case, max_nodes=18):
    qid = case.get("question_id", "NA")
    trace = case.get("trace", [])
    lines = ["```mermaid", "graph TD"]
    lines.append(f'Q["Q{qid}: {esc(case.get("question", ""), 80)}"]')

    node_count = 0
    for i, t in enumerate(trace):
        if node_count >= max_nodes:
            break
        subq = esc(t.get("sub_question", ""), 60)
        sub_id = f"S{i}"
        lines.append(f'{sub_id}["SubQ{i+1}: {subq}"]')
        lines.append(f"Q --> {sub_id}")

        cands = t.get("candidates", [])
        scores = t.get("scores", [])
        sel = int(t.get("selected_index", 0)) if cands else -1
        for j, c in enumerate(cands[:2]):
            if node_count >= max_nodes:
                break
            ans = esc(c.get("answer", ""), 44)
            sc = safe_float(scores[j]) if j < len(scores) else 0.0
            cid = f"C{i}_{j}"
            prefix = "★" if j == sel else ""
            lines.append(f'{cid}["{prefix}Cand{j+1}: {ans} ({sc:.2f})"]')
            lines.append(f"{sub_id} --> {cid}")
            node_count += 1

    pred = esc(case.get("prediction", ""), 70)
    gt = esc(case.get("ground_truth", ""), 70)
    lines.append(f'P["Pred: {pred}"]')
    lines.append(f'G["GT: {gt}"]')
    lines.append("Q --> P")
    lines.append("Q --> G")
    lines.append("```")
    return "\n".join(lines)


def main():
    root = Path("/root/autodl-tmp/vision_qa_platform")
    results = root / "results"

    # Use completed robustness complex50 clean level for interpretability on hard set.
    in_path = results / "modular_gaussian_0.00.json"
    samples = load_json(in_path)

    metrics_rows = []
    for s in samples:
        tr = s.get("trace", [])
        sel = s.get("selected_sub_answers", [])
        q = selected_quality(sel)
        pred = normalize_text(s.get("prediction", ""))
        gt = normalize_text(s.get("ground_truth", ""))
        metrics_rows.append(
            {
                "question_id": s.get("question_id", ""),
                "correct_exact": bool(s.get("correct", False)),
                "depth_subquestions": trace_depth(tr),
                "avg_branch_factor": round(branch_factor(tr), 4),
                "avg_candidate_score": round(confidence(tr), 4),
                "selected_empty_ratio": round(q["empty_ratio"], 4),
                "selected_avg_token_len": round(q["avg_len"], 4),
                "prediction_equals_gt_norm": pred == gt,
            }
        )

    out_metrics = results / "interpretability_metrics_complex50.csv"
    with out_metrics.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    cases = pick_cases(samples, k=8)
    out_cases = results / "figure4_interpretability_cases_complex50.md"
    lines = ["# Figure 4: 思维树/图可视化案例（complex50）", ""]
    for idx, c in enumerate(cases, start=1):
        lines.append(f"## Case {idx} | QID={c.get('question_id','')}")
        lines.append(f"- Question: {c.get('question','')}")
        lines.append(f"- GT: `{c.get('ground_truth','')}` | Pred: `{c.get('prediction','')}` | Correct: `{c.get('correct', False)}`")
        lines.append("")
        lines.append(make_mermaid_for_case(c))
        lines.append("")
    out_cases.write_text("\n".join(lines), encoding="utf-8")

    # Human scoring template (for manual annotation by peers).
    out_human = results / "interpretability_human_scoring_template.csv"
    with out_human.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question_id",
                "question",
                "prediction",
                "ground_truth",
                "logic_coherence_score_1to5",
                "evidence_relevance_score_1to5",
                "redundancy_score_1to5",
                "overall_interpretability_score_1to5",
                "annotator_notes",
            ]
        )
        for c in cases:
            writer.writerow(
                [
                    c.get("question_id", ""),
                    c.get("question", ""),
                    c.get("prediction", ""),
                    c.get("ground_truth", ""),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    print(f"WROTE: {out_metrics}")
    print(f"WROTE: {out_cases}")
    print(f"WROTE: {out_human}")


if __name__ == "__main__":
    main()
