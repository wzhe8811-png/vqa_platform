import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/vision_qa_platform")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reasoning_modules import QuestionDecomposer, ThoughtGenerator, StateEvaluator
from tot_got_integration import VisionToTextReasoner
from vqa_eval_utils import is_complex_question, constrained_finalize_answer, answers_equivalent


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


def load_gqa_questions(json_file_path):
    if json_file_path.endswith(".jsonl"):
        questions_list = []
        with open(json_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions_list.append(json.loads(line))
        return questions_list
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected data format in {json_file_path}")


def find_questions_file(input_dir: str) -> str:
    possible = [
        os.path.join(input_dir, "gqa_clean_test.jsonl"),
        os.path.join(input_dir, "testdev_balanced_questions.json"),
        os.path.join(input_dir, "questions", "testdev_balanced_questions.json"),
    ]
    p = next((x for x in possible if os.path.exists(x)), None)
    if not p:
        raise FileNotFoundError(f"No question file found in {input_dir}")
    return p


def build_image_path(input_dir: str, image_name: str) -> str:
    if os.path.isabs(image_name):
        return image_name
    p = os.path.join(input_dir, "images", image_name)
    if os.path.exists(p):
        return p
    return os.path.join(input_dir, image_name)


def final_answer_from_subanswers(llm_caller, visual_description: str, question: str, sub_answers):
    subqa_text = "\n".join([f"- {x['sub_question']}: {x['answer']}" for x in sub_answers])
    prompt = (
        f"Visual Description:\n{visual_description}\n\n"
        f"Original Question:\n{question}\n\n"
        "Use the selected sub-answers to provide the final answer.\n"
        "Return ONLY the final answer text.\n\n"
        f"Selected sub-answers:\n{subqa_text}"
    )
    return llm_caller(prompt).strip()


def run_mode(mode: str, llm_caller, visual_description: str, question: str, decomposer, generator, evaluator):
    if mode == "full_modular":
        sub_questions = decomposer.decompose(visual_description, question, n_subquestions=3)
        selected = []
        for sq in sub_questions:
            cands = generator.generate_candidates(visual_description, sq, n_candidates=3)
            scores = evaluator.score_candidates(visual_description, sq, cands)
            best_idx = max(range(len(cands)), key=lambda i: scores[i] if i < len(scores) else 0.5) if cands else 0
            best = cands[best_idx] if cands else {"answer": ""}
            selected.append({"sub_question": sq, "answer": best.get("answer", "")})
        draft = final_answer_from_subanswers(llm_caller, visual_description, question, selected)
        return constrained_finalize_answer(llm_caller, visual_description, question, draft)

    if mode == "no_decomposer":
        sq = question
        cands = generator.generate_candidates(visual_description, sq, n_candidates=3)
        scores = evaluator.score_candidates(visual_description, sq, cands)
        best_idx = max(range(len(cands)), key=lambda i: scores[i] if i < len(scores) else 0.5) if cands else 0
        best = cands[best_idx] if cands else {"answer": ""}
        draft = final_answer_from_subanswers(
            llm_caller, visual_description, question, [{"sub_question": sq, "answer": best.get("answer", "")}]
        )
        return constrained_finalize_answer(llm_caller, visual_description, question, draft)

    if mode == "no_evaluator":
        sub_questions = decomposer.decompose(visual_description, question, n_subquestions=3)
        selected = []
        for sq in sub_questions:
            cands = generator.generate_candidates(visual_description, sq, n_candidates=3)
            first = cands[0] if cands else {"answer": ""}
            selected.append({"sub_question": sq, "answer": first.get("answer", "")})
        draft = final_answer_from_subanswers(llm_caller, visual_description, question, selected)
        return constrained_finalize_answer(llm_caller, visual_description, question, draft)

    if mode == "no_searcher":
        prompt = (
            f"Visual Description:\n{visual_description}\n\n"
            f"Question:\n{question}\n\n"
            "Answer the question directly in <= 3 words."
        )
        draft = llm_caller(prompt).strip()
        return constrained_finalize_answer(llm_caller, visual_description, question, draft)

    raise ValueError(f"Unknown mode: {mode}")


def summarize(rows):
    out = []
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        rs = [x for x in rows if x["method"] == m]
        total = len(rs)
        exact = sum(1 for x in rs if x["exact"])
        soft = sum(1 for x in rs if x["soft"])
        out.append(
            {
                "method": m,
                "total": total,
                "correct_exact": exact,
                "exact_acc": (100.0 * exact / total) if total else 0.0,
                "correct_soft": soft,
                "soft_acc": (100.0 * soft / total) if total else 0.0,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for modular reasoning pipeline.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--complex_only", action="store_true")
    parser.add_argument(
        "--output_json",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/ablation_raw_complex50.json",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/ablation_table_complex50.csv",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/ablation_table_complex50.md",
    )
    args = parser.parse_args()

    reasoner = VisionToTextReasoner(
        vision_model_path="/root/autodl-tmp/vision_qa_platform/llava-1.5-7b-hf",
        llm_model_path="/root/autodl-tmp/vision_qa_platform/text_model/qwen2_1_5b_instruct",
    )
    llm_caller = reasoner.get_llm_caller()
    decomposer = QuestionDecomposer(llm_caller)
    generator = ThoughtGenerator(llm_caller)
    evaluator = StateEvaluator(llm_caller)

    q_path = find_questions_file(args.input_dir)
    questions = load_gqa_questions(q_path)
    if args.complex_only:
        questions = [x for x in questions if is_complex_question(x.get("question", ""))]
    questions = questions[: args.max_samples]

    modes = ["full_modular", "no_decomposer", "no_evaluator", "no_searcher"]
    raw_rows = []
    for i, item in enumerate(questions):
        qid = str(item.get("question_id", item.get("id", i)))
        q = item.get("question", "")
        gt = item.get("answer", "")
        img = build_image_path(args.input_dir, item.get("image", item.get("full_image_path", "")))
        print(f"[{i+1}/{len(questions)}] QID={qid}")
        if not os.path.exists(img):
            print(f"  skip missing image: {img}")
            continue
        visual_desc = reasoner.get_visual_description(img, q)
        for m in modes:
            pred = run_mode(m, llm_caller, visual_desc, q, decomposer, generator, evaluator)
            ex = answers_equivalent(pred, gt)
            sf = soft_match(pred, gt)
            raw_rows.append(
                {
                    "question_id": qid,
                    "method": m,
                    "question": q,
                    "ground_truth": gt,
                    "prediction": pred,
                    "exact": ex,
                    "soft": sf,
                }
            )
            print(f"  {m}: pred={pred} | gt={gt} | exact={ex} | soft={sf}")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = summarize(raw_rows)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "total", "correct_exact", "exact_acc", "correct_soft", "soft_acc"],
        )
        writer.writeheader()
        writer.writerows(stats)

    md_lines = []
    md_lines.append("| Method | Total | Exact Correct | Exact Acc(%) | Soft Correct | Soft Acc(%) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for s in stats:
        md_lines.append(
            f"| {s['method']} | {s['total']} | {s['correct_exact']} | {s['exact_acc']:.2f} | {s['correct_soft']} | {s['soft_acc']:.2f} |"
        )
    Path(args.output_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"SAVED: {args.output_json}")
    print(f"SAVED: {args.output_csv}")
    print(f"SAVED: {args.output_md}")


if __name__ == "__main__":
    main()
