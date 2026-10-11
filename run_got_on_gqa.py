import os
import json
import argparse
from tot_got_integration import VisionToTextReasoner
from vqa_eval_utils import is_complex_question, constrained_finalize_answer, answers_equivalent


def load_gqa_questions(json_file_path: str):
    """从 JSON 文件加载 GQA 问题 (兼容 jsonl 和 json)，并标准化键名。"""
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
        questions_list = list(data.values())
    elif isinstance(data, list):
        questions_list = data
    else:
        raise ValueError(f"Unexpected data format in {json_file_path}.")

    if not questions_list:
        raise ValueError(f"No questions found in {json_file_path}.")

    # 标准化键名
    first_item = questions_list[0]
    image_keys = ["image", "imageId", "full_image_path"]
    id_keys = ["question_id", "id", "questionId"]

    img_key = next((k for k in image_keys if k in first_item), None)
    id_key = next((k for k in id_keys if k in first_item), None)
    if not img_key:
        raise ValueError(f"Image key not found. Keys: {list(first_item.keys())}")

    for item in questions_list:
        if img_key != "image" and img_key in item:
            item["image"] = item[img_key]
        if id_key and id_key != "question_id" and id_key in item:
            item["question_id"] = item[id_key]
    return questions_list


def _safe_json_extract(text: str):
    """尽量从模型输出中提取 JSON（list/dict）。"""
    text = (text or "").strip()
    # 优先完整 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再尝试截取数组
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    # 再尝试截取对象
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return None


def solve_got_for_gqa(
    llm_caller,
    visual_description: str,
    question_text: str,
    num_subquestions: int = 3,
    num_candidates: int = 2,
):
    """
    一个可跑通的 GoT 基线：
    1) 用 Qwen 将主问题分解为若干子问题（子图节点）
    2) 分别回答每个子问题（子节点回答）
    3) 聚合子答案得到最终答案（图到答案）
    4) 生成多个分解候选后再让模型打分选最优（增强鲁棒）
    """

    initial_context = (
        f"Visual Description:\n{visual_description}\n\n"
        f"Question:\n{question_text}\n"
    )

    candidates = []
    for cand_i in range(num_candidates):
        # 1) 分解子问题
        decompose_prompt = (
            initial_context
            + "Decompose the question into "
            + f"{num_subquestions} sub-questions that help solve it.\n"
            + "Return ONLY a JSON array of strings (each string is one sub-question).\n"
            + "Example: [\"sub q1\",\"sub q2\",\"sub q3\"]"
        )
        decompose_text = llm_caller(decompose_prompt)
        subqs = _safe_json_extract(decompose_text)
        if not isinstance(subqs, list) or len(subqs) == 0:
            # 兜底：按换行切分
            subqs = [line.strip("- ").strip() for line in decompose_text.splitlines() if line.strip()]
            subqs = subqs[:num_subquestions]

        # 2) 逐个子问题回答
        sub_answers = []
        for subq in subqs[:num_subquestions]:
            sub_prompt = (
                initial_context
                + "Answer the following sub-question using ONLY the provided visual description.\n"
                + f"Sub-question: {subq}\n"
                + "Return ONLY the sub-answer as a short phrase."
            )
            sub_ans = llm_caller(sub_prompt).strip()
            sub_answers.append({"sub_question": subq, "answer": sub_ans})

        # 3) 聚合生成最终答案
        answers_text = "\n".join([f"- {x['sub_question']}: {x['answer']}" for x in sub_answers])
        aggregate_prompt = (
            initial_context
            + "Using the sub-question answers below, produce the final answer to the ORIGINAL question.\n"
            + "Return ONLY the final answer text (no reasoning).\n\n"
            + f"Sub-question answers:\n{answers_text}"
        )
        final_ans = llm_caller(aggregate_prompt).strip()
        candidates.append({"subqs": subqs, "final_answer": final_ans})

    if len(candidates) == 1:
        return candidates[0]["final_answer"]

    # 4) 让模型在多个候选最终答案之间做选择
    candidates_text = "\n".join(
        [f"Candidate {i+1}: {c['final_answer']}" for i, c in enumerate(candidates)]
    )
    score_prompt = (
        initial_context
        + "You will be given multiple candidate final answers for the same VQA question.\n"
        + "Choose the best candidate (most correct) and return ONLY its 1-based index as an integer.\n\n"
        + f"{candidates_text}"
    )
    score_text = llm_caller(score_prompt)
    # 提取整数索引
    import re

    m = re.search(r"([1-9][0-9]?)", score_text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]["final_answer"]

    # 兜底：返回第一个
    return candidates[0]["final_answer"]


def main():
    parser = argparse.ArgumentParser(description="Run GoT on GQA dataset (baseline).")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/data/processed_gqa",
        help="Directory containing processed GQA data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/got_results.json",
        help="Path to save results JSON.",
    )
    parser.add_argument("--max_samples", type=int, default=5, help="Max samples to process for testing.")
    parser.add_argument("--complex_only", action="store_true", help="Evaluate only complex questions.")
    args = parser.parse_args()

    vision_model_path = "/root/autodl-tmp/vision_qa_platform/llava-1.5-7b-hf"
    llm_model_path = "/root/autodl-tmp/vision_qa_platform/text_model/qwen2_1_5b_instruct"

    if not os.path.exists(vision_model_path):
        raise FileNotFoundError(f"Vision model not found at {vision_model_path}")
    if not os.path.exists(llm_model_path):
        raise FileNotFoundError(f"LLM model not found at {llm_model_path}")

    print("🚀 Initializing VisionToTextReasoner...")
    reasoner = VisionToTextReasoner(
        vision_model_path=vision_model_path,
        llm_model_path=llm_model_path,
    )
    print("✅ Reasoner initialized successfully!")

    possible_files = [
        os.path.join(args.input_dir, "gqa_clean_test.jsonl"),
        os.path.join(args.input_dir, "testdev_balanced_questions.json"),
        os.path.join(args.input_dir, "questions", "testdev_balanced_questions.json"),
    ]
    questions_file_path = None
    for p in possible_files:
        if os.path.exists(p):
            questions_file_path = p
            break

    if not questions_file_path:
        raise FileNotFoundError(f"No question file found in {args.input_dir}")

    print(f"📂 Loading questions from: {questions_file_path}")
    questions_data = load_gqa_questions(questions_file_path)
    if args.complex_only:
        questions_data = [x for x in questions_data if is_complex_question(x.get("question", ""))]
        print(f"🧩 Complex-only enabled. Remaining questions: {len(questions_data)}")
    print(f"✅ Loaded {len(questions_data)} questions. Processing top {args.max_samples}...")

    results = []
    for i, item in enumerate(questions_data[: args.max_samples]):
        question_id = item.get("question_id", item.get("id", str(i)))
        question_text = item.get("question", "")
        image_filename = item.get("image", item.get("full_image_path", ""))
        ground_truth = item.get("answer", "")

        if os.path.isabs(image_filename):
            image_path = image_filename
        else:
            image_path = os.path.join(args.input_dir, "images", image_filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(args.input_dir, image_filename)

        print(f"\n--- [{i+1}/{args.max_samples}] Processing ID: {question_id} ---")
        print(f"❓ Question: {question_text}")
        print(f"🖼️  Image Path: {image_path}")

        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}, skipping...")
            results.append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "image": image_filename,
                    "ground_truth": ground_truth,
                    "prediction": "Error: Image not found",
                    "correct": False,
                    "status": "failed",
                }
            )
            continue

        try:
            print("👁️  Generating visual description (LLaVA)...")
            visual_desc = reasoner.get_visual_description(image_path, question_text)
            print(f"   Description: {visual_desc[:100]}...")

            print("🌳 Running Graph of Thoughts (Qwen)...")
            pred = solve_got_for_gqa(
                reasoner.get_llm_caller(),
                visual_desc,
                question_text,
                num_subquestions=3,
                num_candidates=2,
            )
            pred = constrained_finalize_answer(
                reasoner.get_llm_caller(),
                visual_desc,
                question_text,
                pred,
            )
            print(f"✅ GoT Answer: {pred}")
            is_correct = answers_equivalent(pred, ground_truth)
            print(f"📊 Pred: {pred} | GT: {ground_truth} | Correct: {is_correct}")

            results.append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "image": image_filename,
                    "ground_truth": ground_truth,
                    "visual_description": visual_desc,
                    "prediction": pred,
                    "correct": is_correct,
                    "status": "success",
                }
            )
        except Exception as e:
            print(f"❌ Error running GoT: {e}")
            results.append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "image": image_filename,
                    "ground_truth": ground_truth,
                    "prediction": f"Error: {str(e)}",
                    "correct": False,
                    "status": "failed",
                }
            )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
