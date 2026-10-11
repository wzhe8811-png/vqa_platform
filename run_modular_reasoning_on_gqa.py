import argparse
import json
import os

from tot_got_integration import VisionToTextReasoner
from reasoning_modules import (
    QuestionDecomposer,
    ThoughtGenerator,
    StateEvaluator,
    SearchController,
    exact_match,
)
from vqa_eval_utils import is_complex_question, constrained_finalize_answer


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
        questions_list = list(data.values())
    elif isinstance(data, list):
        questions_list = data
    else:
        raise ValueError(f"Unexpected data format in {json_file_path}.")

    # 标准化键名
    first_item = questions_list[0]
    image_keys = ["image", "imageId", "full_image_path"]
    id_keys = ["question_id", "id", "questionId"]
    img_key = next((k for k in image_keys if k in first_item), None)
    id_key = next((k for k in id_keys if k in first_item), None)
    for item in questions_list:
        if img_key and img_key != "image" and img_key in item:
            item["image"] = item[img_key]
        if id_key and id_key != "question_id" and id_key in item:
            item["question_id"] = item[id_key]
    return questions_list


def main():
    parser = argparse.ArgumentParser(description="Run modular 4-component reasoning pipeline on GQA.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/data/processed_gqa",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/modular_reasoning_results.json",
    )
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--n_subquestions", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--complex_only", action="store_true")
    args = parser.parse_args()

    vision_model_path = "/root/autodl-tmp/vision_qa_platform/llava-1.5-7b-hf"
    llm_model_path = "/root/autodl-tmp/vision_qa_platform/text_model/qwen2_1_5b_instruct"
    if not os.path.exists(vision_model_path):
        raise FileNotFoundError(f"Vision model not found at {vision_model_path}")
    if not os.path.exists(llm_model_path):
        raise FileNotFoundError(f"LLM model not found at {llm_model_path}")

    reasoner = VisionToTextReasoner(
        vision_model_path=vision_model_path,
        llm_model_path=llm_model_path,
    )
    llm_caller = reasoner.get_llm_caller()

    decomposer = QuestionDecomposer(llm_caller)
    generator = ThoughtGenerator(llm_caller)
    evaluator = StateEvaluator(llm_caller)
    controller = SearchController(decomposer, generator, evaluator, llm_caller)

    possible_files = [
        os.path.join(args.input_dir, "gqa_clean_test.jsonl"),
        os.path.join(args.input_dir, "testdev_balanced_questions.json"),
        os.path.join(args.input_dir, "questions", "testdev_balanced_questions.json"),
    ]
    questions_file_path = next((p for p in possible_files if os.path.exists(p)), None)
    if not questions_file_path:
        raise FileNotFoundError(f"No question file found in {args.input_dir}")
    questions_data = load_gqa_questions(questions_file_path)
    if args.complex_only:
        questions_data = [x for x in questions_data if is_complex_question(x.get("question", ""))]
        print(f"🧩 Complex-only enabled. Remaining questions: {len(questions_data)}")

    results = []
    for i, item in enumerate(questions_data[: args.max_samples]):
        qid = item.get("question_id", item.get("id", str(i)))
        question = item.get("question", "")
        gt = item.get("answer", "")
        image_name = item.get("image", item.get("full_image_path", ""))
        if os.path.isabs(image_name):
            image_path = image_name
        else:
            image_path = os.path.join(args.input_dir, "images", image_name)
            if not os.path.exists(image_path):
                image_path = os.path.join(args.input_dir, image_name)

        print(f"\n--- [{i+1}/{args.max_samples}] ID: {qid} ---")
        print(f"Q: {question}")
        print(f"IMG: {image_path}")

        if not os.path.exists(image_path):
            results.append(
                {
                    "question_id": str(qid),
                    "question": question,
                    "image": image_name,
                    "ground_truth": gt,
                    "prediction": "Error: Image not found",
                    "correct": False,
                    "status": "failed",
                }
            )
            continue

        visual_desc = reasoner.get_visual_description(image_path, question)
        solved = controller.solve(
            visual_description=visual_desc,
            question=question,
            n_subquestions=args.n_subquestions,
            n_candidates=args.n_candidates,
        )
        pred = constrained_finalize_answer(
            llm_caller,
            visual_desc,
            question,
            solved["final_answer"],
        )
        ok = exact_match(pred, gt)
        print(f"Pred: {pred} | GT: {gt}")

        results.append(
            {
                "question_id": str(qid),
                "question": question,
                "image": image_name,
                "ground_truth": gt,
                "visual_description": visual_desc,
                "prediction": pred,
                "correct": ok,
                "status": "success",
                "trace": solved["trace"],
                "selected_sub_answers": solved["selected_sub_answers"],
            }
        )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved modular reasoning results to: {args.output_file}")


if __name__ == "__main__":
    main()
