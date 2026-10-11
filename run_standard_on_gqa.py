import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vqa_eval_utils import is_complex_question, answers_equivalent


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


def main():
    parser = argparse.ArgumentParser(description="Run Standard baseline on GQA.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/data/processed_gqa",
        help="Directory containing processed GQA data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/results/standard_results.json",
        help="Path to save results JSON.",
    )
    parser.add_argument("--max_samples", type=int, default=5, help="Max samples to process.")
    parser.add_argument("--complex_only", action="store_true", help="Evaluate only complex questions.")
    args = parser.parse_args()

    model_id = "/root/autodl-tmp/vision_qa_platform/llava-1.5-7b-hf"
    print("=== 正在加载模型(Standard) ===")
    print(f"📂 模型路径: {model_id}")
    if not os.path.exists(model_id):
        raise FileNotFoundError(f"错误：模型路径不存在！请检查路径: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("✅ 模型加载成功！\n")

    possible_files = [
        os.path.join(args.input_dir, "gqa_clean_test.jsonl"),
        os.path.join(args.input_dir, "testdev_balanced_questions.json"),
        os.path.join(args.input_dir, "questions", "testdev_balanced_questions.json"),
    ]
    questions_file_path = next((p for p in possible_files if os.path.exists(p)), None)
    if not questions_file_path:
        raise FileNotFoundError(f"错误：找不到 GQA 问题文件 in {args.input_dir}")

    print(f"📂 Loading questions from: {questions_file_path}")
    questions_data = load_gqa_questions(questions_file_path)
    if args.complex_only:
        questions_data = [x for x in questions_data if is_complex_question(x.get("question", ""))]
        print(f"🧩 Complex-only enabled. Remaining questions: {len(questions_data)}")
    print(f"✅ Loaded {len(questions_data)} questions. Processing top {args.max_samples}...")

    results = []
    for i, item in enumerate(questions_data[: args.max_samples]):
        question_id = item.get("questionId", item.get("question_id", str(i)))
        question = item.get("question", "")
        ground_truth_answer = item.get("answer", "")
        image_name = item.get("full_image_path", item.get("image", item.get("imageId", "")))

        if os.path.isabs(image_name):
            image_path = image_name
        else:
            image_path = os.path.join(args.input_dir, "images", str(image_name))
            if not os.path.exists(image_path):
                image_path = os.path.join(args.input_dir, str(image_name))

        print(f"\n--- [{i+1}/{args.max_samples}] ID: {question_id} ---")
        print(f"❓ Question: {question}")
        print(f"🖼️  Image Path: {image_path}")

        if not os.path.exists(image_path):
            results.append(
                {
                    "question_id": str(question_id),
                    "question": question,
                    "image": image_name,
                    "ground_truth": ground_truth_answer,
                    "prediction": "Error: Image not found",
                    "correct": False,
                    "status": "failed",
                }
            )
            continue

        image = Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(
            model.device, dtype=torch.float16
        )

        with torch.inference_mode():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        model_answer = generated_text.split("ASSISTANT:")[-1].strip()
        is_correct = answers_equivalent(model_answer, ground_truth_answer)

        print(f"🤖 Pred: {model_answer} | GT: {ground_truth_answer}")
        results.append(
            {
                "question_id": str(question_id),
                "question": question,
                "image": image_name,
                "ground_truth": ground_truth_answer,
                "prediction": model_answer,
                "correct": is_correct,
                "status": "success",
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
