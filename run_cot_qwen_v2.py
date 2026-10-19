# CoT(Qwen 双模型版): LLaVA 视觉描述 -> Qwen 带推理链回答
# 与 ToT/GoT/Modular 同架构（LLaVA+Qwen），只差推理策略
import json
import os
import time
import argparse
from tot_got_integration import VisionToTextReasoner
from vqa_eval_utils import answers_equivalent, extract_final_answer

def load_gqa_questions(json_file_path):
    if json_file_path.endswith('.jsonl'):
        questions_list = []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions_list.append(json.loads(line))
        return questions_list
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        questions_list = list(data.values())
    elif isinstance(data, list):
        questions_list = data
    else:
        raise ValueError(f"Unexpected data format in {json_file_path}.")
    if not questions_list:
        raise ValueError(f"No questions found in {json_file_path}.")
    first_item = questions_list[0]
    image_keys = ['image', 'imageId', 'full_image_path']
    id_keys = ['question_id', 'id', 'questionId']
    img_key = next((k for k in image_keys if k in first_item), None)
    id_key = next((k for k in id_keys if k in first_item), None)
    if not img_key:
        raise ValueError(f"Image key not found. Keys: {list(first_item.keys())}")
    for item in questions_list:
        if img_key != 'image' and img_key in item:
            item['image'] = item[img_key]
        if id_key and id_key != 'question_id' and id_key in item:
            item['question_id'] = item[id_key]
    return questions_list

def main():
    parser = argparse.ArgumentParser(description='Run CoT (Qwen dual-model) on GQA.')
    parser.add_argument('--input_dir', type=str,
                        default='/root/autodl-tmp/vision_qa_platform/data/processed_gqa')
    parser.add_argument('--output_file', type=str, default='results/cot_qwen_results.json')
    parser.add_argument('--max_samples', type=int, default=50)
    args = parser.parse_args()

    base = "/root/autodl-tmp/vision_qa_platform"
    reasoner = VisionToTextReasoner(
        vision_model_path=f"{base}/llava-1.5-7b-hf",
        llm_model_path=f"{base}/text_model/qwen2_7b_instruct",
    )
    llm_caller = reasoner.get_llm_caller()

    possible_files = [
        os.path.join(args.input_dir, 'gqa_clean_test.jsonl'),
        os.path.join(args.input_dir, 'testdev_balanced_questions.json'),
        os.path.join(args.input_dir, 'questions', 'testdev_balanced_questions.json')
    ]
    questions_file_path = None
    for p in possible_files:
        if os.path.exists(p):
            questions_file_path = p
            break
    if not questions_file_path:
        raise FileNotFoundError(f"No question file found in {args.input_dir}")

    questions_data = load_gqa_questions(questions_file_path)
    print(f"Loaded {len(questions_data)} questions. Processing top {args.max_samples}...")

    results = []
    for i, item in enumerate(questions_data[:args.max_samples]):
        question_id = item.get('question_id', item.get('id', str(i)))
        question_text = item.get('question', '')
        image_filename = item.get('image', item.get('full_image_path', ''))
        ground_truth = item.get("answer", "")

        if os.path.isabs(image_filename):
            image_path = image_filename
        else:
            image_path = os.path.join(args.input_dir, 'images', image_filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(args.input_dir, image_filename)

        print(f"\n--- [{i+1}/{args.max_samples}] ID: {question_id} ---")
        if not os.path.exists(image_path):
            results.append({"question_id": question_id, "question": question_text,
                            "image": image_filename, "ground_truth": ground_truth,
                            "prediction": "Error: Image not found", "correct": False,
                            "status": "failed"})
            continue

        try:
            visual_desc = reasoner.get_visual_description(image_path, question_text)
            # Qwen 带推理链回答
            prompt = (
                f"Visual Description:\n{visual_desc}\n\n"
                f"Question:\n{question_text}\n"
                "Let's think step by step. At the end, state your final answer in a "
                "single word or short phrase in English after 'Final answer:'."
            )
            raw = llm_caller(prompt, do_sample=False, max_new_tokens=256)
            answer_pred = extract_final_answer(raw) or raw.strip()
            print(f"Pred: {answer_pred} | GT: {ground_truth}")
            is_correct = answers_equivalent(answer_pred, ground_truth)
            results.append({
                "question_id": question_id, "question": question_text,
                "image": image_filename, "ground_truth": ground_truth,
                "visual_description": visual_desc,
                "prediction": answer_pred,
                "raw_prediction": raw,
                "correct": is_correct, "status": "success",
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({"question_id": question_id, "question": question_text,
                            "image": image_filename, "ground_truth": ground_truth,
                            "prediction": f"Error: {str(e)}", "correct": False,
                            "status": "failed"})

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()
