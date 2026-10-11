import json
import os
from PIL import Image
import time
import torch
from tot_got_integration import VisionToTextReasoner
import argparse
from vqa_eval_utils import is_complex_question, constrained_finalize_answer, answers_equivalent

def load_gqa_questions(json_file_path):
    """从 JSON 文件加载 GQA 问题 (兼容 jsonl 和 json)"""
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

    # 标准化键名
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


def solve_tot_for_gqa(llm_caller, visual_description, question_text, task="tot"):
    """
    使用“简化思维树”(ToT-like) 解决 GQA 问题

    说明：
    当前环境的 `tree_of_thoughts` 包版本不提供 `TreeOfThoughts/GraphOfThoughts` 等类，
    为了保证 ToT 基线可运行，这里实现一个等价的流程：
    1) 提出多条候选推理路径(含候选答案)
    2) 让模型对候选路径打分
    3) 选择分数最高的路径
    4) 从最优路径中抽取最终简洁答案
    """
    # 给基线一个较小的预算，便于单样本跑通
    num_branches = 3
    num_select = 2

    def safe_json_loads(text):
        try:
            return json.loads(text)
        except Exception:
            # 尝试截取数组/对象片段
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
            return None

    initial_context = (
        f"Visual Description:\n{visual_description}\n\n"
        f"Question:\n{question_text}\n"
    )

    # 1) 提出多条候选推理路径（每条包含 thought + answer）
    propose_prompt = (
        initial_context
        + "You are solving a Visual Question Answering (VQA) problem.\n"
        + f"Generate {num_branches} different reasoning paths.\n"
        + "For each path, include a short reasoning and a candidate final answer.\n"
        + "Output ONLY a JSON array of objects with exactly these keys: "
        + '"thought" (string) and "answer" (string). Example: '
        + '[{"thought":"...","answer":"..."}, ...]'
    )
    propose_text = llm_caller(propose_prompt)
    proposals = safe_json_loads(propose_text)

    # 解析失败时的兜底：把整段当作“一个候选”
    if not isinstance(proposals, list) or len(proposals) == 0:
        proposals = [{"thought": propose_text.strip(), "answer": propose_text.strip()}]
        num_branches = len(proposals)

    # 2) 评分
    # 为了让解析更稳，评分也要求输出 JSON
    candidates_str = "\n".join(
        [f"Candidate {i+1}:\nThought: {p.get('thought','')}\nAnswer: {p.get('answer','')}\n"
         for i, p in enumerate(proposals)]
    )
    score_prompt = (
        initial_context
        + "Now score each candidate path for correctness and consistency with the question and visual description.\n"
        + "Return ONLY a JSON array of floats between 0.1 and 1.0, in the same order as candidates.\n\n"
        + candidates_str
    )
    score_text = llm_caller(score_prompt)
    scores = safe_json_loads(score_text)

    if not isinstance(scores, list) or len(scores) != len(proposals):
        # 兜底：从文本里尽量提取浮点数
        import re as _re
        floats = _re.findall(r"(0\\.\\d+|1\\.0|[0-9]\\.\\d+)", score_text)
        scores = []
        for f in floats[: len(proposals)]:
            try:
                scores.append(float(f))
            except Exception:
                pass
        if len(scores) != len(proposals):
            scores = [0.5 for _ in range(len(proposals))]

    # 3) 选择最优候选
    ranked_idx = sorted(range(len(proposals)), key=lambda i: scores[i], reverse=True)
    selected_idx = ranked_idx[: max(1, min(num_select, len(ranked_idx)))]
    best_idx = selected_idx[0]  # top1
    best = proposals[best_idx]

    # 4) 抽取最终简洁答案
    extraction_prompt = (
        initial_context
        + "Use the best reasoning path below to provide a FINAL answer.\n"
        + "Return ONLY the final answer text (no explanation).\n\n"
        + f"Best thought:\n{best.get('thought','')}\n\n"
        + f"Best candidate answer:\n{best.get('answer','')}\n"
    )
    final_answer = llm_caller(extraction_prompt)
    return final_answer.strip()

def main():
    parser = argparse.ArgumentParser(description='Run ToT on GQA dataset.')
    parser.add_argument('--input_dir', type=str, default='/root/autodl-tmp/vision_qa_platform/data/processed_gqa', help='Directory containing GQA data.')
    parser.add_argument('--output_file', type=str, default='results/tot_results.json', help='Path to save results.')
    parser.add_argument('--max_samples', type=int, default=5, help='Max samples to process for testing.')
    parser.add_argument('--complex_only', action='store_true', help='Evaluate only complex questions.')

    args = parser.parse_args()

    # ✅ 配置模型路径
    # 1. 视觉模型：使用 LLaVA 进行图像理解
    vision_model_path = "/root/autodl-tmp/vision_qa_platform/llava-1.5-7b-hf"
    
    # 2. 语言模型：使用 Qwen2-1.5B-Instruct 进行逻辑推理 (ToT 核心)
    llm_model_path = "/root/autodl-tmp/vision_qa_platform/text_model/qwen2_1_5b_instruct"
    
    # 检查路径
    if not os.path.exists(vision_model_path):
        print(f"❌ Error: Vision model not found at {vision_model_path}")
        return
    if not os.path.exists(llm_model_path):
        print(f"❌ Error: LLM model not found at {llm_model_path}")
        return

    print(f"🚀 Initializing VisionToTextReasoner...")
    print(f"   👁️  Vision Model (LLaVA): {vision_model_path}")
    print(f"   🧠 LLM Model (Qwen):      {llm_model_path}")
    
    # ✅ 初始化 Reasoner，传入双模型路径
    try:
        reasoner = VisionToTextReasoner(
            vision_model_path=vision_model_path,
            llm_model_path=llm_model_path
        )
        print("✅ Reasoner initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize Reasoner: {e}")
        import traceback
        traceback.print_exc()
        return

    # 确定问题文件路径
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
        print(f"❌ Error: No question file found in {args.input_dir}")
        return

    print(f"📂 Loading questions from: {questions_file_path}")
    questions_data = load_gqa_questions(questions_file_path)
    if args.complex_only:
        questions_data = [x for x in questions_data if is_complex_question(x.get("question", ""))]
        print(f"🧩 Complex-only enabled. Remaining questions: {len(questions_data)}")
    print(f"✅ Loaded {len(questions_data)} questions. Processing top {args.max_samples}...")

    results = []
    items_to_process = questions_data[:args.max_samples]

    for i, item in enumerate(items_to_process):
        question_id = item.get('question_id', item.get('id', str(i)))
        question_text = item.get('question', '')
        image_filename = item.get('image', item.get('full_image_path', ''))
        ground_truth = item.get("answer", "")
        
        # 构建图片绝对路径
        if os.path.isabs(image_filename):
            image_path = image_filename
        else:
            image_path = os.path.join(args.input_dir, 'images', image_filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(args.input_dir, image_filename)

        print(f"\n--- [{i+1}/{args.max_samples}] Processing ID: {question_id} ---")
        print(f"❓ Question: {question_text}")
        print(f"🖼️  Image Path: {image_path}")

        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}, skipping...")
            results.append({
                "question_id": question_id,
                "question": question_text,
                "image": image_filename,
                "ground_truth": ground_truth,
                "prediction": "Error: Image not found",
                "correct": False,
                "status": "failed"
            })
            continue

        # 1. 获取视觉描述 (使用 LLaVA)
        try:
            print("👁️  Generating visual description (LLaVA)...")
            visual_desc = reasoner.get_visual_description(image_path, question_text)
            print(f"   Description: {visual_desc[:100]}...")
        except Exception as e:
            print(f"❌ Error getting visual description: {e}")
            visual_desc = "Failed to generate description."

        # 2. 使用 ToT 解决 (使用 Qwen)
        try:
            print("🌳 Running Tree of Thoughts (Qwen)...")
            answer_pred = solve_tot_for_gqa(reasoner.get_llm_caller(), visual_desc, question_text, task="tot")
            answer_pred = constrained_finalize_answer(
                reasoner.get_llm_caller(),
                visual_desc,
                question_text,
                answer_pred,
            )
            print(f"✅ ToT Answer: {answer_pred}")
        except Exception as e:
            print(f"❌ Error running ToT: {e}")
            import traceback
            traceback.print_exc()
            answer_pred = f"Error: {str(e)}"

        is_correct = answers_equivalent(answer_pred, ground_truth)
        print(f"📊 Pred: {answer_pred} | GT: {ground_truth} | Correct: {is_correct}")

        results.append({
            "question_id": question_id,
            "question": question_text,
            "image": image_filename,
            "ground_truth": ground_truth,
            "visual_description": visual_desc,
            "prediction": answer_pred,
            "correct": is_correct,
            "status": "success"
        })

    # 保存结果
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
