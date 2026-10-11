# prepare_gqa_data.py
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_gqa(gqa_dir, output_dir):
    """
    从 GQA 原始数据中加载问题，并根据一定规则划分 Clean 和 Noisy 测试集。
    这里只是一个简化示例，您可能需要根据具体研究定义 "noisy"。
    """
    questions_dir = os.path.join(gqa_dir, 'questions')
    
    # 尝试加载验证集问题文件
    # 常见的文件名包括: val_balanced_questions.json, val_all_questions.json
    possible_files = [
        'val_balanced_questions.json',
        'val_all_questions.json'
    ]
    
    val_questions_file = None
    for filename in possible_files:
        candidate_path = os.path.join(questions_dir, filename)
        if os.path.exists(candidate_path):
            val_questions_file = candidate_path
            print(f"🔍 Found question file: {filename}")
            break
    
    if not val_questions_file:
        print(f"❌ Error: Cannot find any expected question file in {questions_dir}.")
        print("   Expected files:", possible_files)
        return

    print(f"🔍 Loading GQA questions from {val_questions_file}...")
    try:
        with open(val_questions_file, 'r') as f:
            questions_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse JSON file {val_questions_file}. {e}")
        return

    # 将数据转换为 DataFrame，方便处理
    df_list = []
    for qid, qinfo in questions_data.items():
        row = {
            'questionId': qid,
            'imageId': qinfo['imageId'],
            'question': qinfo['question'],
            'answer': qinfo.get('answer', ''), # GQA 的 ground truth answer
            'full_image_path': os.path.join(gqa_dir, 'images', qinfo['imageId'] + '.jpg'), # 构建完整路径
            'is_science_related': 'science' in qinfo['question'].lower() or 'physics' in qinfo['question'].lower(),
            # ... 可以根据需要添加更多特征，用于定义 noisy set
        }
        df_list.append(row)

    df = pd.DataFrame(df_list)
    print(f"✅ Loaded {len(df)} questions.")

    # --- 划分 Clean 和 Noisy 测试集 ---
    # 示例：假设我们认为包含 "science" 或 "physics" 的问题是 "noisy" (因为模型可能不擅长)
    # 并非严谨定义，仅作演示。您可以根据自己的研究定义 "noise"。
    clean_df = df[~df['is_science_related']]
    noisy_df = df[df['is_science_related']]

    # 如果 noisy_df 太小，我们可以随机采样一部分作为 noisy，剩下的作为 clean
    if len(noisy_df) < 500: # 假设我们希望 noisy set 至少有 500 个
        print(f"Noisy set is too small ({len(noisy_df)}). Splitting randomly instead.")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        clean_df, noisy_df = train_test_split(test_df, test_size=0.5, random_state=42)

    print(f"📊 Clean Test Set size: {len(clean_df)}")
    print(f"📊 Noisy Test Set size: {len(noisy_df)}")

    # --- 保存数据 ---
    os.makedirs(output_dir, exist_ok=True)

    clean_path = os.path.join(output_dir, 'gqa_clean_test.jsonl')
    noisy_path = os.path.join(output_dir, 'gqa_noisy_test.jsonl')

    clean_df.to_json(clean_path, orient='records', lines=True)
    noisy_df.to_json(noisy_path, orient='records', lines=True)

    print(f"\n✅ Data preparation complete!")
    print(f"   Clean test data saved to: {clean_path}")
    print(f"   Noisy test data saved to: {noisy_path}")

if __name__ == "__main__":
    # --- 配置路径 ---
    # 修改这里的 GQA_DIR 为您的标准 GQA 数据集路径
    GQA_DIR = "/root/autodl-tmp/vision_qa_platform/data/gqa_standard"
    OUTPUT_DIR = "/root/autodl-tmp/vision_qa_platform/data/processed_gqa"

    load_and_prepare_gqa(GQA_DIR, OUTPUT_DIR)
