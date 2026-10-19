# 100样本统一口径统计: 5方法 + 4消融 (v11全英文)
import json, os, sys
sys.path.insert(0, "/root/autodl-tmp/vision_qa_platform")
from vqa_eval_utils import extract_final_answer, canonicalize_answer, answers_equivalent

RES = "/root/autodl-tmp/vision_qa_platform/results/accuracy"
FILES = {
    "Standard(直答)": "standard_100.json",
    "CoT(链式思维)": "cot_100.json",
    "ToT(思维树)": "tot_100.json",
    "GoT(思维图)": "got_100.json",
    "Modular(四模块)": "modular_full_100_v11.json",
    "消融-缺分解": "modular_abl100_v11_nodecompose.json",
    "消融-缺生成": "modular_abl100_v11_nogenerate.json",
    "消融-缺评分": "modular_abl100_v11_noevaluate.json",
    "消融-缺聚合": "modular_abl100_v11_noaggregate.json",
}

def judge(item):
    pred = item.get("prediction") or ""
    gt = item.get("ground_truth") or ""
    try:
        return answers_equivalent(canonicalize_answer(extract_final_answer(pred)), canonicalize_answer(gt))
    except Exception:
        return False

def load(fn):
    d = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("results") or d.get("samples") or []
    return d

data = {}
print("=== 100样本准确率 (v11 全英文 prompt) ===")
for label, fn in FILES.items():
    d = load(fn)
    c = sum(1 for it in d if judge(it))
    data[label] = d
    print(f"{label}: {c}/{len(d)} = {c/len(d)*100:.1f}%")

full = data["Modular(四模块)"]
print()
print("=== 消融逐题差异 (与四模块完整版对比) ===")
for label in ["消融-缺分解", "消融-缺生成", "消融-缺评分", "消融-缺聚合"]:
    other = data[label]
    gain = [a for a, b in zip(full, other) if judge(a) and not judge(b)]
    lose = [a for a, b in zip(full, other) if not judge(a) and judge(b)]
    print(f"{label}: 完整版挽回={len(gain)}题 {[a['question_id'] for a in gain]} | 完整版改错={len(lose)}题 {[a['question_id'] for a in lose]}")

# 前50/后50 子集稳定性
print()
print("=== 前后50子集稳定性 ===")
for label in FILES:
    d = data[label]
    c1 = sum(1 for it in d[:50] if judge(it))
    c2 = sum(1 for it in d[50:] if judge(it))
    print(f"{label}: 前50={c1/50*100:.0f}% 后50={c2/50*100:.0f}%")
