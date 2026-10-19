# 100样本消融统计 + 交叉分析
import json, os, sys
sys.path.insert(0, "/root/autodl-tmp/vision_qa_platform")
from vqa_eval_utils import extract_final_answer, canonicalize_answer, answers_equivalent

RES = "/root/autodl-tmp/vision_qa_platform/results/accuracy"
FILES = {
    "完整四模块(full)": "modular_full_100.json",
    "缺分解": "modular_abl100_nodecompose.json",
    "缺生成": "modular_abl100_nogenerate.json",
    "缺评分(真随机)": "modular_abl100_noevaluate.json",
    "缺聚合": "modular_abl100_noaggregate.json",
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
print("=== 100样本消融准确率 ===")
for label, fn in FILES.items():
    d = load(fn)
    c = sum(1 for it in d if judge(it))
    data[label] = d
    print(f"{label}: {c}/{len(d)} = {c/len(d)*100:.1f}%")

# 与full的逐题差异
full = data["完整四模块(full)"]
print()
for label in ["缺分解", "缺生成", "缺评分(真随机)", "缺聚合"]:
    other = data[label]
    gain = [a for a, b in zip(full, other) if judge(a) and not judge(b)]
    lose = [a for a, b in zip(full, other) if not judge(a) and judge(b)]
    print(f"{label}: full挽回={len(gain)}题 {[a['question_id'] for a in gain]} | full改错={len(lose)}题 {[a['question_id'] for a in lose]}")

# 前50样本子集对比(与旧50结果对齐)
print()
print("=== 前50样本子集(与旧50口径对齐) ===")
for label in FILES:
    d = data[label][:50]
    c = sum(1 for it in d if judge(it))
    print(f"{label} [50子集]: {c}/50 = {c/50*100:.1f}%")

# 后50样本子集
print()
print("=== 后50样本子集 ===")
for label in FILES:
    d = data[label][50:]
    c = sum(1 for it in d if judge(it))
    print(f"{label} [50子集]: {c}/{len(d)} = {c/len(d)*100:.1f}%")
