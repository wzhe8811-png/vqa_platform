# 交叉分析: 鲁棒性3次一致性 + 消融差异题定位
import json, os, sys
sys.path.insert(0, "/root/autodl-tmp/vision_qa_platform")
from vqa_eval_utils import extract_final_answer, canonicalize_answer, answers_equivalent

RES = "/root/autodl-tmp/vision_qa_platform/results/accuracy"

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

# 1. 鲁棒性3次逐题一致性
r1, r2, r3 = load("modular_7b_eng_test50.json"), load("modular_7b_robust_run2.json"), load("modular_7b_robust_run3.json")
always_c = always_w = fluc = 0
fluc_ids = []
for a, b, d in zip(r1, r2, r3):
    ja, jb, jd = judge(a), judge(b), judge(d)
    if ja and jb and jd:
        always_c += 1
    elif not ja and not jb and not jd:
        always_w += 1
    else:
        fluc += 1
        fluc_ids.append((a["question_id"], ja, jb, jd))
print(f"鲁棒性一致性(50题×3次): 全对={always_c}, 全错={always_w}, 波动={fluc}")
print(f"波动题: {fluc_ids}")
print(f"-> 稳定率(3次结果完全一致) = {(always_c+always_w)/50*100:.0f}%")

# 2. 聚合模块挽回的题 (full对 vs no_aggregate错)
full = load("modular_7b_eng_test50.json")
noagg = load("modular_ablation_noaggregate_test50.json")
print()
print("=== 聚合模块挽回的题 (完整版对 / 缺聚合错) ===")
for a, b in zip(full, noagg):
    if judge(a) and not judge(b):
        print(f"  Q{a['question_id']}: {a['question']}")
        print(f"    GT={a['ground_truth']} | full答案={extract_final_answer(a['prediction'])[:50]} | noagg答案={extract_final_answer(b['prediction'])[:50]}")
print("=== 聚合模块改错的题 (完整版错 / 缺聚合对) ===")
for a, b in zip(full, noagg):
    if not judge(a) and judge(b):
        print(f"  Q{a['question_id']}: {a['question']}")

# 3. 缺分解/缺生成的差异题
print()
for label, fn in [("缺分解", "modular_ablation_nodecompose_test50.json"),
                  ("缺生成", "modular_ablation_nogenerate_test50.json"),
                  ("缺评分", "modular_ablation_noevaluate_test50.json")]:
    other = load(fn)
    gain = [a for a, b in zip(full, other) if judge(a) and not judge(b)]
    lose = [a for a, b in zip(full, other) if not judge(a) and judge(b)]
    print(f"{label}: full挽回={len(gain)}题 {[a['question_id'] for a in gain]} | full改错={len(lose)}题 {[a['question_id'] for a in lose]}")
