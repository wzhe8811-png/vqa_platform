# v10 补丁: no_evaluate 改为真随机均匀选候选 + 真正跳过评分器调用
BASE = "/root/autodl-tmp/vision_qa_platform"
p1 = f"{BASE}/reasoning_modules.py"
s = open(p1, encoding="utf-8").read()

# 1. 顶部加 import random
old_h = """import json
import re
from typing import Any, Dict, List, Optional
"""
new_h = """import json
import random
import re
from typing import Any, Dict, List, Optional
"""
assert s.count(old_h) == 1, f"head anchor={s.count(old_h)}"
s = s.replace(old_h, new_h)

# 2. 跳过评分器调用 + 真随机选候选
old_ev = """            scores = self.evaluator.score_candidates(
                visual_description=visual_description,
                sub_question=sub_q,
                candidates=candidates,
                visual_sub_answer=visual_sub_answer,
            )
            if ablation == "no_generate" and len(candidates) == 1:
                scores = [1.0]
            if ablation == "no_evaluate":
                # 消融: 跳过评分模块，等同所有候选同分（选候选0）
                scores = [0.5] * len(candidates)

            best_idx = 0
            if scores:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best = candidates[best_idx]
"""
new_ev = """            if ablation == "no_evaluate":
                # 消融: 跳过评分模块，真随机均匀选一个候选(种子按子问题序号变化, 可复现)
                scores = [0.5] * len(candidates)
                best_idx = random.Random(2026 + len(selected_sub_answers)).randint(0, len(candidates) - 1)
            else:
                scores = self.evaluator.score_candidates(
                    visual_description=visual_description,
                    sub_question=sub_q,
                    candidates=candidates,
                    visual_sub_answer=visual_sub_answer,
                )
                if ablation == "no_generate" and len(candidates) == 1:
                    scores = [1.0]
                best_idx = 0
                if scores:
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best = candidates[best_idx]
"""
assert s.count(old_ev) == 1, f"ev anchor={s.count(old_ev)}"
s = s.replace(old_ev, new_ev)

open(p1, "w", encoding="utf-8").write(s)
print("PATCHED reasoning_modules.py (v10 clean ablation)", flush=True)
