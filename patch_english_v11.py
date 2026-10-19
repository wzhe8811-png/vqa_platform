# v11 补丁: 分解/生成/评分三个模块的 prompt 全部英文化, 消除中文污染
BASE = "/root/autodl-tmp/vision_qa_platform"
p1 = f"{BASE}/reasoning_modules.py"
s = open(p1, encoding="utf-8").read()

# 1. 分解模块
old1 = """        prompt = (
            f"图像描述：\\n{visual_description}\\n\\n"
            f"主问题：\\n{question}\\n\\n"
            f"请将主问题拆解为 {n_subquestions} 个简短、可验证的子问题。\\n"
            "要求：\\n"
            "1. 每个子问题必须是完整的疑问句（含问号），不能是单个名词或关键词；\\n"
            "2. 子问题使用中文；\\n"
            "3. 只返回 JSON 字符串数组，不要解释。\\n"
            '示例：["图中左边的物体是什么？", "它是什么颜色？"]'
        )
"""
new1 = """        prompt = (
            f"Image description:\\n{visual_description}\\n\\n"
            f"Question:\\n{question}\\n\\n"
            f"Decompose the question into {n_subquestions} short, verifiable sub-questions.\\n"
            "Requirements:\\n"
            "1. Each sub-question must be a complete question in English (ending with '?'), not a single noun or keyword;\\n"
            "2. Sub-questions must be written in English;\\n"
            "3. Return ONLY a JSON array of strings, no explanation.\\n"
            'Example: ["What is the object on the left?", "What color is it?"]'
        )
"""
assert s.count(old1) == 1, f"anchor1={s.count(old1)}"
s = s.replace(old1, new1)

# 2. 生成模块
old2 = """        visual_ref = (
            f"视觉模型直接答案：{visual_sub_answer}\\n\\n"
            if visual_sub_answer
            else ""
        )
        prompt = (
            f"图像描述：\\n{visual_description}\\n\\n"
            f"子问题：\\n{sub_question}\\n\\n"
            f"{visual_ref}"
            f"请生成 {n_candidates} 条候选推理路径。\\n"
            "要求：每条候选都使用中文，内容简洁且与图像描述一致；"
            "候选之间应体现不同推理角度。\\n"
            '只返回 JSON 数组，元素格式为 {"thought": "...", "answer": "..."}；不要解释。'
        )
"""
new2 = """        visual_ref = (
            f"Direct answer from the vision model: {visual_sub_answer}\\n\\n"
            if visual_sub_answer
            else ""
        )
        prompt = (
            f"Image description:\\n{visual_description}\\n\\n"
            f"Sub-question:\\n{sub_question}\\n\\n"
            f"{visual_ref}"
            f"Generate {n_candidates} candidate reasoning paths.\\n"
            "Requirements: each candidate must be in English, concise, and consistent with the image description; "
            "candidates should reflect different reasoning angles.\\n"
            'Return ONLY a JSON array with elements in the format {"thought": "...", "answer": "..."}; no explanation.'
        )
"""
assert s.count(old2) == 1, f"anchor2={s.count(old2)}"
s = s.replace(old2, new2)

# 3. 评分模块
old3 = """        cand_text = "\\n".join(
            [
                f"候选{i+1}：\\n思路：{c.get('thought','')}\\n答案：{c.get('answer','')}\\n"
                for i, c in enumerate(candidates)
            ]
        )
        visual_ref = (
            f"视觉模型直接答案（重要参考）：{visual_sub_answer}\\n\\n"
            if visual_sub_answer
            else ""
        )
        prompt = (
            f"图像描述：\\n{visual_description}\\n\\n"
            f"子问题：\\n{sub_question}\\n\\n"
            f"{visual_ref}"
            "请按正确性与一致性，为每个候选打分（0.1 到 1.0）。\\n"
            "只返回与候选顺序一致的 JSON 浮点数数组，不要解释。\\n\\n"
            f"{cand_text}"
        )
"""
new3 = """        cand_text = "\\n".join(
            [
                f"Candidate {i+1}:\\nThought: {c.get('thought','')}\\nAnswer: {c.get('answer','')}\\n"
                for i, c in enumerate(candidates)
            ]
        )
        visual_ref = (
            f"Direct answer from the vision model (important reference): {visual_sub_answer}\\n\\n"
            if visual_sub_answer
            else ""
        )
        prompt = (
            f"Image description:\\n{visual_description}\\n\\n"
            f"Sub-question:\\n{sub_question}\\n\\n"
            f"{visual_ref}"
            "Score each candidate from 0.1 to 1.0 based on correctness and consistency with the image description.\\n"
            "Return ONLY a JSON array of floats in the same order as the candidates, no explanation.\\n\\n"
            f"{cand_text}"
        )
"""
assert s.count(old3) == 1, f"anchor3={s.count(old3)}"
s = s.replace(old3, new3)

# 4. 评分重试 prompt
old4 = """        raw2 = self.llm_caller(
            "你只需要输出一个 JSON 数组，包含 %d 个 0.1 到 1.0 之间的数字，"
            "对应每个候选的分数。不要输出任何其他内容。\\n\\n%s"
            % (len(candidates), cand_text),
"""
new4 = """        raw2 = self.llm_caller(
            "You only need to output a JSON array of %d numbers between 0.1 and 1.0, "
            "one score per candidate. Do not output anything else.\\n\\n%s"
            % (len(candidates), cand_text),
"""
assert s.count(old4) == 1, f"anchor4={s.count(old4)}"
s = s.replace(old4, new4)

open(p1, "w", encoding="utf-8").write(s)
print("PATCHED reasoning_modules.py (v11 english prompts)", flush=True)
