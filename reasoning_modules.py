import json
import re
from typing import Any, Dict, List

from vqa_eval_utils import answers_equivalent


def _safe_json_extract(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

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


def _clean_model_text(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json|text)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip().strip('"').strip("'").strip()
    return s


def _clean_sub_question_text(text: str) -> str:
    s = _clean_model_text(text)
    if s.startswith("[[") and s.endswith("]]"):
        s = s[2:-2].strip()
    s = s.strip("[]{}()").strip()
    s = s.replace('\\"', '"')
    s = s.replace('"', "")
    s = re.sub(r"\s*,\s*", "：", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class QuestionDecomposer:
    """模块1：将主问题拆解为子问题。"""

    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

    def decompose(self, visual_description: str, question: str, n_subquestions: int = 3) -> List[str]:
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"主问题：\n{question}\n\n"
            f"请将主问题拆解为 {n_subquestions} 个简短、可验证的子问题。\n"
            "要求：子问题使用中文。\n"
            "只返回 JSON 字符串数组，不要解释，不要输出数组外文本。\n"
            '示例：["子问题1", "子问题2", "子问题3"]'
        )
        raw = self.llm_caller(prompt)
        parsed = _safe_json_extract(raw)
        if isinstance(parsed, dict):
            # 常见结构: {"sub_questions": [...]} / {"questions":[...]}
            maybe = parsed.get("sub_questions", parsed.get("questions", None))
            if isinstance(maybe, list):
                parsed = maybe
        if isinstance(parsed, list):
            out = []
            for x in parsed:
                if isinstance(x, str):
                    s = _clean_sub_question_text(x)
                elif isinstance(x, list) and x:
                    s = _clean_sub_question_text(str(x[0]))
                elif isinstance(x, dict):
                    s = _clean_sub_question_text(str(x.get("question", x.get("sub_question", ""))))
                else:
                    s = _clean_sub_question_text(str(x))
                if s and s not in {"[]", "[ ]"}:
                    out.append(s)
            return out[:n_subquestions] if out else [question]

        # fallback: line split
        lines = [
            _clean_sub_question_text(ln.strip("- ").strip())
            for ln in (raw or "").splitlines()
            if ln.strip() and ln.strip() not in {"[]", "[ ]"}
        ]
        if not lines:
            return [question]
        return lines[:n_subquestions]


class ThoughtGenerator:
    """模块2：对每个子问题生成多条候选思路与候选答案。"""

    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

    def generate_candidates(
        self,
        visual_description: str,
        sub_question: str,
        n_candidates: int = 3,
    ) -> List[Dict[str, str]]:
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"子问题：\n{sub_question}\n\n"
            f"请生成 {n_candidates} 条候选推理路径。\n"
            "要求：每条候选都使用中文，内容简洁且与图像描述一致。\n"
            '只返回 JSON 数组，元素格式为 {"thought": "...", "answer": "..."}；不要解释。'
        )
        raw = self.llm_caller(prompt)
        parsed = _safe_json_extract(raw)
        if isinstance(parsed, dict):
            thought = str(parsed.get("thought", "")).strip()
            answer = str(parsed.get("answer", "")).strip()
            if thought or answer:
                return [{"thought": thought, "answer": answer}]
        if isinstance(parsed, list) and parsed:
            out = []
            for x in parsed:
                if isinstance(x, dict):
                    thought = _clean_model_text(str(x.get("thought", "")))
                    answer = _clean_model_text(str(x.get("answer", "")))
                    if thought or answer:
                        out.append({"thought": thought, "answer": answer})
            if out:
                return out[:n_candidates]

        # fallback: single candidate
        txt = _clean_model_text(raw)
        return [{"thought": txt, "answer": txt}]


class StateEvaluator:
    """模块3：为候选路径打分，支持评分与排序。"""

    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

    def score_candidates(
        self,
        visual_description: str,
        sub_question: str,
        candidates: List[Dict[str, str]],
    ) -> List[float]:
        if not candidates:
            return []

        cand_text = "\n".join(
            [
                f"候选{i+1}：\n思路：{c.get('thought','')}\n答案：{c.get('answer','')}\n"
                for i, c in enumerate(candidates)
            ]
        )
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"子问题：\n{sub_question}\n\n"
            "请按正确性与一致性，为每个候选打分（0.1 到 1.0）。\n"
            "只返回与候选顺序一致的 JSON 浮点数数组，不要解释。\n\n"
            f"{cand_text}"
        )
        raw = self.llm_caller(prompt)
        parsed = _safe_json_extract(raw)
        if isinstance(parsed, list) and len(parsed) == len(candidates):
            try:
                return [float(x) for x in parsed]
            except Exception:
                pass

        # fallback: regex float parse
        nums = re.findall(r"(0\.\d+|1\.0|[0-9]\.\d+)", raw or "")
        scores = []
        for n in nums[: len(candidates)]:
            try:
                scores.append(float(n))
            except Exception:
                continue
        if len(scores) == len(candidates):
            return scores
        return [0.5 for _ in candidates]


class SearchController:
    """模块4：主控搜索流程（分解 -> 生成 -> 评估 -> 选择 -> 聚合）。"""

    def __init__(
        self,
        decomposer: QuestionDecomposer,
        generator: ThoughtGenerator,
        evaluator: StateEvaluator,
        llm_caller,
    ):
        self.decomposer = decomposer
        self.generator = generator
        self.evaluator = evaluator
        self.llm_caller = llm_caller

    def solve(
        self,
        visual_description: str,
        question: str,
        n_subquestions: int = 3,
        n_candidates: int = 3,
    ) -> Dict[str, Any]:
        sub_questions = self.decomposer.decompose(
            visual_description=visual_description,
            question=question,
            n_subquestions=n_subquestions,
        )

        trace = []
        selected_sub_answers = []
        for sub_q in sub_questions:
            candidates = self.generator.generate_candidates(
                visual_description=visual_description,
                sub_question=sub_q,
                n_candidates=n_candidates,
            )
            scores = self.evaluator.score_candidates(
                visual_description=visual_description,
                sub_question=sub_q,
                candidates=candidates,
            )

            best_idx = 0
            if scores:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best = candidates[best_idx]
            selected_sub_answers.append(
                {
                    "sub_question": sub_q,
                    "selected_answer": best.get("answer", ""),
                    "selected_score": scores[best_idx] if scores else 0.5,
                }
            )
            trace.append(
                {
                    "sub_question": sub_q,
                    "candidates": candidates,
                    "scores": scores,
                    "selected_index": best_idx,
                }
            )

        subqa_text = "\n".join(
            [
                f"- {x['sub_question']}: {x['selected_answer']}"
                for x in selected_sub_answers
            ]
        )
        final_prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"原始问题：\n{question}\n\n"
            "请基于已选择的子问题答案，给出最终答案。\n"
            "要求：只输出最终答案文本，不要解释。\n\n"
            f"已选子答案：\n{subqa_text}"
        )
        final_answer = self.llm_caller(final_prompt).strip()

        return {
            "final_answer": final_answer,
            "sub_questions": sub_questions,
            "selected_sub_answers": selected_sub_answers,
            "trace": trace,
        }


def exact_match(pred: str, gt: str) -> bool:
    return answers_equivalent(pred, gt)
