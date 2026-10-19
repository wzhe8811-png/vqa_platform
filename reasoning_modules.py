import json
import random
import re
from typing import Any, Dict, List, Optional

from vqa_eval_utils import answers_equivalent, canonicalize_answer


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


def _is_valid_sub_question(s: str) -> bool:
    """v3: 过滤「绿色」「金属」这类关键词伪子问题。

    合法子问题必须是完整问句：长度足够且（含疑问词 或 以问号结尾）。
    """
    if not s or len(s) < 4:
        return False
    s_low = s.lower()
    zh_q = any(w in s for w in ("吗", "么", "什么", "哪", "谁", "多少", "是否", "如何", "怎样", "几"))
    en_q = s.endswith("?") or any(
        w in s_low for w in ("what", "which", "where", "who", "how many", "how much", "is ", "are ", "does ", "do ", "can ", "have ")
    )
    # 拒绝纯关键词（无问句特征且无谓语动词特征、非常短）
    if not zh_q and not en_q:
        return False
    return True


class QuestionDecomposer:
    """模块1：将主问题拆解为子问题（v3: 子问题质量过滤）。"""

    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

    def decompose(self, visual_description: str, question: str, n_subquestions: int = 3) -> List[str]:
        prompt = (
            f"Image description:\n{visual_description}\n\n"
            f"Question:\n{question}\n\n"
            f"Decompose the question into {n_subquestions} short, verifiable sub-questions.\n"
            "Requirements:\n"
            "1. Each sub-question must be a complete question in English (ending with '?'), not a single noun or keyword;\n"
            "2. Sub-questions must be written in English;\n"
            "3. Return ONLY a JSON array of strings, no explanation.\n"
            'Example: ["What is the object on the left?", "What color is it?"]'
        )
        raw = self.llm_caller(prompt, max_new_tokens=192)
        parsed = _safe_json_extract(raw)
        if isinstance(parsed, dict):
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
                if s and s not in {"[]", "[ ]"} and _is_valid_sub_question(s):
                    out.append(s)
            if out:
                return out[:n_subquestions]

        # fallback: line split + 质量过滤
        lines = [
            _clean_sub_question_text(ln.strip("- ").strip())
            for ln in (raw or "").splitlines()
            if ln.strip() and ln.strip() not in {"[]", "[ ]"}
        ]
        lines = [ln for ln in lines if _is_valid_sub_question(ln)]
        if lines:
            return lines[:n_subquestions]

        # v3: 全部无效时回退为原问题本身（不再用关键词污染）
        return [question]


class ThoughtGenerator:
    """模块2：对每个子问题生成多条候选思路与候选答案（v3: 垃圾候选过滤 + 视觉答案兜底）。"""

    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

    def generate_candidates(
        self,
        visual_description: str,
        sub_question: str,
        n_candidates: int = 3,
        visual_sub_answer: str = "",
    ) -> List[Dict[str, str]]:
        visual_ref = (
            f"Direct answer from the vision model: {visual_sub_answer}\n\n"
            if visual_sub_answer
            else ""
        )
        prompt = (
            f"Image description:\n{visual_description}\n\n"
            f"Sub-question:\n{sub_question}\n\n"
            f"{visual_ref}"
            f"Generate {n_candidates} candidate reasoning paths.\n"
            "Requirements: each candidate must be in English, concise, and consistent with the image description; "
            "candidates should reflect different reasoning angles.\n"
            'Return ONLY a JSON array with elements in the format {"thought": "...", "answer": "..."}; no explanation.'
        )
        raw = self.llm_caller(prompt, max_new_tokens=96)
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
                    # v3: 过滤 "[]" / 纯 JSON 残骸 等垃圾候选
                    if answer in {"[]", "{}", "[ ]"}:
                        answer = ""
                    if thought in {"[]", "{}", "[ ]"}:
                        thought = ""
                    if thought or answer:
                        out.append({"thought": thought, "answer": answer})
            if out:
                return out[:n_candidates]

        # v3: 全部失败时，视觉答案作为唯一候选（保底正确信号）
        txt = _clean_model_text(raw)
        if txt in {"[]", "{}", "[ ]", ""} and visual_sub_answer:
            return [{"thought": visual_sub_answer, "answer": visual_sub_answer}]
        return [{"thought": txt, "answer": txt}]


class StateEvaluator:
    """模块3：为候选路径打分（v3: 垃圾候选处理 + 消灭 [0.5] fallback）。"""

    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

    def score_candidates(
        self,
        visual_description: str,
        sub_question: str,
        candidates: List[Dict[str, str]],
        visual_sub_answer: str = "",
    ) -> List[float]:
        if not candidates:
            return []

        cand_text = "\n".join(
            [
                f"Candidate {i+1}:\nThought: {c.get('thought','')}\nAnswer: {c.get('answer','')}\n"
                for i, c in enumerate(candidates)
            ]
        )
        visual_ref = (
            f"Direct answer from the vision model (important reference): {visual_sub_answer}\n\n"
            if visual_sub_answer
            else ""
        )
        prompt = (
            f"Image description:\n{visual_description}\n\n"
            f"Sub-question:\n{sub_question}\n\n"
            f"{visual_ref}"
            "Score each candidate from 0.1 to 1.0 based on correctness and consistency with the image description.\n"
            "Return ONLY a JSON array of floats in the same order as the candidates, no explanation.\n\n"
            f"{cand_text}"
        )

        raw = self.llm_caller(prompt, max_new_tokens=32)
        scores = self._parse_scores(raw, len(candidates))
        if scores is not None:
            return scores

        raw2 = self.llm_caller(
            "You only need to output a JSON array of %d numbers between 0.1 and 1.0, "
            "one score per candidate. Do not output anything else.\n\n%s"
            % (len(candidates), cand_text),
            do_sample=False,
            max_new_tokens=32,
        )
        scores = self._parse_scores(raw2, len(candidates))
        if scores is not None:
            return scores

        return self._heuristic_scores(candidates, visual_sub_answer)

    def _parse_scores(self, raw: str, n: int) -> Optional[List[float]]:
        parsed = _safe_json_extract(raw)
        if isinstance(parsed, list) and len(parsed) == n:
            try:
                vals = [float(x) for x in parsed]
                if all(0.0 <= v <= 1.0 for v in vals):
                    return vals
            except Exception:
                pass
        nums = re.findall(r"(0\.\d+|1\.0|[0-9]\.\d+)", raw or "")
        vals = []
        for x in nums[:n]:
            try:
                vals.append(float(x))
            except Exception:
                continue
        if len(vals) == n:
            return vals
        return None

    def _heuristic_scores(
        self, candidates: List[Dict[str, str]], visual_sub_answer: str
    ) -> List[float]:
        vs = canonicalize_answer(visual_sub_answer)
        vs_tokens = set(vs.split()) if vs else set()
        scores = []
        for c in candidates:
            ans = canonicalize_answer(c.get("answer", ""))
            ans_tokens = set(ans.split()) if ans else set()
            if not ans:
                scores.append(0.2)
            elif vs and ans == vs:
                scores.append(1.0)
            elif vs_tokens and (vs_tokens.issubset(ans_tokens) or ans_tokens.issubset(vs_tokens)):
                scores.append(0.7)
            else:
                scores.append(0.4 if len(ans_tokens) <= 6 else 0.2)
        return scores


class SearchController:
    """模块4：主控搜索流程（v3: 视觉答案贯穿分解/生成/评分/聚合）。"""

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
        visual_answer_fn=None,
        ablation: str = "full",
    ) -> Dict[str, Any]:
        # ablation: full | no_visual | no_decompose | no_generate | no_evaluate | no_aggregate
        if ablation == "no_visual":
            visual_answer_fn = None
        if ablation == "no_decompose":
            # 消融: 跳过分解模块，直接对原问题生成候选
            sub_questions = [question]
        else:
            sub_questions = self.decomposer.decompose(
                visual_description=visual_description,
                question=question,
                n_subquestions=n_subquestions,
            )

        # v3: 视觉模型对主问题的直接答案，作为最终聚合的强参考
        main_visual_answer = ""
        if visual_answer_fn is not None:
            try:
                main_visual_answer = visual_answer_fn(question) or ""
            except Exception as e:
                print(f"  [warn] main visual answer failed: {e}")

        trace = []
        selected_sub_answers = []
        for sub_q in sub_questions:
            visual_sub_answer = ""
            if visual_answer_fn is not None and sub_q != question:
                try:
                    visual_sub_answer = visual_answer_fn(sub_q) or ""
                except Exception as e:
                    print(f"  [warn] visual answer failed for sub_q: {e}")
            elif sub_q == question:
                visual_sub_answer = main_visual_answer

            if ablation == "no_generate":
                # 消融: 跳过候选生成模块，视觉答案即唯一候选
                if visual_sub_answer:
                    candidates = [{"thought": "", "answer": visual_sub_answer}]
                else:
                    direct = self.llm_caller(
                        f"Question: {sub_q}\nAnswer with a single word or short phrase in English.",
                        max_new_tokens=64,
                    ).strip()
                    candidates = [{"thought": "", "answer": direct}]
            else:
                candidates = self.generator.generate_candidates(
                    visual_description=visual_description,
                    sub_question=sub_q,
                    n_candidates=n_candidates,
                    visual_sub_answer=visual_sub_answer,
                )
            if ablation == "no_evaluate":
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
            selected_sub_answers.append(
                {
                    "sub_question": sub_q,
                    "visual_answer": visual_sub_answer,
                    "selected_answer": best.get("answer", ""),
                    "selected_score": scores[best_idx] if scores else 0.5,
                }
            )
            trace.append(
                {
                    "sub_question": sub_q,
                    "visual_answer": visual_sub_answer,
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
        # v3: 视觉模型的主问题答案作为强参考进入聚合 prompt
        main_ref = (
            f"Direct answer from the vision model to the original question (highly reliable, prioritize it): {main_visual_answer}\n\n"
            if main_visual_answer
            else ""
        )
        final_prompt = (
            f"Image Description:\n{visual_description}\n\n"
            f"Original Question:\n{question}\n\n"
            f"{main_ref}"
            "Based on the direct visual answer and the selected sub-question answers, give the final answer.\n"
            "Output ONLY the final answer text, no explanation.\n\n"
            f"Selected sub-answers:\n{subqa_text}"
        )
        final_answer = self.llm_caller(final_prompt, max_new_tokens=96).strip()
        if ablation == "no_aggregate":
            # 消融: 跳过聚合模块，取评分最高的子问题答案作为最终答案
            best_sub = max(selected_sub_answers, key=lambda x: x.get("selected_score", 0))
            final_answer = best_sub["selected_answer"]

        return {
            "final_answer": final_answer,
            "main_visual_answer": main_visual_answer,
            "sub_questions": sub_questions,
            "selected_sub_answers": selected_sub_answers,
            "trace": trace,
        }


def exact_match(pred: str, gt: str) -> bool:
    return answers_equivalent(pred, gt)
