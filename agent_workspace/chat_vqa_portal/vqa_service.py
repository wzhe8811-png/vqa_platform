"""四模块 VQA 推理（与 chat_vqa_webapp 相同逻辑）。"""

import json
import re
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/vision_qa_platform")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reasoning_modules import QuestionDecomposer, SearchController, StateEvaluator, ThoughtGenerator
from tot_got_integration import VisionToTextReasoner
from vqa_eval_utils import constrained_finalize_answer

VISION_MODEL_PATH = str(PROJECT_ROOT / "llava-1.5-7b-hf")
LLM_MODEL_PATH = str(PROJECT_ROOT / "text_model" / "qwen2_1_5b_instruct")

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()
_TRANSLATION_CACHE = {}
_TRANSLATION_LOCK = threading.Lock()


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        reasoner = VisionToTextReasoner(
            vision_model_path=VISION_MODEL_PATH,
            llm_model_path=LLM_MODEL_PATH,
        )
        llm_caller = reasoner.get_llm_caller()
        decomposer = QuestionDecomposer(llm_caller)
        generator = ThoughtGenerator(llm_caller)
        evaluator = StateEvaluator(llm_caller)
        controller = SearchController(decomposer, generator, evaluator, llm_caller)

        _PIPELINE = {
            "reasoner": reasoner,
            "llm_caller": llm_caller,
            "controller": controller,
        }
    return _PIPELINE


def build_trace_graph(question: str, trace, prediction: str):
    nodes = []
    links = []
    qid = "q-main"
    nodes.append(
        {
            "id": qid,
            "label": "问题",
            "type": "question",
            "detail": {"问题": question},
        }
    )

    for i, t in enumerate(trace or []):
        sq_id = f"sq-{i}"
        sq = t.get("sub_question", "")
        nodes.append(
            {
                "id": sq_id,
                "label": f"子问题{i+1}",
                "type": "subq",
                "detail": {"子问题": sq},
            }
        )
        links.append({"source": qid, "target": sq_id})

        cands = t.get("candidates", [])
        scores = t.get("scores", [])
        selected_idx = int(t.get("selected_index", 0)) if cands else -1

        for j, c in enumerate(cands):
            cid = f"cand-{i}-{j}"
            score = float(scores[j]) if j < len(scores) else 0.0
            selected = j == selected_idx
            nodes.append(
                {
                    "id": cid,
                    "label": f"候选{j+1}",
                    "type": "candidate_selected" if selected else "candidate",
                    "detail": {
                        "思路": c.get("thought", ""),
                        "候选答案": c.get("answer", ""),
                        "评分": score,
                        "是否选中": selected,
                    },
                }
            )
            links.append({"source": sq_id, "target": cid})

    pred_id = "pred-main"
    nodes.append(
        {
            "id": pred_id,
            "label": "最终答案",
            "type": "pred",
            "detail": {"最终答案": prediction},
        }
    )
    links.append({"source": qid, "target": pred_id})
    return {"nodes": nodes, "links": links}


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def should_translate_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if contains_chinese(t):
        return False
    return bool(re.search(r"[A-Za-z]", t))


def _safe_json_array(text: str):
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    s = text.find("[")
    e = text.rfind("]")
    if s != -1 and e != -1 and e > s:
        try:
            arr = json.loads(text[s : e + 1])
            if isinstance(arr, list):
                return arr
        except Exception:
            return None
    return None


def translate_texts_batch(llm_caller, texts):
    if not texts:
        return []

    unresolved = []
    indices = []
    output = [""] * len(texts)

    with _TRANSLATION_LOCK:
        for i, t in enumerate(texts):
            key = (t or "").strip()
            if key in _TRANSLATION_CACHE:
                output[i] = _TRANSLATION_CACHE[key]
            else:
                unresolved.append(key)
                indices.append(i)

    if unresolved:
        clipped = [u[:280] for u in unresolved]
        payload = json.dumps(clipped, ensure_ascii=False)
        prompt = (
            "请将下面 JSON 数组中的每个英文文本翻译成自然、简洁的中文。\n"
            "要求：\n"
            "1) 输出必须是 JSON 数组，长度与输入一致。\n"
            "2) 不要解释，不要添加数组外文本。\n"
            "3) 保留专有名词（如 yes/no 可翻译为 是/否）。\n\n"
            f"输入：{payload}"
        )
        translated_raw = llm_caller(prompt)
        translated = _safe_json_array(translated_raw) or []

        if len(translated) != len(unresolved):
            translated = unresolved

        with _TRANSLATION_LOCK:
            for src, zh in zip(unresolved, translated):
                zh_text = (str(zh).strip() or src).strip()
                _TRANSLATION_CACHE[src] = zh_text
            for idx, src in zip(indices, unresolved):
                output[idx] = _TRANSLATION_CACHE.get(src, src)

    for i, src in enumerate(texts):
        out = output[i]
        if should_translate_text(src) and (not contains_chinese(out)):
            prompt = (
                "请把下面文本翻译成中文。\n"
                "要求：只输出中文译文，不要英文，不要解释。\n\n"
                f"原文：{src[:320]}"
            )
            try:
                zh2 = (llm_caller(prompt) or "").strip()
                if zh2 and contains_chinese(zh2):
                    output[i] = zh2
                    with _TRANSLATION_LOCK:
                        _TRANSLATION_CACHE[src] = zh2
            except Exception:
                pass
    return output


def to_chinese_text(llm_caller, text: str) -> str:
    t = (text or "").strip()
    if not should_translate_text(t):
        return t
    return translate_texts_batch(llm_caller, [t])[0]


def localize_short_answer(answer: str) -> str:
    a = (answer or "").strip().lower()
    mapping = {
        "yes": "是",
        "no": "否",
        "left": "左侧",
        "right": "右侧",
        "top": "上方",
        "bottom": "下方",
        "true": "是",
        "false": "否",
    }
    return mapping.get(a, "")


def to_chinese_answer(llm_caller, answer: str, question: str) -> str:
    mapped = localize_short_answer(answer)
    if mapped:
        return mapped
    prompt = (
        "请将下面的视觉问答短答案翻译为中文，只输出中文答案本身，不要解释，不要添加标点。\n\n"
        f"问题：{question}\n"
        f"答案：{answer}\n"
    )
    try:
        zh = (llm_caller(prompt) or "").strip()
        return zh if zh else answer
    except Exception:
        return answer


def localize_trace_to_chinese(llm_caller, trace):
    localized = []
    texts = []
    slots = []

    for i, t in enumerate(trace or []):
        nt = {
            "sub_question": t.get("sub_question", ""),
            "candidates": [],
            "scores": list(t.get("scores", [])),
            "selected_index": t.get("selected_index", 0),
        }
        if should_translate_text(nt["sub_question"]):
            slots.append(("sub_question", i, None, None))
            texts.append(nt["sub_question"])
        for j, c in enumerate(t.get("candidates", [])):
            thought = c.get("thought", "")
            answer = c.get("answer", "")
            nc = {"thought": thought, "answer": answer}
            if should_translate_text(thought):
                slots.append(("thought", i, j, None))
                texts.append(thought)
            if should_translate_text(answer):
                slots.append(("answer", i, j, None))
                texts.append(answer)
            nt["candidates"].append(nc)
        localized.append(nt)

    translated = translate_texts_batch(llm_caller, texts)
    for k, slot in enumerate(slots):
        field, i, j, _ = slot
        if field == "sub_question":
            localized[i]["sub_question"] = translated[k]
        elif field == "thought":
            localized[i]["candidates"][j]["thought"] = translated[k]
        elif field == "answer":
            localized[i]["candidates"][j]["answer"] = translated[k]
    return localized


def run_vqa(image_path: str, question: str) -> dict:
    pipeline = get_pipeline()
    reasoner = pipeline["reasoner"]
    llm_caller = pipeline["llm_caller"]
    controller = pipeline["controller"]

    visual_desc = reasoner.get_visual_description(image_path, question)
    solved = controller.solve(
        visual_description=visual_desc,
        question=question,
        n_subquestions=3,
        n_candidates=3,
    )
    answer = constrained_finalize_answer(
        llm_caller,
        visual_desc,
        question,
        solved.get("final_answer", ""),
    )
    question_zh = to_chinese_text(llm_caller, question)
    answer_zh = to_chinese_answer(llm_caller, answer, question)
    trace_zh = localize_trace_to_chinese(llm_caller, solved.get("trace", []))
    visual_desc_zh = to_chinese_text(llm_caller, visual_desc)
    graph = build_trace_graph(question_zh, trace_zh, answer_zh)

    return {
        "answer": answer_zh,
        "raw_answer": answer,
        "visual_description": visual_desc_zh,
        "trace": trace_zh,
        "graph": graph,
        "question_zh": question_zh,
    }
