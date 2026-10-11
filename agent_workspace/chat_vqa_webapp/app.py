import os
import threading
import uuid
import sys
import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path("/root/autodl-tmp/vision_qa_platform")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reasoning_modules import QuestionDecomposer, SearchController, StateEvaluator, ThoughtGenerator
from tot_got_integration import VisionToTextReasoner
from vqa_eval_utils import constrained_finalize_answer


UPLOAD_DIR = PROJECT_ROOT / "agent_workspace" / "chat_vqa_webapp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    # 如果没有英文字符，通常不需要翻译
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
        # 控制单条长度，避免提示词过长
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

    # 兜底
    for i, t in enumerate(output):
        if not t:
            output[i] = texts[i]

    # 二次校验：若仍明显是英文，则单条强制翻译
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
    # 优先使用规则映射，避免额外推理开销
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

    # 先复制结构，再收集需翻译文本
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


def localize_selected_answers_to_chinese(llm_caller, selected_sub_answers):
    localized = []
    texts = []
    slots = []
    for i, s in enumerate(selected_sub_answers or []):
        ns = {
            "sub_question": s.get("sub_question", ""),
            "selected_answer": s.get("selected_answer", ""),
            "selected_score": s.get("selected_score", 0),
        }
        if should_translate_text(ns["sub_question"]):
            slots.append(("sub_question", i))
            texts.append(ns["sub_question"])
        if should_translate_text(ns["selected_answer"]):
            slots.append(("selected_answer", i))
            texts.append(ns["selected_answer"])
        localized.append(ns)

    translated = translate_texts_batch(llm_caller, texts)
    for k, slot in enumerate(slots):
        field, i = slot
        localized[i][field] = translated[k]
    return localized


def create_app():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/uploads/<path:filename>")
    def uploads(filename):
        return send_from_directory(str(UPLOAD_DIR), filename)

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True, "service": "chat_vqa_webapp"})

    @app.route("/api/ask", methods=["POST"])
    def ask():
        question = (request.form.get("question") or "").strip()
        image = request.files.get("image")

        if not question:
            return jsonify({"error": "question is required"}), 400
        if image is None or image.filename is None or image.filename.strip() == "":
            return jsonify({"error": "image is required"}), 400

        ext = Path(secure_filename(image.filename)).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return jsonify({"error": "unsupported image extension"}), 400

        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = UPLOAD_DIR / filename
        image.save(str(save_path))

        pipeline = get_pipeline()
        reasoner = pipeline["reasoner"]
        llm_caller = pipeline["llm_caller"]
        controller = pipeline["controller"]

        visual_desc = reasoner.get_visual_description(str(save_path), question)
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
        selected_zh = localize_selected_answers_to_chinese(
            llm_caller, solved.get("selected_sub_answers", [])
        )

        graph = build_trace_graph(question_zh, trace_zh, answer_zh)
        return jsonify(
            {
                "answer": answer_zh,
                "raw_answer": answer,
                "visual_description": visual_desc_zh,
                "raw_visual_description": visual_desc,
                "trace": trace_zh,
                "selected_sub_answers": selected_zh,
                "graph": graph,
                "image_url": f"/uploads/{filename}",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7861, debug=False)
