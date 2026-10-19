# 基于推理时拓展的视觉问答系统 - Web 可视化服务
# 功能: 单题交互推理(5方法) / 思维树可视化 / 多方法对比 / 实验数据浏览 / 系统说明
import io
import json
import os
import re
import sys
import time
import uuid

BASE = "/root/autodl-tmp/vision_qa_platform"
sys.path.insert(0, BASE)

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=None)
UPLOAD_DIR = os.path.join(BASE, "web_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DATA_SAMPLE = os.path.join(BASE, "data_sample")

# GQA 高频答案中英对照词典（命中则免模型调用）
ANS_ZH = {
    "yes": "是", "no": "否", "red": "红色", "blue": "蓝色", "green": "绿色",
    "yellow": "黄色", "orange": "橙色", "purple": "紫色", "pink": "粉色",
    "black": "黑色", "white": "白色", "gray": "灰色", "grey": "灰色",
    "brown": "棕色", "light blue": "浅蓝色", "dark blue": "深蓝色",
    "light green": "浅绿色", "dark green": "深绿色", "light brown": "浅棕色",
    "dark brown": "深棕色", "left": "左边", "right": "右边", "front": "前面",
    "back": "后面", "middle": "中间", "top": "顶部", "bottom": "底部",
    "man": "男人", "woman": "女人", "boy": "男孩", "girl": "女孩",
    "cat": "猫", "dog": "狗", "car": "汽车", "bus": "公交车", "train": "火车",
    "bike": "自行车", "table": "桌子", "chair": "椅子", "door": "门",
    "window": "窗户", "shirt": "衬衫", "hat": "帽子", "helmet": "头盔",
    "glasses": "眼镜", "phone": "手机", "book": "书", "banana": "香蕉",
    "apple": "苹果", "pizza": "披萨", "bread": "面包", "cake": "蛋糕",
    "water": "水", "milk": "牛奶", "coffee": "咖啡", "tea": "茶", "tree": "树",
    "grass": "草地", "sky": "天空", "wall": "墙壁", "floor": "地板",
    "bed": "床", "shoes": "鞋子", "pants": "裤子", "jacket": "夹克", "tie": "领带",
    "watch": "手表", "computer": "电脑", "laptop": "笔记本电脑", "tv": "电视",
    "television": "电视", "cup": "杯子", "cupboard": "橱柜", "cabinet": "橱柜",
    "plate": "盘子", "bowl": "碗", "fork": "叉子", "spoon": "勺子",
    "knife": "刀", "bottle": "瓶子", "bag": "包", "backpack": "背包",
    "umbrella": "雨伞", "ball": "球", "clock": "钟表", "mirror": "镜子",
    "picture": "画", "lamp": "台灯", "refrigerator": "冰箱", "fridge": "冰箱",
    "sink": "水槽", "toilet": "马桶", "shower": "淋浴", "curtain": "窗帘",
    "couch": "沙发", "sofa": "沙发", "desk": "书桌", "stairs": "楼梯",
    "fence": "栅栏", "gate": "大门", "street": "街道", "road": "马路",
    "building": "建筑", "house": "房子", "kitchen": "厨房",
    "bathroom": "浴室", "bedroom": "卧室", "food": "食物", "animal": "动物",
}


def _trans_en2zh(text, m):
    """英文答案 → 中文（词典优先，未命中用 Qwen 翻译）"""
    t = (text or "").strip().lower().rstrip(".")
    if t in ANS_ZH:
        return ANS_ZH[t]
    try:
        prompt = ("Translate the following English answer into Chinese. "
                  "Return ONLY the Chinese translation, no explanation.\n\n"
                  f"Answer: {t}\nChinese:")
        out = m.get_llm_caller()(prompt, do_sample=False, max_new_tokens=64)
        zh = out.strip().split("\n")[0].strip()
        return zh if zh else t
    except Exception:
        return t

_model = None
_controller = None


def get_model():
    global _model
    if _model is None:
        from tot_got_integration import VisionToTextReasoner
        _model = VisionToTextReasoner(
            vision_model_path=os.path.join(BASE, "llava-1.5-7b-hf"),
            llm_model_path=os.path.join(BASE, "text_model", "qwen2_7b_instruct"),
        )
    return _model


def get_controller():
    global _controller
    if _controller is None:
        from reasoning_modules import (
            QuestionDecomposer, ThoughtGenerator, StateEvaluator, SearchController,
        )
        m = get_model()
        llm = m.get_llm_caller()
        _controller = SearchController(
            QuestionDecomposer(llm), ThoughtGenerator(llm), StateEvaluator(llm), llm
        )
    return _controller


def _safe_json(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for a, b in (("[", "]"), ("{", "}")):
        s, e = text.find(a), text.rfind(b)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
    return None


# ---------------- 单题推理 ----------------

def infer_standard(m, image_path, question):
    desc = m.get_visual_description(image_path, question)
    visual_ans = m.get_visual_answer(image_path, question)
    prompt = (f"Visual Description:\n{desc}\n\nQuestion:\n{question}\n"
              "Answer with a single word or short phrase in English.")
    raw = m.get_llm_caller()(prompt, do_sample=False, max_new_tokens=64)
    from vqa_eval_utils import extract_final_answer
    return {"final_answer": extract_final_answer(raw) or raw.strip(), "raw": raw,
            "visual_description": desc, "visual_answer": visual_ans, "tree": None}


def infer_cot(m, image_path, question):
    desc = m.get_visual_description(image_path, question)
    visual_ans = m.get_visual_answer(image_path, question)
    prompt = (f"Visual Description:\n{desc}\n\nQuestion:\n{question}\n"
              "Let's think step by step. At the end, state your final answer in a "
              "single word or short phrase in English after 'Final answer:'.")
    raw = m.get_llm_caller()(prompt, do_sample=False, max_new_tokens=256)
    from vqa_eval_utils import extract_final_answer
    chain = raw.split("Final answer:")[0].strip() if "Final answer:" in raw else raw
    return {"final_answer": extract_final_answer(raw) or raw.strip(), "raw": raw,
            "chain": chain, "visual_description": desc, "visual_answer": visual_ans,
            "tree": None}


def infer_tot_got(m, image_path, question, task):
    desc = m.get_visual_description(image_path, question)
    visual_ans = m.get_visual_answer(image_path, question)
    llm = m.get_llm_caller()
    num_branches = 3
    initial_context = (f"Visual Description:\n{desc}\n\nQuestion:\n{question}\n")
    propose_prompt = (initial_context
        + "You are solving a Visual Question Answering (VQA) problem.\n"
        + f"Generate {num_branches} different reasoning paths.\n"
        + "For each path, include a short reasoning and a candidate final answer.\n"
        + 'Output ONLY a JSON array of objects with exactly these keys: "thought" (string) and "answer" (string).')
    propose_text = llm(propose_prompt, max_new_tokens=256)
    proposals = _safe_json(propose_text)
    if not isinstance(proposals, list) or not proposals:
        proposals = [{"thought": propose_text.strip(), "answer": propose_text.strip()}]
        num_branches = len(proposals)

    visual_candidate = None
    if visual_ans and visual_ans.strip():
        visual_candidate = {"thought": f"The vision model directly looked at the image and answered: {visual_ans}. This direct visual observation is usually reliable.",
                            "answer": visual_ans.strip()}
        proposals = [visual_candidate] + proposals

    candidates_str = "\n".join(
        f"Candidate {i+1}:\nThought: {p.get('thought','')}\nAnswer: {p.get('answer','')}\n"
        for i, p in enumerate(proposals))
    score_prompt = (initial_context
        + "Now score each candidate path for correctness and consistency with the question and visual description.\n"
        + "Return ONLY a JSON array of floats between 0.1 and 1.0, in the same order as candidates.\n\n"
        + candidates_str)
    score_text = llm(score_prompt, max_new_tokens=64)
    scores = _safe_json(score_text)
    if not isinstance(scores, list) or len(scores) != len(proposals):
        floats = re.findall(r"(0\.\d+|1\.0|[0-9]\.\d+)", score_text)
        scores = [float(f) for f in floats[:len(proposals)]]
    if len(scores) != len(proposals):
        scores = ([1.0] + [0.5] * (len(proposals) - 1)) if visual_candidate else [0.5] * len(proposals)

    ranked = sorted(range(len(proposals)), key=lambda i: scores[i], reverse=True)
    best_idx = ranked[0]
    best = proposals[best_idx]

    visual_hint = ""
    if visual_ans and visual_ans.strip():
        visual_hint = (f"\nNote: the vision model directly looked at the image and answered: {visual_ans}. Treat this direct answer as a strong reference.\n")
    extraction_prompt = (initial_context
        + "Use the best reasoning path below to provide a FINAL answer.\n" + visual_hint
        + "Return ONLY the final answer text (no explanation).\n\n"
        + f"Best thought:\n{best.get('thought','')}\n\nBest candidate answer:\n{best.get('answer','')}\n")
    final_raw = llm(extraction_prompt, max_new_tokens=64)
    from vqa_eval_utils import extract_final_answer
    final = extract_final_answer(final_raw) or final_raw.strip()

    tree = {
        "name": question, "children": [{
            "name": p.get("answer", "")[:80], "thought": p.get("thought", "")[:300],
            "score": round(float(scores[i]), 2), "visual": i == 0 and visual_candidate is not None,
            "selected": i == best_idx,
        } for i, p in enumerate(proposals)]
    }
    tree["children"].append({"name": "Final: " + final[:80], "final": True})
    return {"final_answer": final, "raw": final_raw, "visual_description": desc,
            "visual_answer": visual_ans, "tree": tree, "candidates": proposals,
            "scores": [round(float(s), 2) for s in scores], "best_idx": best_idx}


def infer_modular(m, image_path, question):
    from vqa_eval_utils import constrained_finalize_answer
    desc = m.get_visual_description(image_path, question)
    visual_ans = m.get_visual_answer(image_path, question)
    ctrl = get_controller()
    solved = ctrl.solve(
        ablation="full", visual_description=desc, question=question,
        n_subquestions=3, n_candidates=3,
        visual_answer_fn=lambda sq: m.get_visual_answer(image_path, sq),
    )
    final = constrained_finalize_answer(m.get_llm_caller(), desc, question, solved["final_answer"])
    tree = {"name": question, "children": []}
    for sub in solved["trace"]:
        node = {"name": sub.get("sub_question", "")[:80], "visual_answer": sub.get("visual_answer", ""),
                "children": []}
        cands = sub.get("candidates", [])
        scores = sub.get("scores", [])
        sel = sub.get("selected_index", 0)
        for i, c in enumerate(cands):
            node["children"].append({
                "name": c.get("answer", "")[:80], "thought": c.get("thought", "")[:300],
                "score": round(float(scores[i]), 2) if i < len(scores) else 0.5,
                "visual": bool(sub.get("visual_answer")) and c.get("answer", "").strip() == sub.get("visual_answer", "").strip(),
                "selected": i == sel,
            })
        tree["children"].append(node)
    tree["children"].append({"name": "Final: " + final[:80], "final": True})
    return {"final_answer": final, "raw": solved.get("final_answer", ""),
            "visual_description": desc, "visual_answer": visual_ans, "tree": tree,
            "sub_questions": [t["sub_question"] for t in solved["trace"]]}


METHODS = {"standard": infer_standard, "cot": infer_cot, "tot": lambda m, p, q: infer_tot_got(m, p, q, "tot"),
           "got": lambda m, p, q: infer_tot_got(m, p, q, "got"), "modular": infer_modular}


@app.route("/api/infer", methods=["POST"])
def api_infer():
    if "image" not in request.files:
        return jsonify({"error": "缺少图片文件"}), 400
    f = request.files["image"]
    question = (request.form.get("question") or "").strip()
    method = request.form.get("method") or "modular"
    gt = (request.form.get("ground_truth") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    if method not in METHODS:
        return jsonify({"error": f"未知方法: {method}"}), 400

    ext = os.path.splitext(f.filename or "")[1] or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    f.save(fpath)

    try:
        m = get_model()
        t0 = time.time()
        res = METHODS[method](m, fpath, question)
        res["time_cost"] = round(time.time() - t0, 1)
        res["image_url"] = f"/uploads/{fname}"
        res["method"] = method
        res["question"] = question
        try:
            res["chinese_answer"] = _trans_en2zh(res.get("final_answer", ""), m)
        except Exception:
            pass
        if gt:
            from vqa_eval_utils import answers_equivalent, canonicalize_answer
            try:
                res["correct"] = bool(answers_equivalent(
                    canonicalize_answer(res["final_answer"]), canonicalize_answer(gt)))
                res["ground_truth"] = gt
            except Exception:
                pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": f"推理失败: {e}"}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    if "image" not in request.files:
        return jsonify({"error": "缺少图片文件"}), 400
    f = request.files["image"]
    question = (request.form.get("question") or "").strip()
    methods = request.form.get("methods") or "standard,cot,tot,got,modular"
    methods = [x.strip() for x in methods.split(",") if x.strip() in METHODS]
    gt = (request.form.get("ground_truth") or "").strip()
    if not question or not methods:
        return jsonify({"error": "参数不完整"}), 400

    ext = os.path.splitext(f.filename or "")[1] or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    f.save(fpath)

    results = []
    try:
        m = get_model()
        for method in methods:
            t0 = time.time()
            try:
                res = METHODS[method](m, fpath, question)
                res["time_cost"] = round(time.time() - t0, 1)
                res["method"] = method
                try:
                    res["chinese_answer"] = _trans_en2zh(res.get("final_answer", ""), m)
                except Exception:
                    pass
                if gt:
                    from vqa_eval_utils import answers_equivalent, canonicalize_answer
                    try:
                        res["correct"] = bool(answers_equivalent(
                            canonicalize_answer(res["final_answer"]), canonicalize_answer(gt)))
                    except Exception:
                        pass
                results.append(res)
            except Exception as e:
                results.append({"method": method, "error": str(e)})
        return jsonify({"image_url": f"/uploads/{fname}", "question": question,
                        "ground_truth": gt, "results": results})
    except Exception as e:
        return jsonify({"error": f"推理失败: {e}"}), 500


# ---------------- 实验数据浏览 ----------------

RES_DIR = os.path.join(BASE, "results", "accuracy")

EXPERIMENT_FILES = [
    ("Standard(直答)", "standard_100.json"),
    ("CoT(链式思维)", "cot_100.json"),
    ("ToT(思维树)", "tot_100.json"),
    ("GoT(思维图)", "got_100.json"),
    ("Modular(四模块)", "modular_full_100_v11.json"),
    ("消融-缺分解", "modular_abl100_v11_nodecompose.json"),
    ("消融-缺生成", "modular_abl100_v11_nogenerate.json"),
    ("消融-缺评分", "modular_abl100_v11_noevaluate.json"),
    ("消融-缺聚合", "modular_abl100_v11_noaggregate.json"),
]
ROBUST_FILES = [
    ("鲁棒性-第1次", "modular_7b_eng_test50.json"),
    ("鲁棒性-第2次", "modular_7b_robust_run2.json"),
    ("鲁棒性-第3次", "modular_7b_robust_run3.json"),
]


def _judge(pred, gt):
    try:
        from vqa_eval_utils import answers_equivalent, canonicalize_answer, extract_final_answer
        return bool(answers_equivalent(canonicalize_answer(extract_final_answer(pred or "")),
                                       canonicalize_answer(gt or "")))
    except Exception:
        return False


def _load_result(fn):
    path = os.path.join(RES_DIR, fn)
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("results") or []
    return d


@app.route("/api/stats")
def api_stats():
    rows = []
    for label, fn in EXPERIMENT_FILES:
        d = _load_result(fn)
        if d is None:
            continue
        c = sum(1 for it in d if _judge(it.get("prediction"), it.get("ground_truth")))
        rows.append({"label": label, "file": fn, "correct": c, "total": len(d),
                     "accuracy": round(c / len(d) * 100, 1) if d else 0})
    robust = []
    for label, fn in ROBUST_FILES:
        d = _load_result(fn)
        if d is None:
            continue
        c = sum(1 for it in d if _judge(it.get("prediction"), it.get("ground_truth")))
        robust.append({"label": label, "file": fn, "correct": c, "total": len(d),
                       "accuracy": round(c / len(d) * 100, 1) if d else 0})
    return jsonify({"experiments": rows, "robustness": robust})


@app.route("/api/result/<fn>")
def api_result(fn):
    if not re.fullmatch(r"[\w\-.]+\.json", fn):
        return jsonify({"error": "非法文件名"}), 400
    d = _load_result(fn)
    if d is None:
        return jsonify({"error": "文件不存在"}), 404
    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 20)), 100)
    wrong_only = request.args.get("wrong") == "1"
    q = (request.args.get("q") or "").strip().lower()
    items = []
    for it in d:
        pred = it.get("prediction") or ""
        gt = it.get("ground_truth") or ""
        ok = _judge(pred, gt)
        if wrong_only and ok:
            continue
        if q and q not in (it.get("question") or "").lower():
            continue
        img = it.get("image") or ""
        img_name = os.path.basename(img) if img else ""
        items.append({"id": it.get("question_id"), "question": it.get("question"),
                      "gt": gt, "pred": pred, "correct": ok,
                      "img": f"/api/img/{img_name}" if img_name else None,
                      "status": it.get("status"),
                      "has_trace": bool(it.get("trace"))})
    total = len(items)
    return jsonify({"total": total, "items": items[offset:offset + limit]})


@app.route("/api/trace/<fn>/<idx>")
def api_trace(fn, idx):
    if not re.fullmatch(r"[\w\-.]+\.json", fn):
        return jsonify({"error": "非法文件名"}), 400
    d = _load_result(fn)
    if d is None:
        return jsonify({"error": "文件不存在"}), 404
    try:
        it = d[int(idx)]
    except Exception:
        return jsonify({"error": "索引越界"}), 404
    tree = None
    tr = it.get("trace")
    if tr:
        tree = {"name": it.get("question", ""), "children": []}
        for sub in tr:
            node = {"name": sub.get("sub_question", "")[:80],
                    "visual_answer": sub.get("visual_answer", ""), "children": []}
            cands = sub.get("candidates", [])
            scores = sub.get("scores", [])
            sel = sub.get("selected_index", 0)
            for i, c in enumerate(cands):
                node["children"].append({
                    "name": c.get("answer", "")[:80],
                    "thought": c.get("thought", "")[:300],
                    "score": round(float(scores[i]), 2) if i < len(scores) else 0.5,
                    "visual": bool(sub.get("visual_answer")) and c.get("answer", "").strip() == sub.get("visual_answer", "").strip(),
                    "selected": i == sel,
                })
            tree["children"].append(node)
        tree["children"].append({"name": "Final: " + str(it.get("prediction", ""))[:80], "final": True})
    img = it.get("image") or ""
    img_name = os.path.basename(img) if img else ""
    return jsonify({"item": {"id": it.get("question_id"), "question": it.get("question"),
                             "gt": it.get("ground_truth"), "pred": it.get("prediction"),
                             "visual_description": it.get("visual_description"),
                             "correct": _judge(it.get("prediction"), it.get("ground_truth")),
                             "img": f"/api/img/{img_name}" if img_name else None},
                    "tree": tree,
                    "sub_answers": it.get("selected_sub_answers")})


@app.route("/api/translate", methods=["POST"])
def api_translate():
    """中文问题 → 英文"""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "文本不能为空"}), 400
    try:
        m = get_model()
        prompt = ("Translate the following Chinese question into English. "
                  "Return ONLY the English translation, no explanation.\n\n"
                  f"Chinese: {text}\nEnglish:")
        out = m.get_llm_caller()(prompt, do_sample=False, max_new_tokens=128)
        en = out.strip().split("\n")[0].strip()
        return jsonify({"translated": en})
    except Exception as e:
        return jsonify({"error": f"翻译失败: {e}"}), 500


@app.route("/api/img/<fname>")
def api_img(fname):
    if not re.fullmatch(r"[\w\-.]+\.(jpg|jpeg|png)", fname):
        return jsonify({"error": "非法文件名"}), 400
    if not os.path.exists(os.path.join(DATA_SAMPLE, fname)):
        return jsonify({"error": "图片不存在"}), 404
    return send_from_directory(DATA_SAMPLE, fname)


@app.route("/uploads/<fname>")
def uploads(fname):
    return send_from_directory(UPLOAD_DIR, fname)


@app.after_request
def no_cache(resp):
    """静态页面与脚本禁止缓存，保证前端更新即时生效"""
    if request.path.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE, "web_static"), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(BASE, "web_static"), path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
