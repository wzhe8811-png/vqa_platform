import json
import os
import sys
import threading
import uuid
from functools import wraps
from pathlib import Path
from urllib.parse import quote

from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, session
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path("/root/autodl-tmp/vision_qa_platform")
PORTAL_DIR = PROJECT_ROOT / "agent_workspace" / "chat_vqa_portal"
SHARED_DIR = PROJECT_ROOT / "agent_workspace" / "shared"
for p in (str(PROJECT_ROOT), str(PORTAL_DIR), str(SHARED_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import db as portal_db  # noqa: E402
from auth_config import portal_base_url, secret_key  # noqa: E402

from reasoning_modules import QuestionDecomposer, SearchController, StateEvaluator, ThoughtGenerator
from tot_got_integration import VisionToTextReasoner
from compare_gqa_baselines import soft_match
from vqa_eval_utils import answers_equivalent, constrained_finalize_answer


DEFAULT_RESULTS_FILE = (
    "/root/autodl-tmp/vision_qa_platform/results/accuracy/modular_results_complex1000.json"
)
UPLOAD_DIR = PROJECT_ROOT / "agent_workspace" / "trace_webapp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VISION_MODEL_PATH = str(PROJECT_ROOT / "llava-1.5-7b-hf")
LLM_MODEL_PATH = str(PROJECT_ROOT / "text_model" / "qwen2_1_5b_instruct")

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


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
        controller = SearchController(
            QuestionDecomposer(llm_caller),
            ThoughtGenerator(llm_caller),
            StateEvaluator(llm_caller),
            llm_caller,
        )
        _PIPELINE = {"reasoner": reasoner, "llm_caller": llm_caller, "controller": controller}
    return _PIPELINE


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["RESULTS_FILE"] = os.environ.get("TRACE_RESULTS_FILE", DEFAULT_RESULTS_FILE)
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.secret_key = secret_key()

    portal_db.get_db()

    def _login_url():
        next_url = request.url
        return f"{portal_base_url()}/login?next={quote(next_url, safe='')}"

    def _current_user():
        user_id = session.get("user_id")
        if not user_id:
            return None
        return portal_db.find_user_by_id(user_id)

    def admin_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = _current_user()
            if not user:
                session.clear()
                if request.path.startswith("/api/"):
                    return jsonify({"error": "请先登录", "login_url": _login_url()}), 401
                return redirect(_login_url())

            if not portal_db.is_admin(user):
                if request.path.startswith("/api/"):
                    return jsonify(
                        {
                            "error": "仅管理员可访问轨迹浏览界面",
                            "redirect": f"{portal_base_url()}/chat",
                        }
                    ), 403
                return redirect(f"{portal_base_url()}/chat")

            session["username"] = user.get("username", session.get("username", "管理员"))
            session["role"] = portal_db.user_role(user)
            return view(*args, **kwargs)

        return wrapped

    def eval_exact_soft(pred: str, gt: str):
        pred = pred or ""
        gt = gt or ""
        exact = bool(answers_equivalent(pred, gt))
        soft = bool(soft_match(pred, gt))
        return exact, soft

    def load_cases():
        """直接读取原 JSON 测试结果文件，用于浏览 1000 次测试案例。"""
        path = Path(app.config["RESULTS_FILE"])
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def build_graph(case: dict):
        qid = str(case.get("question_id", "NA"))
        question = case.get("question", "")
        pred = case.get("prediction", "")
        gt = case.get("ground_truth") or case.get("answer", "")
        trace = case.get("trace", [])
        exact, soft = eval_exact_soft(pred, gt)

        nodes = []
        links = []

        q_node_id = f"q-{qid}"
        nodes.append(
            {
                "id": q_node_id,
                "label": f"问题{qid}",
                "type": "question",
                "detail": {
                    "question_id": qid,
                    "question": question,
                    "prediction": pred,
                    "ground_truth": gt,
                    "correct": bool(case.get("correct", False)),
                    "exact": exact,
                    "soft": soft,
                },
            }
        )

        for i, t in enumerate(trace):
            sub_q = t.get("sub_question", "")
            sq_id = f"sq-{qid}-{i}"
            nodes.append(
                {
                    "id": sq_id,
                    "label": f"子问题{i+1}",
                    "type": "subq",
                    "detail": {"sub_question": sub_q},
                }
            )
            links.append({"source": q_node_id, "target": sq_id, "type": "decompose"})

            candidates = t.get("candidates", [])
            scores = t.get("scores", [])
            selected_idx = int(t.get("selected_index", 0)) if candidates else -1
            for j, c in enumerate(candidates):
                cid = f"cand-{qid}-{i}-{j}"
                score = float(scores[j]) if j < len(scores) else 0.0
                selected = j == selected_idx
                nodes.append(
                    {
                        "id": cid,
                        "label": f"候选{j+1}",
                        "type": "candidate_selected" if selected else "candidate",
                        "detail": {
                            "thought": c.get("thought", ""),
                            "answer": c.get("answer", ""),
                            "score": score,
                            "selected": selected,
                        },
                    }
                )
                links.append({"source": sq_id, "target": cid, "type": "expand"})

        pred_id = f"pred-{qid}"
        gt_id = f"gt-{qid}"
        nodes.append(
            {
                "id": pred_id,
                "label": "预测",
                "type": "pred",
                "detail": {"prediction": pred},
            }
        )
        nodes.append(
            {
                "id": gt_id,
                "label": "真值",
                "type": "gt",
                "detail": {"ground_truth": gt},
            }
        )
        links.append({"source": q_node_id, "target": pred_id, "type": "finalize"})
        links.append({"source": q_node_id, "target": gt_id, "type": "reference"})

        return {
            "question_id": qid,
            "question": question,
            "prediction": pred,
            "ground_truth": gt,
            "correct": bool(case.get("correct", False)),
            "exact": exact,
            "soft": soft,
            "nodes": nodes,
            "links": links,
        }

    @app.route("/")
    @admin_required
    def index():
        return render_template(
            "index.html",
            username=session.get("username", "管理员"),
            portal_url=portal_base_url(),
        )

    @app.route("/api/cases")
    @admin_required
    def api_cases():
        rows = []
        for c in load_cases():
            pred = c.get("prediction", "")
            gt = c.get("ground_truth", "")
            exact, soft = eval_exact_soft(pred, gt)
            rows.append(
                {
                    "image_id": str(c.get("image_id", "")),
                    "question_id": str(c.get("question_id", "")),
                    "question": c.get("question", ""),
                    "prediction": pred,
                    "ground_truth": gt,
                    "answer": c.get("answer", gt),
                    "correct": bool(c.get("correct", False)),
                    "exact": exact,
                    "soft": soft,
                    "has_image": bool(c.get("image")),
                }
            )
        return jsonify(rows)

    @app.route("/api/users")
    @admin_required
    def api_users():
        """管理员用户信息：直接读取 users 与 messages，确保已有用户和问答记录可见。"""
        users = []
        portal_db.sync_all_user_info_records()
        for user in portal_db.list_users(limit=1000):
            uid = str(user.get("_id", ""))
            messages = portal_db.list_user_messages(uid, limit=1000)
            assistant_by_question = {}
            for msg in messages:
                if msg.get("role") == "assistant":
                    key = (msg.get("question") or msg.get("content") or "").strip()
                    if key and key not in assistant_by_question:
                        assistant_by_question[key] = msg

            records = []
            history_idx = 1
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                question = msg.get("question") or msg.get("content") or ""
                answer_msg = assistant_by_question.get(question.strip(), {})
                image_filename = msg.get("image_filename") or Path(str(msg.get("image_path", ""))).name
                records.append(
                    {
                        "message_id": str(msg.get("_id", "")),
                        "history_image_id": history_idx,
                        "history_question_id": str(msg.get("_id", "")),
                        "question": question,
                        "answer": answer_msg.get("answer") or answer_msg.get("content") or "",
                        "image_url": f"/portal_uploads/{uid}/{image_filename}"
                        if image_filename
                        else "",
                        "created_at": msg.get("created_at").isoformat()
                        if msg.get("created_at")
                        else "",
                    }
                )
                history_idx += 1

            users.append(
                {
                    "id": uid,
                    "username": user.get("username", ""),
                    "account": user.get("username", ""),
                    "password": user.get("password_hash", ""),
                    "email": user.get("email", ""),
                    "role": portal_db.user_role(user),
                    "records": records,
                }
            )
        return jsonify(users)

    @app.route("/api/database/summary")
    @admin_required
    def api_database_summary():
        case_rows = load_cases()
        user_rows = portal_db.list_user_info_records()
        history_count = sum(len(u.get("histories", [])) for u in user_rows)
        return jsonify(
            {
                "database": "vision_qa_portal",
                "case_database": {
                    "collection": "case_records",
                    "description": "案例数据库：保存 image_id、question_id、image、question、prediction、answer。",
                    "count": len(case_rows),
                },
                "user_info_database": {
                    "collection": "user_info_records",
                    "description": "用户信息数据库：保存 account、password、email 与历史问答记录。",
                    "count": len(user_rows),
                    "history_count": history_count,
                },
                "source": "MongoDB；若服务器不可用则使用本地 Mongita 文件库。",
            }
        )

    @app.route("/api/case/<qid>")
    @admin_required
    def api_case(qid: str):
        for c in load_cases():
            if str(c.get("question_id", "")) == qid:
                graph = build_graph(c)
                graph["image_url"] = f"/api/case/{qid}/image" if c.get("image") else ""
                return jsonify(graph)
        return jsonify({"error": f"question_id={qid} not found"}), 404

    @app.route("/api/case/<qid>/image")
    @admin_required
    def api_case_image(qid: str):
        for c in load_cases():
            if str(c.get("question_id", "")) == qid:
                image_path = Path(str(c.get("image", "")))
                if image_path.exists() and image_path.is_file():
                    return send_file(str(image_path))
                return jsonify({"error": "image not found"}), 404
        return jsonify({"error": f"question_id={qid} not found"}), 404

    @app.route("/uploads/<path:filename>")
    @admin_required
    def uploads(filename):
        return send_from_directory(str(UPLOAD_DIR), filename)

    @app.route("/portal_uploads/<user_id>/<path:filename>")
    @admin_required
    def portal_uploads(user_id, filename):
        folder = PROJECT_ROOT / "agent_workspace" / "chat_vqa_portal" / "uploads" / user_id
        return send_from_directory(str(folder), filename)

    @app.route("/api/ask", methods=["POST"])
    @admin_required
    def api_ask():
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
        graph = build_graph(
            {
                "question_id": "new",
                "question": question,
                "prediction": answer,
                "ground_truth": "",
                "correct": False,
                "trace": solved.get("trace", []),
            }
        )
        return jsonify(
            {
                "answer": answer,
                "visual_description": visual_desc,
                "graph": graph,
                "image_url": f"/uploads/{filename}",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
