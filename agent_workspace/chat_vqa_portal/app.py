import os
import sys
import uuid
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

import auth_utils
import db
from vqa_service import run_vqa

_SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from auth_config import (  # noqa: E402
    portal_base_url,
    redirect_url_for_role,
    secret_key,
    trace_base_url,
)

PROJECT_ROOT = Path("/root/autodl-tmp/vision_qa_platform")
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PORT = int(os.environ.get("PORTAL_PORT", "7862"))


def _current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.find_user_by_id(user_id)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "请先登录"}), 401
            return redirect(url_for("login_page"))
        if not _current_user():
            session.clear()
            if request.path.startswith("/api/"):
                return jsonify({"error": "请先登录"}), 401
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)

    return wrapped


def user_only(view):
    """普通用户入口：管理员应使用 7860 轨迹浏览界面。"""

    @wraps(view)
    def wrapped(*args, **kwargs):
        user = _current_user()
        if not user:
            if request.path.startswith("/api/"):
                return jsonify({"error": "请先登录"}), 401
            return redirect(url_for("login_page"))
        if db.is_admin(user):
            if request.path.startswith("/api/"):
                return jsonify({"error": "管理员请使用轨迹浏览界面", "redirect": trace_base_url()}), 403
            return redirect(trace_base_url() + "/")
        return view(*args, **kwargs)

    return wrapped


def _redirect_for_session():
    user = _current_user()
    if not user:
        session.clear()
        return redirect(url_for("login_page"))
    return redirect(redirect_url_for_role(db.user_role(user)))


def create_app():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.secret_key = secret_key()

    db.get_db()

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True, "service": "chat_vqa_portal", "port": PORT})

    @app.route("/login")
    def login_page():
        if session.get("user_id"):
            return _redirect_for_session()
        return render_template(
            "login.html",
            portal_url=portal_base_url(),
            trace_url=trace_base_url(),
        )

    @app.route("/register")
    def register_page():
        if session.get("user_id"):
            return _redirect_for_session()
        return render_template("register.html")

    @app.route("/api/auth/register", methods=["POST"])
    def api_register():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        email = (data.get("email") or "").strip()
        password = data.get("password") or ""

        err = auth_utils.validate_register(username, email, password)
        if err:
            return jsonify({"error": err}), 400

        if db.find_user_by_username(username):
            return jsonify({"error": "用户名已存在"}), 400
        if db.find_user_by_email(email):
            return jsonify({"error": "邮箱已被注册"}), 400

        user = db.create_user(
            username, email, auth_utils.hash_password(password), role=db.ROLE_USER
        )
        session["user_id"] = str(user["_id"])
        session["username"] = username
        session["role"] = db.ROLE_USER
        return jsonify(
            {
                "ok": True,
                "role": db.ROLE_USER,
                "redirect_url": redirect_url_for_role(db.ROLE_USER),
                "user": {
                    "id": str(user["_id"]),
                    "username": username,
                    "email": email,
                    "role": db.ROLE_USER,
                },
            }
        )

    @app.route("/api/auth/login", methods=["POST"])
    def api_login():
        data = request.get_json(silent=True) or {}
        account = (data.get("account") or data.get("username") or "").strip()
        password = data.get("password") or ""

        if not account or not password:
            return jsonify({"error": "请输入账号和密码"}), 400

        user = db.find_user_by_username(account)
        if user is None and "@" in account:
            user = db.find_user_by_email(account)
        if user is None:
            return jsonify({"error": "用户不存在"}), 401
        if not auth_utils.verify_password(password, user["password_hash"]):
            return jsonify({"error": "密码错误"}), 401

        session["user_id"] = str(user["_id"])
        session["username"] = user["username"]
        role = db.user_role(user)
        session["role"] = role
        return jsonify(
            {
                "ok": True,
                "role": role,
                "redirect_url": redirect_url_for_role(role),
                "user": {
                    "id": str(user["_id"]),
                    "username": user["username"],
                    "email": user["email"],
                    "role": role,
                },
            }
        )

    @app.route("/api/auth/logout", methods=["POST"])
    def api_logout():
        session.clear()
        return jsonify({"ok": True})

    @app.route("/api/auth/me")
    @login_required
    def api_me():
        user = db.find_user_by_id(session["user_id"])
        if not user:
            session.clear()
            return jsonify({"error": "用户不存在"}), 401
        return jsonify(
            {
                "id": str(user["_id"]),
                "username": user["username"],
                "email": user["email"],
                "role": db.user_role(user),
            }
        )

    @app.route("/")
    def index():
        if session.get("user_id"):
            return _redirect_for_session()
        return redirect(url_for("login_page"))

    @app.route("/chat")
    @login_required
    @user_only
    def chat_page():
        return render_template(
            "chat.html", username=session.get("username", "用户")
        )

    @app.route("/uploads/<user_id>/<path:filename>")
    @login_required
    @user_only
    def uploads(user_id, filename):
        if user_id != session["user_id"]:
            return jsonify({"error": "无权访问"}), 403
        folder = UPLOAD_DIR / user_id
        return send_from_directory(str(folder), filename)

    @app.route("/api/conversations", methods=["GET"])
    @login_required
    @user_only
    def list_conversations():
        convs = db.list_conversations(session["user_id"])
        return jsonify({"conversations": db.serialize_docs(convs)})

    @app.route("/api/conversations", methods=["POST"])
    @login_required
    @user_only
    def create_conversation():
        data = request.get_json(silent=True) or {}
        title = (data.get("title") or "新对话").strip() or "新对话"
        conv = db.create_conversation(session["user_id"], title=title)
        return jsonify({"conversation": db.serialize_doc(conv)})

    @app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
    @login_required
    @user_only
    def delete_conversation(conversation_id):
        ok = db.delete_conversation(conversation_id, session["user_id"])
        if not ok:
            return jsonify({"error": "会话不存在"}), 404
        return jsonify({"ok": True})

    @app.route("/api/conversations/<conversation_id>/messages", methods=["GET"])
    @login_required
    @user_only
    def get_messages(conversation_id):
        messages = db.list_messages(conversation_id, session["user_id"])
        if messages is None:
            return jsonify({"error": "会话不存在"}), 404
        serialized = []
        for m in messages:
            item = db.serialize_doc(m)
            if m.get("image_filename"):
                item["image_url"] = (
                    f"/uploads/{session['user_id']}/{m['image_filename']}"
                )
            serialized.append(item)
        return jsonify({"messages": serialized})

    @app.route("/api/conversations/<conversation_id>/ask", methods=["POST"])
    @login_required
    @user_only
    def ask_in_conversation(conversation_id):
        conv = db.get_conversation(conversation_id, session["user_id"])
        if not conv:
            return jsonify({"error": "会话不存在"}), 404

        question = (request.form.get("question") or "").strip()
        image = request.files.get("image")

        if not question:
            return jsonify({"error": "请输入问题"}), 400
        if image is None or not image.filename:
            return jsonify({"error": "请上传图片"}), 400

        ext = Path(secure_filename(image.filename)).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return jsonify({"error": "不支持的图片格式"}), 400

        user_upload_dir = UPLOAD_DIR / session["user_id"]
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = user_upload_dir / filename
        image.save(str(save_path))

        result = run_vqa(str(save_path), question)

        title = conv.get("title") or "新对话"
        if title == "新对话":
            short = question[:24] + ("…" if len(question) > 24 else "")
            db.touch_conversation(conversation_id, title=short)

        _, assistant_msg = db.insert_message(
            conversation_id,
            session["user_id"],
            question=result.get("question_zh") or question,
            answer=result["answer"],
            image_path=str(save_path),
            image_filename=filename,
            graph=result.get("graph"),
            trace=result.get("trace"),
            visual_description=result.get("visual_description"),
        )

        image_url = f"/uploads/{session['user_id']}/{filename}"
        return jsonify(
            {
                "answer": result["answer"],
                "visual_description": result.get("visual_description"),
                "graph": result.get("graph"),
                "trace": result.get("trace"),
                "image_url": image_url,
                "message_id": str(assistant_msg.get("_id", "")),
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
