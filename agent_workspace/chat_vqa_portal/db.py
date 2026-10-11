"""MongoDB 文档库：用户、会话、问答记录。"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from pymongo import ASCENDING, MongoClient
from pymongo.errors import ServerSelectionTimeoutError

_SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from auth_config import ROLE_ADMIN, ROLE_USER  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "mongo_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_client = None
_db = None


def _utcnow():
    return datetime.now(timezone.utc)


def get_db():
    global _client, _db
    if _db is not None:
        return _db

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
    try:
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2500)
        _client.admin.command("ping")
        _db = _client["vision_qa_portal"]
        _using_mongita = False
    except (ServerSelectionTimeoutError, Exception):
        from mongita import MongitaClientDisk

        _client = MongitaClientDisk(str(DATA_DIR))
        _db = _client["vision_qa_portal"]

    _ensure_indexes(_db)
    ensure_bootstrap_admin()
    return _db


def _ensure_indexes(db):
    try:
        db.users.create_index("username", unique=True)
        db.users.create_index("email", unique=True)
        db.conversations.create_index(
            [("user_id", ASCENDING), ("updated_at", ASCENDING)]
        )
        db.messages.create_index(
            [("conversation_id", ASCENDING), ("created_at", ASCENDING)]
        )
        db.case_records.create_index(
            [("image_id", ASCENDING), ("question_id", ASCENDING)], unique=True
        )
        db.user_info_records.create_index("account", unique=True)
    except Exception:
        # Mongita 等嵌入式后端可能不支持 unique 索引，由应用层校验唯一性
        pass


def oid(value):
    return ObjectId(str(value))


# --- users ---


def user_role(user) -> str:
    if not user:
        return ROLE_USER
    role = (user.get("role") or ROLE_USER).strip().lower()
    return role if role in {ROLE_USER, ROLE_ADMIN} else ROLE_USER


def is_admin(user) -> bool:
    return user_role(user) == ROLE_ADMIN


def create_user(username: str, email: str, password_hash: str, role: str = ROLE_USER):
    db = get_db()
    role = (role or ROLE_USER).strip().lower()
    if role not in {ROLE_USER, ROLE_ADMIN}:
        role = ROLE_USER
    doc = {
        "username": username,
        "email": email.lower(),
        "password_hash": password_hash,
        "role": role,
        "created_at": _utcnow(),
    }
    result = db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def ensure_bootstrap_admin():
    """首次启动时创建默认管理员（可通过环境变量覆盖）。"""
    import auth_utils

    username = (os.environ.get("ADMIN_USERNAME") or "admin").strip()
    password = os.environ.get("ADMIN_PASSWORD") or "admin123456"
    email = (os.environ.get("ADMIN_EMAIL") or "admin@local.dev").strip().lower()

    db = get_db()
    existing = db.users.find_one({"username": username})
    if existing:
        if user_role(existing) != ROLE_ADMIN:
            db.users.update_one(
                {"_id": existing["_id"]},
                {"$set": {"role": ROLE_ADMIN}},
            )
        return existing

    return create_user(
        username,
        email,
        auth_utils.hash_password(password),
        role=ROLE_ADMIN,
    )


def find_user_by_username(username: str):
    return get_db().users.find_one({"username": username})


def find_user_by_email(email: str):
    return get_db().users.find_one({"email": email.lower()})


def find_user_by_id(user_id):
    return get_db().users.find_one({"_id": oid(user_id)})


# --- conversations ---


def list_conversations(user_id, limit=80):
    db = get_db()
    cursor = (
        db.conversations.find({"user_id": str(user_id)})
        .sort("updated_at", -1)
        .limit(limit)
    )
    return list(cursor)


def create_conversation(user_id, title="新对话"):
    db = get_db()
    now = _utcnow()
    doc = {
        "user_id": str(user_id),
        "title": title,
        "created_at": now,
        "updated_at": now,
    }
    result = db.conversations.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_conversation(conversation_id, user_id):
    return get_db().conversations.find_one(
        {"_id": oid(conversation_id), "user_id": str(user_id)}
    )


def touch_conversation(conversation_id, title=None):
    update = {"updated_at": _utcnow()}
    if title:
        update["title"] = title
    get_db().conversations.update_one({"_id": oid(conversation_id)}, {"$set": update})


def delete_conversation(conversation_id, user_id):
    db = get_db()
    conv = get_conversation(conversation_id, user_id)
    if not conv:
        return False
    db.messages.delete_many({"conversation_id": str(conversation_id)})
    db.conversations.delete_one({"_id": conv["_id"]})
    return True


# --- messages ---


def list_messages(conversation_id, user_id):
    conv = get_conversation(conversation_id, user_id)
    if not conv:
        return None
    return list(
        get_db()
        .messages.find({"conversation_id": str(conversation_id)})
        .sort("created_at", ASCENDING)
    )


def insert_message(
    conversation_id,
    user_id,
    *,
    question,
    answer,
    image_path,
    image_filename,
    graph=None,
    trace=None,
    visual_description=None,
):
    db = get_db()
    now = _utcnow()
    user_msg = {
        "conversation_id": str(conversation_id),
        "user_id": str(user_id),
        "role": "user",
        "content": question,
        "question": question,
        "image_path": image_path,
        "image_filename": image_filename,
        "created_at": now,
    }
    db.messages.insert_one(user_msg)

    assistant_msg = {
        "conversation_id": str(conversation_id),
        "user_id": str(user_id),
        "role": "assistant",
        "content": answer,
        "question": question,
        "answer": answer,
        "image_path": image_path,
        "image_filename": image_filename,
        "graph": graph,
        "trace": trace,
        "visual_description": visual_description,
        "created_at": _utcnow(),
    }
    result = db.messages.insert_one(assistant_msg)
    assistant_msg["_id"] = result.inserted_id
    touch_conversation(conversation_id)
    return result.inserted_id, assistant_msg


def serialize_doc(doc):
    if doc is None:
        return None
    out = {}
    for k, v in doc.items():
        if k == "_id":
            out["id"] = str(v)
        elif isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def serialize_docs(docs):
    return [serialize_doc(d) for d in docs]


def list_users(limit=200):
    return list(get_db().users.find({}).sort("created_at", -1).limit(limit))


def list_user_messages(user_id, limit=500):
    return list(
        get_db()
        .messages.find({"user_id": str(user_id)})
        .sort("created_at", -1)
        .limit(limit)
    )


# --- case database：案例数据库，对应论文表4-1 ---


def upsert_case_record(
    *,
    image_id,
    question_id,
    image,
    question,
    prediction,
    answer,
    trace=None,
    correct=False,
):
    """保存一条视觉问答案例记录，image_id + question_id 作为联合唯一标识。"""
    db = get_db()
    doc = {
        "image_id": int(image_id) if str(image_id).isdigit() else image_id,
        "question_id": int(question_id) if str(question_id).isdigit() else question_id,
        "image": image or "",
        "question": question or "",
        "prediction": prediction or "",
        "answer": answer or "",
        "ground_truth": answer or "",
        "trace": trace or [],
        "correct": bool(correct),
        "updated_at": _utcnow(),
    }
    result = db.case_records.update_one(
        {"image_id": doc["image_id"], "question_id": doc["question_id"]},
        {"$set": doc, "$setOnInsert": {"created_at": _utcnow()}},
        upsert=True,
    )
    return result


def import_cases_from_json(cases):
    """把原 JSON 测试案例导入 MongoDB 案例数据库。"""
    count = 0
    for idx, case in enumerate(cases or []):
        image = case.get("image", "")
        image_id = case.get("image_id") or case.get("imageId") or idx + 1
        question_id = case.get("question_id") or case.get("questionId") or idx + 1
        upsert_case_record(
            image_id=image_id,
            question_id=question_id,
            image=image,
            question=case.get("question", ""),
            prediction=case.get("prediction", ""),
            answer=case.get("answer") or case.get("ground_truth") or "",
            trace=case.get("trace", []),
            correct=case.get("correct", False),
        )
        count += 1
    return count


def list_case_records(limit=2000):
    return list(get_db().case_records.find({}).sort("question_id", ASCENDING).limit(limit))


# --- user information database：用户信息数据库，对应论文表4-2 ---


def sync_user_info_record(user_id):
    """汇总用户账户信息及历史问答记录，形成用户信息数据库记录。"""
    user = find_user_by_id(user_id)
    if not user:
        return None

    messages = list_user_messages(user_id)
    histories = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        histories.append(
            {
                "history_image_id": idx + 1,
                "history_question_id": str(msg.get("_id", "")),
                "history_image": msg.get("image_path", ""),
                "history_question": msg.get("question", ""),
                "history_answer": msg.get("answer") or msg.get("content") or "",
            }
        )

    doc = {
        "account": user.get("username", ""),
        "password": user.get("password_hash", ""),
        "email": user.get("email", ""),
        "user_id": str(user.get("_id", "")),
        "histories": histories,
        "updated_at": _utcnow(),
    }
    get_db().user_info_records.update_one(
        {"account": doc["account"]},
        {"$set": doc, "$setOnInsert": {"created_at": _utcnow()}},
        upsert=True,
    )
    return doc


def sync_all_user_info_records():
    docs = []
    for user in list_users(limit=1000):
        doc = sync_user_info_record(user.get("_id"))
        if doc:
            docs.append(doc)
    return docs


def list_user_info_records(limit=1000):
    sync_all_user_info_records()
    return list(get_db().user_info_records.find({}).sort("account", ASCENDING).limit(limit))
