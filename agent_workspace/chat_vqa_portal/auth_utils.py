import re

import bcrypt

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\u4e00-\u9fff]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def validate_register(username: str, email: str, password: str):
    username = (username or "").strip()
    email = (email or "").strip()
    password = password or ""

    if not USERNAME_RE.match(username):
        return "用户名需为 3-32 位，支持字母、数字、下划线或中文"
    if not EMAIL_RE.match(email):
        return "邮箱格式不正确"
    if len(password) < 6:
        return "密码至少 6 位"
    return None
