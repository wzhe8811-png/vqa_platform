"""跨服务共享的认证配置（7860 trace / 7862 portal 共用 Session）。"""

import os

ROLE_USER = "user"
ROLE_ADMIN = "admin"

DEFAULT_SECRET_KEY = "vision-qa-portal-dev-change-in-production"


def secret_key() -> str:
    return os.environ.get("PORTAL_SECRET_KEY", DEFAULT_SECRET_KEY)


def portal_base_url() -> str:
    return os.environ.get("PORTAL_BASE_URL", "").rstrip("/")


def trace_base_url() -> str:
    return os.environ.get("TRACE_BASE_URL", "").rstrip("/")


def redirect_url_for_role(role: str) -> str:
    if role == ROLE_ADMIN:
        return trace_base_url() + "/"
    return portal_base_url() + "/chat"
