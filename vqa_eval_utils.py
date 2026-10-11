import re
from typing import List


_LEADING_LABEL_RE = re.compile(r"^[a-zA-Z]\s*:\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_TRAILING_PUNCT_RE = re.compile(r"^[\s\.,;:!?\-_'\"`]+|[\s\.,;:!?\-_'\"`]+$")
_TOKEN_STOPWORDS = {"a", "an", "the", "on", "in", "at", "of"}

_PHRASE_CANONICAL_MAP = {
    "on the street": "road",
    "in the street": "road",
    "street": "road",
    "roadway": "road",
}

_TOKEN_CANONICAL_MAP = {
    "y": "yes",
    "yeah": "yes",
    "yep": "yes",
    "n": "no",
    "nope": "no",
    "street": "road",
    "roadway": "road",
    "bike": "bicycle",
    "bikes": "bicycle",
    "bicycles": "bicycle",
    "tv": "television",
    "tvs": "television",
    "couch": "sofa",
    "couches": "sofa",
}


def _normalize_eval_text(text: str) -> str:
    if text is None:
        return ""
    out = text.strip().lower()
    out = _LEADING_LABEL_RE.sub("", out)
    out = _TRAILING_PUNCT_RE.sub("", out)
    out = _MULTI_SPACE_RE.sub(" ", out)
    return out


def _canonicalize_token(token: str) -> str:
    t = token.strip().lower()
    t = _TOKEN_CANONICAL_MAP.get(t, t)
    if t in {"yes", "no"}:
        return t
    # Very light singularization for short answer nouns.
    if len(t) > 3 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 2 and t.endswith("s"):
        t = t[:-1]
    return t


def canonicalize_answer(text: str) -> str:
    norm = _normalize_eval_text(text)
    if not norm:
        return ""

    if norm in _PHRASE_CANONICAL_MAP:
        return _PHRASE_CANONICAL_MAP[norm]

    norm = _NON_ALNUM_RE.sub(" ", norm)
    tokens = [tok for tok in _MULTI_SPACE_RE.sub(" ", norm).strip().split(" ") if tok]
    if not tokens:
        return ""

    # Drop leading/trailing function words to keep location answers compact.
    while tokens and tokens[0] in _TOKEN_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _TOKEN_STOPWORDS:
        tokens = tokens[:-1]
    if not tokens:
        return ""

    tokens = [_canonicalize_token(tok) for tok in tokens]
    return " ".join(tokens)


def answers_equivalent(pred: str, gt: str) -> bool:
    canonical_gt = canonicalize_answer(gt)
    if not canonical_gt:
        return False
    canonical_pred = canonicalize_answer(pred)
    return canonical_pred == canonical_gt


def is_complex_question(question: str) -> bool:
    """简单启发式：筛选多跳/关系/比较类问题。"""
    if not question:
        return False
    q = question.lower()
    cues = [
        "left",
        "right",
        "behind",
        "in front of",
        "between",
        "both",
        "either",
        "or",
        "and",
        "not",
        "near",
        "next to",
        "same",
        "different",
        "more",
        "less",
        "top",
        "bottom",
    ]
    cue_hit = any(c in q for c in cues)
    # 适度提高难度阈值：长度+关系词
    return cue_hit or len(q.split()) >= 11


def normalize_short_answer(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    t = re.sub(r"^[a-zA-Z]\s*:\s*", "", t)  # 去掉 "T: "
    t = t.strip().strip(" .,:;!?\"'`")
    t_low = t.lower()
    canonical_map = {
        "是": "yes",
        "否": "no",
        "对": "yes",
        "不对": "no",
        "正确": "yes",
        "错误": "no",
        "左": "left",
        "左边": "left",
        "左侧": "left",
        "右": "right",
        "右边": "right",
        "右侧": "right",
        "上": "top",
        "上方": "top",
        "上面": "top",
        "下": "bottom",
        "下方": "bottom",
        "下面": "bottom",
    }
    if t in canonical_map:
        return canonical_map[t]
    if t_low in canonical_map:
        return canonical_map[t_low]
    return t_low


def extract_or_options(question: str) -> List[str]:
    """粗提取 'A or B' 里的 A/B 选项。"""
    q = question.strip().rstrip("?")
    m = re.search(r"(.+?)\s+or\s+(.+)$", q, flags=re.IGNORECASE)
    if not m:
        return []
    left = m.group(1).strip()
    right = m.group(2).strip()
    # 只保留最后几个词，减少噪声
    left = " ".join(left.split()[-3:]).lower()
    right = " ".join(right.split()[-3:]).lower()
    return [left, right]


def constrained_finalize_answer(llm_caller, visual_description: str, question: str, draft_answer: str) -> str:
    """对 ToT/GoT/Modular 的草稿答案做约束式收敛，提升可评测性。"""
    q_raw = question or ""
    q = q_raw.lower()
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", q_raw))

    # 左右题强约束（优先于 yes/no，避免方向题被错误收敛成 yes/no）
    if ("left" in q and "right" in q) or ("左" in q_raw and "右" in q_raw):
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"问题：\n{question}\n\n"
            f"草稿答案：\n{draft_answer}\n\n"
            "只输出一个词：left 或 right。不要解释。"
        )
        return normalize_short_answer(llm_caller(prompt))

    # 上下题强约束
    if ("top" in q and "bottom" in q) or ("上" in q_raw and "下" in q_raw):
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"问题：\n{question}\n\n"
            f"草稿答案：\n{draft_answer}\n\n"
            "只输出一个词：top 或 bottom。不要解释。"
        )
        return normalize_short_answer(llm_caller(prompt))

    # yes/no 题强约束
    zh_yes_no_cues = ("是否", "是不是", "能否", "可否", "有没有", "有无")
    is_yn = q.startswith(("is ", "are ", "does ", "do ", "did ", "can ", "could ", "has ", "have ")) or any(
        cue in q_raw for cue in zh_yes_no_cues
    )
    if is_yn:
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"问题：\n{question}\n\n"
            f"草稿答案：\n{draft_answer}\n\n"
            "只输出一个词：yes 或 no。不要解释。"
        )
        return normalize_short_answer(llm_caller(prompt))

    # 二选一题约束
    opts = extract_or_options(question)
    if len(opts) == 2:
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"问题：\n{question}\n\n"
            f"草稿答案：\n{draft_answer}\n\n"
            f"只能从以下两个选项中选择一个：\n1) {opts[0]}\n2) {opts[1]}\n"
            "只输出被选中的选项文本，不要解释。"
        )
        return normalize_short_answer(llm_caller(prompt))

    # 通用短答案收敛
    if has_zh:
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"问题：\n{question}\n\n"
            f"草稿答案：\n{draft_answer}\n\n"
            "请输出精炼的中文最终答案（不超过 6 个字），不要解释。"
        )
    else:
        prompt = (
            f"Visual Description:\n{visual_description}\n\n"
            f"Question:\n{question}\n\n"
            f"Draft Answer:\n{draft_answer}\n\n"
            "Return a concise final answer in English (<= 3 words), no explanation."
        )
    ans = normalize_short_answer(llm_caller(prompt))
    if has_zh:
        return ans
    return ans.lower()
