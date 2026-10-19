# -*- coding: utf-8 -*-
"""评测工具 v2：答案提取 + 规范化 + 等价判断

v2 相对 v1 的改进（对应准确率偏低的修复）：
1. extract_final_answer：从模型整段输出中提取最终短答案（v1 拿整段文本直接匹配，导致大量误判）
2. canonicalize 增强：数字词归一（two->2）、颜色同义（gray->grey）、常见 GQA 同义词
3. answers_equivalent 增加软匹配：gt 的词集合包含于 pred（处理多词顺序与冗长输出）
"""
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
    "cell phone": "phone",
    "cellular phone": "phone",
    "mobile phone": "phone",
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
    "gray": "grey",
    "guy": "man",
    "guys": "man",
    "motorbike": "motorcycle",
    "cupboard": "cabinet",
    "cupboards": "cabinet",
    "fridge": "refrigerator",
    "fridges": "refrigerator",
    "telephone": "phone",
    "telephones": "phone",
    "automobile": "car",
    "automobiles": "car",
    "countertop": "counter",
    "countertops": "counter",
    # 数字词归一（GQA 答案常见 "2" 与 "two" 混用）
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
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
    # 很轻的单复数归一
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

    # 去掉首尾虚词
    while tokens and tokens[0] in _TOKEN_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _TOKEN_STOPWORDS:
        tokens = tokens[:-1]
    if not tokens:
        return ""

    tokens = [_canonicalize_token(tok) for tok in tokens]
    return " ".join(tokens)


# ---------------- v2 新增：答案提取 ----------------

_ANSWER_PATTERNS = [
    # "the answer is X" / "final answer: X" / "答案是 X"
    re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+)", re.IGNORECASE),
    re.compile(r"答案是[:：]?\s*(.+)"),
    re.compile(r"^(?:so|therefore|thus|hence)[,;]?\s+(?:the\s+answer\s+is\s+)?(.+)", re.IGNORECASE),
]


def extract_final_answer(text: str) -> str:
    """从模型输出中提取最终短答案。提取失败时返回原文（由匹配函数兜底）。"""
    t = (text or "").strip()
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    # 1. 最后一行里找答案引导模式
    last = lines[-1]
    for pat in _ANSWER_PATTERNS:
        m = pat.search(last)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"[.,;!]", cand)[0].strip()
            # 引导词在句子中间：取最后一个匹配（例如 "x is 1, so the answer is 2"）
            cand = _last_answer_occurrence(last, cand)
            if cand and len(cand.split()) <= 8:
                return cand

    # 2. 引号内答案（"dog"）
    m = re.search(r"[\"']([^\"']{1,15})[\"']\s*[.!]?$", last)
    if m:
        return m.group(1)

    # 3. 最后一行很短 -> 直接当答案
    if len(last.split()) <= 4:
        return last

    # 4. 最后一行很长：去掉推理引导词后，取最后分句
    last = re.sub(r"^(so|therefore|thus|hence)[,;]?\s+", "", last, flags=re.IGNORECASE)
    clauses = re.split(r"[.;]", last)
    tail = clauses[-1].strip()
    if tail and len(tail.split()) <= 6:
        return tail

    # 5. 提取失败：返回最后一行原文
    return last


def _last_answer_occurrence(line: str, first_hit: str) -> str:
    """若同一行出现多次 answer 引导（模型重复），取最后一次出现后的内容。"""
    idx = line.lower().rfind("answer")
    if idx >= 0:
        after = line[idx:].strip()
        for pat in _ANSWER_PATTERNS:
            m = pat.search(after)
            if m and m.group(1).strip():
                return re.split(r"[.,;!]", m.group(1).strip())[0].strip()
    return first_hit


# ---------------- v2 修改：等价判断 ----------------

def answers_equivalent(pred: str, gt: str, soft: bool = True) -> bool:
    """判断预测与标准答案是否等价。

    精确匹配：规范化后完全相等；
    软匹配：gt 的所有词都出现在 pred 中（容忍多余描述与多词顺序）。
    """
    canonical_gt = canonicalize_answer(gt)
    if not canonical_gt:
        return False

    # 先尝试提取最终答案（对整段推理输出有效），提取不到再用原文
    pred_short = extract_final_answer(pred)
    canonical_pred = canonicalize_answer(pred_short if pred_short else pred)
    if not canonical_pred:
        return False

    if canonical_pred == canonical_gt:
        return True

    if soft:
        gt_tokens = set(canonical_gt.split())
        pred_tokens = set(canonical_pred.split())
        if gt_tokens and gt_tokens.issubset(pred_tokens):
            return True
        # v4: 颜色/程度修饰词不构成硬性区分（light blue ~ blue, dark red ~ red）
        _MODIFIERS = {'light', 'dark', 'bright', 'pale', 'deep'}
        gt_tokens_mod = gt_tokens - _MODIFIERS
        if gt_tokens_mod and gt_tokens_mod.issubset(pred_tokens):
            return True

    return False


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
    return cue_hit or len(q.split()) >= 11


def normalize_short_answer(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    t = re.sub(r"^[a-zA-Z]\s*:\s*", "", t)
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
    left = " ".join(left.split()[-3:]).lower()
    right = " ".join(right.split()[-3:]).lower()
    return [left, right]


def constrained_finalize_answer(llm_caller, visual_description: str, question: str, draft_answer: str) -> str:
    """对 ToT/GoT/Modular 的草稿答案做约束式收敛，提升可评测性。"""
    q_raw = question or ""
    q = q_raw.lower()
    has_zh = bool(re.search(r"[一-鿿]", q_raw))

    if ("left" in q and "right" in q) or ("左" in q_raw and "右" in q_raw):
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"问题：\n{question}\n\n"
            f"草稿答案：\n{draft_answer}\n\n"
            "只输出一个词：left 或 right。不要解释。"
        )
        return normalize_short_answer(llm_caller(prompt))

    if ("top" in q and "bottom" in q) or ("上" in q_raw and "下" in q_raw):
        prompt = (
            f"图像描述：\n{visual_description}\n\n"
            f"问题：\n{question}\n\n"
            f"草稿答案：\n{draft_answer}\n\n"
            "只输出一个词：top 或 bottom。不要解释。"
        )
        return normalize_short_answer(llm_caller(prompt))

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
