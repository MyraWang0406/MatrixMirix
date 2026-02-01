"""
Vertical 语料决定器：ecommerce / casual_game 两套词库。
bucket（桶）与 phrase（具体短语）分离，保证 pydantic 校验不崩。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
CONFIG_PATH = SAMPLES_DIR / "vertical_config.json"

_config: dict[str, Any] | None = None
_VERTICALS = ("ecommerce", "casual_game")

# 通用桶（Literal 校验用）
WHY_YOU_BUCKETS = ("更省钱", "更省事", "更强效果", "更匹配我", "更安全可信", "更好体验", "其他")
WHY_NOW_BUCKETS = ("限时稀缺", "节点事件", "痛点升高", "机会出现", "社会驱动", "损失厌恶", "即时替代", "其他")


def _normalize_vertical(v: str) -> str:
    v = (v or "casual_game").lower().strip()
    return v if v in _VERTICALS else "casual_game"


def load_vertical_config() -> dict[str, Any]:
    global _config
    if _config is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config = json.load(f)
        else:
            _config = {}
    return _config


def get_corpus(vertical: str) -> dict[str, Any]:
    cfg = load_vertical_config()
    corpus = cfg.get("corpus", {})
    return dict(corpus.get(_normalize_vertical(vertical), {}))


def get_why_you_buckets() -> list[str]:
    """通用 Why you 桶"""
    cfg = load_vertical_config()
    return list(cfg.get("why_you_buckets", WHY_YOU_BUCKETS))


def get_why_now_buckets() -> list[str]:
    """通用 Why now 桶"""
    cfg = load_vertical_config()
    return list(cfg.get("why_now_buckets", WHY_NOW_BUCKETS))


def get_why_you_phrases(vertical: str) -> dict[str, list[str]]:
    """获取 Why you 桶 -> 短语列表。key=桶名，value=该桶下的具体短语"""
    c = get_corpus(vertical)
    phrases = c.get("why_you_phrases") or {}
    result: dict[str, list[str]] = {}
    for bucket in WHY_YOU_BUCKETS:
        result[bucket] = list(phrases.get(bucket, []))
    return result


def get_why_now_phrases(vertical: str) -> dict[str, list[str]]:
    """获取 Why now 桶 -> 短语列表"""
    c = get_corpus(vertical)
    phrases = c.get("why_now_phrases") or {}
    result: dict[str, list[str]] = {}
    for bucket in WHY_NOW_BUCKETS:
        result[bucket] = list(phrases.get(bucket, []))
    return result


def get_sell_point_options(vertical: str) -> list[str]:
    """获取 sell_point 候选（Why you + Why now 组合展示用）"""
    c = get_corpus(vertical)
    return list(c.get("sell_point") or [])


def get_why_you_options(vertical: str) -> list[tuple[str, str]]:
    """兼容：返回 [(bucket, phrase), ...]，phrase 取该桶下首个或桶名"""
    phrases_map = get_why_you_phrases(vertical)
    result: list[tuple[str, str]] = []
    for bucket in WHY_YOU_BUCKETS:
        ph_list = phrases_map.get(bucket, [])
        phrase = ph_list[0] if ph_list else bucket
        result.append((bucket, phrase))
    return result if result else [("其他", "其他")]


def get_sample_strategy_card(vertical: str) -> dict[str, Any]:
    cfg = load_vertical_config()
    samples = cfg.get("sample_strategy_card", {})
    return dict(samples.get(_normalize_vertical(vertical), {}))


def get_root_cause_gap(vertical: str, index: int = 0) -> str:
    c = get_corpus(vertical)
    gaps = c.get("root_cause_gap", [])
    if isinstance(gaps, list) and gaps:
        return gaps[index % len(gaps)]
    return ""


def get_why_you_examples(vertical: str) -> dict[str, list[str]]:
    """兼容：按桶返回示例短语"""
    return get_why_you_phrases(vertical)


def get_why_now_pool(vertical: str) -> list[str]:
    """获取 Why now 短语候选池（扁平列表）"""
    phrases_map = get_why_now_phrases(vertical)
    result: list[str] = []
    for ph_list in phrases_map.values():
        result.extend(ph_list)
    return result if result else ["其他"]


def get_why_you_phrase_list(vertical: str) -> list[str]:
    """获取 Why you 短语候选池（扁平列表）"""
    phrases_map = get_why_you_phrases(vertical)
    result: list[str] = []
    for ph_list in phrases_map.values():
        result.extend(ph_list)
    return result if result else ["其他"]


def get_pool(vertical: str, key: str) -> list[str] | list[dict] | dict[str, list[str]]:
    c = get_corpus(vertical)
    return c.get(key) or []


def get_metric_weights(vertical: str, os: str = "") -> dict[str, float]:
    cfg = load_vertical_config()
    weights = cfg.get("metric_weights", {})
    w = dict(weights.get(_normalize_vertical(vertical), {"ipm": 0.4, "cpi": 0.35, "early_roas": 0.25}))
    w.setdefault("ctr", w.get("ipm", 0.4) * 0.5)
    return w


def get_why_now_strong_stimulus_penalty(vertical: str) -> float:
    cfg = load_vertical_config()
    rules = cfg.get("risk_rules", {})
    return float(rules.get(_normalize_vertical(vertical), {}).get("why_now_strong_stimulus_penalty", 3.0))


def get_why_now_strong_triggers(vertical: str) -> list[str]:
    cfg = load_vertical_config()
    rules = cfg.get("risk_rules", {})
    return list(rules.get(_normalize_vertical(vertical), {}).get("why_now_strong_triggers", []))


def use_refund_risk(vertical: str) -> bool:
    cfg = load_vertical_config()
    w = cfg.get("metric_weights", {}).get(_normalize_vertical(vertical), {})
    return bool(w.get("use_refund_risk", False))


def early_roas_as_proxy(vertical: str) -> bool:
    cfg = load_vertical_config()
    w = cfg.get("metric_weights", {}).get(_normalize_vertical(vertical), {})
    return bool(w.get("early_roas_as_proxy", True))
