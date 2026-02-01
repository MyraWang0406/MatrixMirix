"""熔断(fuse) + 白量风险：二次校验（不完全信任模型）"""
from __future__ import annotations

from schemas import CreativeCard, CreativeVariant, ReviewResult, Verdict

# 敏感词/夸大词词表（严重 => 直接 RED）
EXAGGERATION_WORDS_SEVERE = [
    "永久免费", "暴富", "稳赚", "稳赚不赔", "100%", "百分百", "绝对", "包赚", "躺赚",
    "秒变", "秒到", "一夜", "必中", "必上王者", "guaranteed", "guarantee",
]

# 一般夸大词（no_exaggeration 时至少 YELLOW）
EXAGGERATION_WORDS_NORMAL = [
    "免费领", "零成本", "无风险", "稳赢", "必涨", "绝对有效",
    "立竿见影", "根治", "特效", "神效", "顶级", "第一",
]


def _collect_variant_text(variant: CreativeVariant) -> str:
    """收集变体所有可扫描文本"""
    parts = [
        getattr(variant, "hook_type", "") or "",
        getattr(variant, "notes", "") or "",
        getattr(variant, "cta", "") or variant.cta_text or "",
    ]
    w = getattr(variant, "who_why_now", None)
    if w:
        parts.extend([getattr(w, "who", ""), getattr(w, "why", ""), getattr(w, "why_now", "")])
    script = getattr(variant, "script", None)
    if script and getattr(script, "shots", None):
        for s in script.shots:
            parts.extend([
                getattr(s, "visual", ""),
                getattr(s, "overlay_text", ""),
                getattr(s, "voiceover", ""),
            ])
    parts.append(getattr(variant, "headline", "") or variant.title or "")
    parts.append(getattr(variant, "core_message", "") or variant.script_15s or "")
    return " ".join(str(p) for p in parts if p)


def _scan_exaggeration(text: str) -> tuple[bool, bool]:
    """返回 (命中严重词, 命中一般词)"""
    t = text.lower()
    hit_severe = any(w.lower() in t for w in EXAGGERATION_WORDS_SEVERE)
    hit_normal = any(w.lower() in t for w in EXAGGERATION_WORDS_NORMAL)
    return hit_severe, hit_normal


def _white_traffic_risk_rule_based(
    variant: CreativeVariant, review: ReviewResult
) -> int:
    """规则计算的白量风险 0-100"""
    s = review.scores
    risk = 0

    # 新格式 0-100：低分=高风险
    risk += (100 - s.clarity) * 0.1
    risk += (100 - s.compliance_safety) * 0.25
    risk += (100 - s.expected_test_value) * 0.15

    return min(100, max(0, int(risk)))


def compute_fuse_decision(
    card: CreativeCard,
    variant: CreativeVariant,
    review: ReviewResult,
) -> tuple[Verdict, int, str]:
    """
    二次校验：熔断 + 白量风险，输出最终 decision。
    返回 (verdict, white_traffic_risk_final, fuse_level)
    """
    if review.error:
        return "KILL", 100, "RED"

    s = review.scores
    text = _collect_variant_text(variant)
    hit_severe, hit_normal = _scan_exaggeration(text)
    no_exag = getattr(card, "no_exaggeration", True)

    fuse_level = "GREEN"

    # 1. 敏感词：严重词 => RED；一般词 + no_exaggeration => 至少 YELLOW
    if hit_severe and no_exag:
        fuse_level = "RED"
    elif hit_normal and no_exag:
        if fuse_level == "GREEN":
            fuse_level = "YELLOW"

    # 2. 评分规则（0-100 格式）
    if s.compliance_safety < 40:
        fuse_level = "RED"
    elif s.clarity < 40 or s.expected_test_value < 40:
        fuse_level = "YELLOW" if fuse_level != "RED" else fuse_level

    # 3. 白量风险：模型 white_traffic_risk_final 为 low|medium|high
    rule_risk = _white_traffic_risk_rule_based(variant, review)
    wt_str = getattr(review, "white_traffic_risk_final", "low") or "low"
    model_risk = {"low": 20, "medium": 55, "high": 90}.get(str(wt_str).lower(), 20)
    white_traffic_risk_final = max(rule_risk, model_risk)

    # 4. 白量风险映射 fuse_level（取更严，不降级）
    fuse_from_risk = "GREEN"
    if white_traffic_risk_final >= 70:
        fuse_from_risk = "RED"
    elif white_traffic_risk_final >= 40:
        fuse_from_risk = "YELLOW"
    # 合并：RED > YELLOW > GREEN
    severity = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    fuse_level = max(
        [fuse_level, fuse_from_risk],
        key=lambda x: severity.get(x, 0),
    )

    # 5. 最终 decision（优先规则熔断；否则用模型 decision：HARD_FAIL/KILL->KILL, SOFT_FAIL/REVISE->REVISE）
    model_decision = (getattr(review, "decision", "") or "").upper()
    if fuse_level == "RED":
        verdict = "KILL"
    elif fuse_level == "YELLOW":
        verdict = "REVISE"
    else:
        if model_decision in ("HARD_FAIL", "KILL"):
            verdict = "KILL"
        elif model_decision in ("SOFT_FAIL", "REVISE"):
            verdict = "REVISE"
        else:
            verdict = "PASS"

    return verdict, white_traffic_risk_final, fuse_level
