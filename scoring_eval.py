"""
评测评分体系：variant_score、element_score 标准化、card_score。
基于 Explore 指标加权，按 vertical 配置。
"""
from __future__ import annotations

from typing import Any

from simulate_metrics import SimulatedMetrics
from vertical_config import get_metric_weights, use_refund_risk


def _get_weights(os: str, vertical: str) -> dict[str, float]:
    """从 vertical_config 取权重，兼容 ctr、refund_risk"""
    w = get_metric_weights(vertical, os)
    # 若缺少某项，补默认
    for k in ("ipm", "cpi", "early_roas", "ctr"):
        if k not in w:
            w = dict(w)
            w.setdefault("ipm", 0.4)
            w.setdefault("cpi", 0.35)
            w.setdefault("early_roas", 0.25)
            w.setdefault("ctr", 0.0)
    return w


def compute_variant_score(
    metric: SimulatedMetrics | dict,
    cohort: list[SimulatedMetrics | dict],
    *,
    os: str = "",
    vertical: str = "casual_game",
    weights: dict[str, float] | None = None,
) -> float:
    """
    基于 Explore 指标（IPM↑、CPI↓、early_roas↑）加权计算 variant_score。

    输入：
    - metric: 单条指标
    - cohort: 同 OS 的全体指标（用于 min-max 归一化）
    - os, vertical: 用于选取权重
    - weights: 可覆盖，{ipm, cpi, early_roas} 权重

    输出：0~100 分
    """
    m = SimulatedMetrics.model_validate(metric) if isinstance(metric, dict) else metric
    cohort_parsed = [
        SimulatedMetrics.model_validate(x) if isinstance(x, dict) else x for x in cohort
    ]
    same_os = [x for x in cohort_parsed if (os and x.os == os) or (not os and x.os == m.os)]
    if not same_os:
        same_os = cohort_parsed

    ipms = [x.ipm for x in same_os]
    cpis = [x.cpi for x in same_os]
    roas_list = [x.early_roas for x in same_os]
    ctrs = [x.ctr for x in same_os]

    w = weights or _get_weights(m.os, vertical)
    min_ipm, max_ipm = min(ipms), max(ipms)
    min_cpi, max_cpi = min(cpis), max(cpis)
    min_roas, max_roas = min(roas_list), max(roas_list)
    min_ctr, max_ctr = min(ctrs), max(ctrs)

    def _norm_high(val: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 50.0
        return 100.0 * (val - lo) / (hi - lo)

    def _norm_low(val: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 50.0
        return 100.0 * (hi - val) / (hi - lo)

    norm_ipm = _norm_high(m.ipm, min_ipm, max_ipm)
    norm_cpi = _norm_low(m.cpi, min_cpi, max_cpi)
    norm_roas = _norm_high(m.early_roas, min_roas, max_roas)
    norm_ctr = _norm_high(m.ctr, min_ctr, max_ctr) if w.get("ctr", 0) > 0 else 0.0

    score = (
        w.get("ipm", 0.4) * norm_ipm
        + w.get("cpi", 0.35) * norm_cpi
        + w.get("early_roas", 0.25) * norm_roas
        + w.get("ctr", 0) * norm_ctr
    )
    # 电商：退款风险扣分
    if use_refund_risk(vertical):
        rr = getattr(m, "refund_risk", 0) or 0
        score -= rr * 15
    return round(min(100.0, max(0.0, score)), 1)


def compute_element_normalized_score(
    ipm_delta: float,
    cpi_delta: float,
    *,
    ipm_scale: float = 8.0,
    cpi_scale: float = 1.5,
) -> float:
    """
    将 element 的 IPM/CPI delta 转为标准化分 -100~100。

    IPM 增量↑ 为正贡献，CPI 增量↑ 为负贡献。
    简单线性：score = ipm_delta/ipm_scale*50 - cpi_delta/cpi_scale*50，截断到 -100~100。
    """
    contrib = (ipm_delta / ipm_scale) * 50 - (cpi_delta / cpi_scale) * 50
    return round(min(100.0, max(-100.0, contrib)), 1)


def compute_card_score(
    eligible_variants: list[str],
    variant_scores: dict[str, float],
    *,
    top_k: int = 5,
    stability_penalty: float = 0.0,
    why_now_strong_stimulus_penalty: float = 0.0,
) -> dict[str, Any]:
    """
    卡片总分：取通过门禁的 topK variants 的 variant_score 均值 + 稳定性惩罚。

    输入：
    - eligible_variants: 通过 Explore Gate 的 variant_id 列表
    - variant_scores: variant_id -> 分数（若同一 variant 多 OS，取均值或最大值，由调用方聚合）
    - top_k: 取前 K 个
    - stability_penalty: 稳定性扣分（如 validate 不稳定）
    - why_now_strong_stimulus_penalty: why_now 强刺激风险扣分

    输出：{card_score, top_variants, penalty_breakdown}
    """
    scores_for_eligible = [
        (vid, variant_scores.get(vid, 0.0))
        for vid in eligible_variants
        if vid in variant_scores
    ]
    if not scores_for_eligible:
        return {
            "card_score": 0.0,
            "top_variants": [],
            "penalty_breakdown": {
                "stability_penalty": stability_penalty,
                "why_now_penalty": why_now_strong_stimulus_penalty,
                "base_mean": 0.0,
            },
        }

    scores_for_eligible.sort(key=lambda x: x[1], reverse=True)
    top = scores_for_eligible[:top_k]
    base_mean = sum(s for _, s in top) / len(top) if top else 0.0
    total_penalty = stability_penalty + why_now_strong_stimulus_penalty
    card_score = max(0.0, min(100.0, base_mean - total_penalty))

    return {
        "card_score": round(card_score, 1),
        "top_variants": [vid for vid, _ in top],
        "penalty_breakdown": {
            "stability_penalty": stability_penalty,
            "why_now_penalty": why_now_strong_stimulus_penalty,
            "base_mean": round(base_mean, 1),
        },
    }
