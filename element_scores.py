"""
元素级贡献分析：基于多 Variant 的 metrics + ElementTag 拆解，
计算「是否包含某元素」时的 IPM/CPI 均值差。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from eval_schemas import ElementTag, Variant, decompose_variant_to_element_tags
from simulate_metrics import SimulatedMetrics
from scoring_eval import compute_element_normalized_score

ConfidenceLevel = Literal["low", "medium", "high"]
CrossOSConsistency = Literal["pos", "neg", "mixed"]


# -------- 输出模型 --------


class ElementScore(BaseModel):
    """元素贡献得分"""

    element_type: str = Field(..., description="hook / why_you / why_now / cta / asset")
    element_value: str = Field(..., description="元素取值")
    avg_IPM_delta_vs_card_mean: float = Field(
        ...,
        description="含该元素的变体 IPM 均值 - 卡片整体 IPM 均值",
    )
    avg_CPI_delta_vs_card_mean: float = Field(
        ...,
        description="含该元素的变体 CPI 均值 - 卡片整体 CPI 均值（正=更贵）",
    )
    sample_size: int = Field(..., description="含该元素的 (variant_id, os) 样本数")
    stability_flag: bool = Field(
        ...,
        description="样本是否足够（≥ min_sample_size）",
    )
    normalized_score: float = Field(
        default=0.0,
        description="标准化分 -100~100（拉为正，拖为负）",
    )
    confidence_level: ConfidenceLevel = Field(
        default="low",
        description="low: 样本不足 | medium: 倾向 | high: 稳定结论",
    )
    cross_os_consistency: CrossOSConsistency = Field(
        default="mixed",
        description="pos: 双端一致拉 | neg: 双端一致拖 | mixed: 双端不一致（mixed 降一级）",
    )


# -------- 贡献分析 --------


def compute_element_scores(
    variant_metrics: list[SimulatedMetrics | dict],
    variant_to_tags: dict[str, list[ElementTag]] | None = None,
    variants: list[Variant] | None = None,
    *,
    parent_card_id: str | None = None,
    min_sample_size: int = 2,
) -> list[ElementScore]:
    """
    元素级贡献分析：对同一 StrategyCard 下的 Variant，
    比较「是否包含某 ElementTag」时 IPM/CPI 的均值差。

    输入：
    - variant_metrics: 多个 Variant 的 metrics（含 variant_id, os, ipm, cpi）
    - variant_to_tags: 可选，variant_id -> ElementTag 列表；若不提供则用 variants + decompose
    - variants: 可选，Variant 列表，用于自动拆解 ElementTag
    - parent_card_id: 可选，仅分析该 card 下的变体
    - min_sample_size: 样本足够阈值，默认 2

    方法：
    - 卡片整体均值 = 所有 metrics 的 IPM/CPI 均值
    - 对每个 (element_type, element_value)，取含该元素的变体的 metrics
    - avg_IPM_delta = 含元素组的 IPM 均值 - 卡片 IPM 均值
    - avg_CPI_delta = 含元素组的 CPI 均值 - 卡片 CPI 均值
    """
    metrics_list = [
        SimulatedMetrics.model_validate(m) if isinstance(m, dict) else m
        for m in variant_metrics
    ]

    # 1. 构建 variant_id -> tags
    if variant_to_tags is None and variants:
        variant_to_tags = {}
        for v in variants:
            variant_to_tags[v.variant_id] = decompose_variant_to_element_tags(v)
    if not variant_to_tags:
        return []

    # 2. 筛选本 card 的 metrics
    variant_ids_in_card = set(variant_to_tags.keys())
    if parent_card_id:
        pass  # 若 variants 有 parent_card_id 可过滤，此处简化：用 variant_to_tags 的 key
    metrics_in_scope = [m for m in metrics_list if m.variant_id in variant_ids_in_card]
    if not metrics_in_scope:
        return []

    # 3. 卡片整体均值
    card_mean_ipm = sum(m.ipm for m in metrics_in_scope) / len(metrics_in_scope)
    card_mean_cpi = sum(m.cpi for m in metrics_in_scope) / len(metrics_in_scope)

    # 4. 每个 (element_type, element_value) 对应的 metrics 行（含 os）
    # key: (element_type, element_value), value: list of (ipm, cpi, os)
    element_metrics: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)

    for m in metrics_in_scope:
        tags = variant_to_tags.get(m.variant_id, [])
        seen = set()
        for t in tags:
            key = (t.element_type, t.element_value)
            if key not in seen:
                seen.add(key)
                element_metrics[key].append((m.ipm, m.cpi, m.os))

    def _cross_os_consistency(
        ipm_cpi_os_list: list[tuple[float, float, str]],
        card_ipm: float,
        card_cpi: float,
    ) -> CrossOSConsistency:
        """pos: 双端一致拉 | neg: 双端一致拖 | mixed: 双端不一致"""
        by_os: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for ipm, cpi, os_ in ipm_cpi_os_list:
            by_os[os_].append((ipm, cpi))
        if len(by_os) < 2:
            return "mixed"
        os_deltas: list[tuple[float, float]] = []
        for os_, rows in by_os.items():
            mean_ipm = sum(x[0] for x in rows) / len(rows)
            mean_cpi = sum(x[1] for x in rows) / len(rows)
            os_deltas.append((mean_ipm - card_ipm, mean_cpi - card_cpi))
        # 方向：拉 = IPMΔ>0 或 CPIΔ<0；拖 = IPMΔ<0 或 CPIΔ>0
        dirs = []
        for ipm_d, cpi_d in os_deltas:
            if ipm_d > 0 or cpi_d < 0:
                dirs.append(1)  # 拉
            elif ipm_d < 0 or cpi_d > 0:
                dirs.append(-1)  # 拖
            else:
                dirs.append(0)
        if len(set(dirs)) > 1:
            return "mixed"
        if dirs[0] == 1:
            return "pos"
        return "neg"

    def _confidence_level(n: int, cross_os: CrossOSConsistency) -> ConfidenceLevel:
        if n < 6:
            base = "low"
        elif n <= 15:
            base = "medium"
        else:
            base = "high"
        if cross_os == "mixed":
            if base == "high":
                return "medium"
            if base == "medium":
                return "low"
        return base

    # 5. 计算 ElementScore
    results: list[ElementScore] = []
    for (et, ev), ipm_cpi_os_list in element_metrics.items():
        n = len(ipm_cpi_os_list)
        mean_ipm = sum(x[0] for x in ipm_cpi_os_list) / n
        mean_cpi = sum(x[1] for x in ipm_cpi_os_list) / n
        ipm_delta = mean_ipm - card_mean_ipm
        cpi_delta = mean_cpi - card_mean_cpi

        cross_os = _cross_os_consistency(ipm_cpi_os_list, card_mean_ipm, card_mean_cpi)
        conf = _confidence_level(n, cross_os)

        ns = compute_element_normalized_score(ipm_delta, cpi_delta)
        results.append(
            ElementScore(
                element_type=et,
                element_value=ev,
                avg_IPM_delta_vs_card_mean=round(ipm_delta, 4),
                avg_CPI_delta_vs_card_mean=round(cpi_delta, 4),
                sample_size=n,
                stability_flag=n >= min_sample_size,
                normalized_score=ns,
                confidence_level=conf,
                cross_os_consistency=cross_os,
            )
        )

    return results
