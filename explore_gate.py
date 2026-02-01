"""
Explore Gate 评测逻辑：基于模拟投放数据，判断变体是否进入验证期。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from simulate_metrics import SimulatedMetrics


# -------- 配置 --------


@dataclass
class ExploreGateConfig:
    """Gate 规则配置（可配）"""

    min_spend: float = 500
    min_better_metrics: int = 2
    improvement_pct: float = 0.0
    proxy_metrics: tuple[str, ...] = ("ctr", "ipm", "cpi")


# -------- 输出 --------


class ExploreGateResult(BaseModel):
    """Explore Gate 评测结果"""

    gate_status: str = Field(
        ...,
        description="PASS / FAIL / INSUFFICIENT / INVALID",
    )
    reasons: list[str] = Field(default_factory=list, description="原因说明（中文）")
    eligible_variants: list[str] = Field(
        default_factory=list,
        description="进入验证期的 variant_id 列表",
    )
    variant_details: dict[str, str] = Field(
        default_factory=dict,
        description="每个 variant 的 gate 状态（PASS/FAIL/INSUFFICIENT/INVALID）",
    )
    context: dict[str, Any] = Field(default_factory=dict, description="评测上下文")


# -------- 评测逻辑 --------


def _get_metrics_by_os(
    metrics_list: list[SimulatedMetrics | dict],
) -> dict[str, SimulatedMetrics]:
    """按 os 索引 metrics（用于取 baseline）"""
    result: dict[str, SimulatedMetrics] = {}
    for m in metrics_list:
        obj = SimulatedMetrics.model_validate(m) if isinstance(m, dict) else m
        result[obj.os] = obj
    return result


def _bucket_key(b: dict[str, Any] | None) -> tuple:
    """用于 bucket 一致性比较。支持 why_you_key（稳定）或 why_you_bucket/why_you_label（兼容）"""
    if not b:
        return ()
    wy = b.get("why_you_key") or b.get("why_you_bucket") or b.get("why_you_label", "")
    return (
        b.get("motivation_bucket", ""),
        wy,
        b.get("why_now_trigger", ""),
    )


def _count_better(
    variant: SimulatedMetrics,
    baseline: SimulatedMetrics,
    improvement_pct: float,
) -> tuple[int, list[str]]:
    """
    探索期代理指标：CTR、IPM、CPI。
    CTR/IPM 越高越好，CPI 越低越好。
    返回 (优于 baseline 的指标数, 每个指标的胜出说明)
    """
    better_count = 0
    details: list[str] = []

    # CTR: 越高越好
    if improvement_pct <= 0:
        ctr_win = variant.ctr > baseline.ctr
    else:
        ctr_win = variant.ctr >= baseline.ctr * (1 + improvement_pct / 100)
    if ctr_win:
        better_count += 1
        details.append(f"CTR {variant.ctr:.4%} > baseline {baseline.ctr:.4%}")
    else:
        details.append(f"CTR {variant.ctr:.4%} ≤ baseline {baseline.ctr:.4%}")

    # IPM: 越高越好
    if improvement_pct <= 0:
        ipm_win = variant.ipm > baseline.ipm
    else:
        ipm_win = variant.ipm >= baseline.ipm * (1 + improvement_pct / 100)
    if ipm_win:
        better_count += 1
        details.append(f"IPM {variant.ipm:.2f} > baseline {baseline.ipm:.2f}")
    else:
        details.append(f"IPM {variant.ipm:.2f} ≤ baseline {baseline.ipm:.2f}")

    # CPI: 越低越好
    if improvement_pct <= 0:
        cpi_win = variant.cpi < baseline.cpi
    else:
        cpi_win = variant.cpi <= baseline.cpi * (1 - improvement_pct / 100)
    if cpi_win:
        better_count += 1
        details.append(f"CPI {variant.cpi:.2f} < baseline {baseline.cpi:.2f}")
    else:
        details.append(f"CPI {variant.cpi:.2f} ≥ baseline {baseline.cpi:.2f}")

    return better_count, details


def evaluate_explore_gate(
    variant_metrics: list[SimulatedMetrics | dict],
    baseline_metrics: SimulatedMetrics | dict | list[SimulatedMetrics | dict],
    context: dict[str, Any],
    *,
    config: ExploreGateConfig | None = None,
    bucket_info: dict[str, dict[str, Any]] | None = None,
) -> ExploreGateResult:
    """
    Explore Gate 评测：判断变体是否进入验证期。

    输入：
    - variant_metrics: 待评测的变体指标列表（不含 baseline）
    - baseline_metrics: baseline 指标；若为 list，则按 os 取对应 baseline
    - context: {country, os, objective, segment}，用于筛选与说明
    - config: 可配置阈值，默认 min_spend=500, min_better_metrics=2
    - bucket_info: 可选，variant_id -> {motivation_bucket, why_you_bucket, why_now_trigger}
                   若提供且与 baseline 不一致则 INVALID

    Gate 规则：
    1. bucket 必须一致（否则 INVALID）
    2. spend < 最小预算门槛 → INSUFFICIENT
    3. 探索期只用代理指标：CTR / IPM / CPI
    4. Variant 必须在 ≥ min_better_metrics 个指标上优于 baseline
    """
    cfg = config or ExploreGateConfig()
    target_os = context.get("os", "")
    reasons: list[str] = []
    eligible: list[str] = []
    variant_details: dict[str, str] = {}

    # 1. 解析 baseline
    if isinstance(baseline_metrics, list):
        baseline_by_os = _get_metrics_by_os(baseline_metrics)
        baseline = baseline_by_os.get(target_os) if target_os else None
    else:
        bl = (
            SimulatedMetrics.model_validate(baseline_metrics)
            if isinstance(baseline_metrics, dict)
            else baseline_metrics
        )
        baseline = bl if (not target_os or bl.os == target_os) else None

    if not baseline:
        return ExploreGateResult(
            gate_status="INVALID",
            reasons=["缺少 baseline 数据或 baseline 与 context.os 不匹配"],
            eligible_variants=[],
            variant_details={},
            context=context,
        )

    baseline_bucket = _bucket_key(bucket_info.get("__baseline__", {}) if bucket_info else None)

    # 2. 筛选本 os 的 variant
    variants_for_os = []
    for m in variant_metrics:
        obj = SimulatedMetrics.model_validate(m) if isinstance(m, dict) else m
        if obj.baseline:
            continue
        if target_os and obj.os != target_os:
            continue
        variants_for_os.append(obj)

    if not variants_for_os:
        return ExploreGateResult(
            gate_status="FAIL",
            reasons=["无待评测变体或变体与 context.os 不匹配"],
            eligible_variants=[],
            variant_details={},
            context=context,
        )

    # 3. 逐变体判断
    for v in variants_for_os:
        vid = v.variant_id

        # 3a. bucket 一致性（若提供了 baseline 与 variant 的 bucket）
        if bucket_info and "__baseline__" in bucket_info and baseline_bucket:
            vb = _bucket_key(bucket_info.get(vid))
            if vb and vb != baseline_bucket:
                variant_details[vid] = "INVALID"
                reasons.append(f"{vid}: bucket 与 baseline 不一致")
                continue

        # 3b. 预算门槛
        if v.spend < cfg.min_spend:
            variant_details[vid] = "INSUFFICIENT"
            reasons.append(f"{vid}: spend={v.spend:.0f} < 最小预算门槛 {cfg.min_spend}")
            continue

        # 3c. 代理指标优于 baseline
        better_count, _ = _count_better(v, baseline, cfg.improvement_pct)
        if better_count >= cfg.min_better_metrics:
            variant_details[vid] = "PASS"
            eligible.append(vid)
            reasons.append(f"{vid}: 在 {better_count} 个指标上优于 baseline，通过")
        else:
            variant_details[vid] = "FAIL"
            reasons.append(f"{vid}: 仅 {better_count} 个指标优于 baseline，需 ≥{cfg.min_better_metrics}")

    # 4. 汇总 gate_status
    if eligible:
        gate_status = "PASS"
    elif any(v == "INSUFFICIENT" for v in variant_details.values()):
        gate_status = "INSUFFICIENT"
    elif any(v == "INVALID" for v in variant_details.values()):
        gate_status = "INVALID"
    else:
        gate_status = "FAIL"

    # 5. reasons 必须引用 motivation_bucket（解释门禁合理性）
    mb = context.get("motivation_bucket", "")
    if mb:
        if mb == "省钱":
            gate_reason = f"【motivation_bucket={mb}】省钱桶对 CTR 更敏感，门禁侧重点击转化；当前 {gate_status}。"
        elif mb == "体验":
            gate_reason = f"【motivation_bucket={mb}】体验桶对 early_roas 更敏感，通过变体需验证转化质量；当前 {gate_status}。"
        elif mb in ("胜负欲", "成就感", "爽感"):
            gate_reason = f"【motivation_bucket={mb}】胜负欲/成就感/爽感桶关注 IPM 与 CPI 平衡；当前 {gate_status}。"
        else:
            gate_reason = f"【motivation_bucket={mb}】当前 {gate_status}，符合该动机桶评测口径。"
        reasons = [gate_reason] + reasons

    return ExploreGateResult(
        gate_status=gate_status,
        reasons=reasons,
        eligible_variants=eligible,
        variant_details=variant_details,
        context=context,
    )
