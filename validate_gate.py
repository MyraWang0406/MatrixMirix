"""
Validate Gate 评测逻辑：基于 ≥2 时间窗口 + 轻扩人群的 metrics，
判断结构组合是否可加量、是否需止损。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


# -------- 输入结构 --------


class WindowMetrics(BaseModel):
    """单时间窗口的聚合 metrics"""

    window_id: str = Field(..., description="窗口标识，如 T1/T2")
    impressions: int = 0
    clicks: int = 0
    installs: int = 0
    spend: float = 0.0
    early_events: int = 0
    early_revenue: float = 0.0
    ipm: float = 0.0
    cpi: float = 0.0
    early_roas: float = 0.0


# -------- 配置 --------


@dataclass
class ValidateGateConfig:
    """Validate Gate 阈值配置"""

    ipm_cv_max: float = 0.35  # IPM 变异系数上限（超出=波动大）
    ipm_drop_max_pct: float = 0.30  # 窗口间 IPM 最大允许跌幅
    cpi_increase_max_pct: float = 0.25  # CPI 最大允许涨幅（回撤容忍）
    early_direction_min_corr: float = 0.0  # early_events 与 early_roas 方向一致性（简化：同向为 1）
    light_expansion_ipm_drop_max: float = 0.20  # 轻扩人群 IPM 相对主人群最大跌幅
    light_expansion_cpi_increase_max: float = 0.30  # 轻扩人群 CPI 相对主人群最大涨幅


# -------- 输出 --------


class ValidateDetailRow(BaseModel):
    """单行明细：window 或 expand_segment"""

    window_id: str = Field(..., description="window_1 / window_2 / expand_segment")
    ipm: float = 0.0
    cpi: float = 0.0
    early_roas: float = 0.0
    impressions: int = 0
    spend: float = 0.0


class ValidateStabilityMetrics(BaseModel):
    """稳定性指标"""

    ipm_cv: float = Field(default=0.0, description="IPM 变异系数（波动）")
    ipm_drop_pct: float = Field(default=0.0, description="IPM 回撤百分比")
    cpi_increase_pct: float = Field(default=0.0, description="CPI 涨幅百分比")
    learning_iterations: int = Field(default=0, description="学习反复次数（模拟）")


class ValidateGateResult(BaseModel):
    """Validate Gate 评测结果"""

    validate_status: str = Field(..., description="PASS / FAIL")
    risk_notes: list[str] = Field(default_factory=list, description="风险提示")
    scale_recommendation: dict[str, str] = Field(
        default_factory=dict,
        description="加量步长 / 止损线等建议",
    )
    detail_rows: list[ValidateDetailRow] = Field(
        default_factory=list,
        description="window_1 / window_2 / expand_segment 明细",
    )
    stability_metrics: ValidateStabilityMetrics = Field(
        default_factory=ValidateStabilityMetrics,
        description="波动、回撤、learning 反复次数",
    )


# -------- 评测逻辑 --------


def _parse_metrics(m: Any) -> WindowMetrics:
    if isinstance(m, dict):
        return WindowMetrics.model_validate(m)
    if hasattr(m, "model_dump"):
        return WindowMetrics.model_validate(m.model_dump())
    return m


def evaluate_validate_gate(
    windowed_metrics: list[WindowMetrics | dict],
    light_expansion_metrics: WindowMetrics | dict | None = None,
    *,
    config: ValidateGateConfig | None = None,
) -> ValidateGateResult:
    """
    Validate Gate 评测：基于多时间窗口 + 轻扩人群，判断是否可加量。

    输入：
    - windowed_metrics: ≥2 个时间窗口的 metrics，每个窗口一条
    - light_expansion_metrics: 轻扩人群 variant 的 metrics（可选）
    - config: 可配置阈值

    判断：
    1. IPM 波动是否在可接受范围
    2. CPI 是否回撤（涨幅超阈值）
    3. early_event / early_ROAS 是否方向一致
    4. 轻扩人群是否明显劣化（若有）
    """
    cfg = config or ValidateGateConfig()
    risk_notes: list[str] = []
    scale_up_pct = "20%"
    stop_loss_line = "CPI 较首窗口涨幅 >25% 或 IPM 跌幅 >30%"

    if len(windowed_metrics) < 2:
        return ValidateGateResult(
            validate_status="FAIL",
            risk_notes=["时间窗口不足，需 ≥2 个窗口方能验证稳定性"],
            scale_recommendation={
                "scale_up_step": "暂不加量",
                "stop_loss": "待补足窗口后再评估",
            },
        )

    windows = [_parse_metrics(w) for w in windowed_metrics]
    ipms = [w.ipm for w in windows if w.impressions > 0]
    cpis = [w.cpi for w in windows if w.installs > 0]
    early_events = [w.early_events for w in windows]
    early_roas_list = [w.early_roas for w in windows if w.spend > 0]

    if not ipms:
        return ValidateGateResult(
            validate_status="FAIL",
            risk_notes=["无有效 IPM 数据"],
            scale_recommendation={"scale_up_step": "-", "stop_loss": "-"},
        )

    mean_ipm = sum(ipms) / len(ipms)
    mean_cpi = sum(cpis) / len(cpis) if cpis else 0
    fail_count = 0

    # 1. IPM 波动
    ipm_cv = (sum((x - mean_ipm) ** 2 for x in ipms) ** 0.5 / mean_ipm) if mean_ipm else 0
    if ipm_cv > cfg.ipm_cv_max:
        risk_notes.append("IPM 波动过大，结构稳定性存疑")
        fail_count += 1

    ipm_first = ipms[0]
    ipm_drop = (ipm_first - min(ipms)) / ipm_first if ipm_first else 0
    if ipm_drop > cfg.ipm_drop_max_pct:
        risk_notes.append("IPM 回撤超出可接受范围，可能 Hook 依赖强刺激")
        fail_count += 1

    # 2. CPI 回撤
    cpi_first = cpis[0] if cpis else 0
    cpi_max = max(cpis) if cpis else 0
    cpi_increase = (cpi_max - cpi_first) / cpi_first if cpi_first else 0
    if cpi_increase > cfg.cpi_increase_max_pct:
        risk_notes.append("CPI 回撤，成本抬升明显")
        fail_count += 1

    # 3. early_event / early_ROAS 方向一致
    if len(early_events) >= 2 and len(early_roas_list) >= 2:
        ev_dirs = [1 if early_events[i] >= early_events[i - 1] else -1 for i in range(1, len(early_events))]
        roas_dirs = [1 if early_roas_list[i] >= early_roas_list[i - 1] else -1 for i in range(1, len(early_roas_list))]
        matches = sum(1 for a, b in zip(ev_dirs, roas_dirs) if a == b)
        total = min(len(ev_dirs), len(roas_dirs))
        if total > 0 and matches / total < 0.5:
            risk_notes.append("early_event 与 early_ROAS 方向不一致，转化质量存疑")
            fail_count += 1

    # 4. 轻扩人群
    if light_expansion_metrics:
        le = _parse_metrics(light_expansion_metrics)
        if mean_ipm and le.ipm > 0:
            le_ipm_drop = (mean_ipm - le.ipm) / mean_ipm
            if le_ipm_drop > cfg.light_expansion_ipm_drop_max:
                risk_notes.append("轻扩人群 IPM 明显劣化，Why now 可能虚高")
                fail_count += 1
        if mean_cpi and le.cpi > 0:
            le_cpi_inc = (le.cpi - mean_cpi) / mean_cpi
            if le_cpi_inc > cfg.light_expansion_cpi_increase_max:
                risk_notes.append("轻扩人群 CPI 抬升过大")
                fail_count += 1

    # 风险备注生成规则补充
    if ipm_cv > 0.4:
        risk_notes.append("IPM 波动过大，建议延长观察窗口")
    if ipm_drop > 0.25 and "Hook" not in " ".join(risk_notes):
        risk_notes.append("IPM 回撤明显，可能 Hook 依赖强刺激")
    if cpi_increase > 0.2:
        risk_notes.append("CPI 抬升过快，需关注人群质量")

    # 模拟 learning_iterations：基于波动与回撤
    learning_iterations = 0
    if ipm_cv > 0.3:
        learning_iterations += 1
    if ipm_drop > 0.2:
        learning_iterations += 1
    if cpi_increase > 0.15:
        learning_iterations += 1
    if light_expansion_metrics:
        le = _parse_metrics(light_expansion_metrics)
        if mean_ipm and le.ipm < mean_ipm * 0.85:
            learning_iterations += 1

    # 明细行
    detail_rows = []
    for w in windows:
        detail_rows.append(
            ValidateDetailRow(
                window_id=w.window_id,
                ipm=round(w.ipm, 2),
                cpi=round(w.cpi, 2),
                early_roas=round(w.early_roas, 4),
                impressions=w.impressions,
                spend=w.spend,
            )
        )
    if light_expansion_metrics:
        le = _parse_metrics(light_expansion_metrics)
        detail_rows.append(
            ValidateDetailRow(
                window_id="expand_segment",
                ipm=round(le.ipm, 2),
                cpi=round(le.cpi, 2),
                early_roas=round(le.early_roas, 4),
                impressions=le.impressions,
                spend=le.spend,
            )
        )

    stability_metrics = ValidateStabilityMetrics(
        ipm_cv=round(ipm_cv, 4),
        ipm_drop_pct=round(ipm_drop * 100, 2),
        cpi_increase_pct=round(cpi_increase * 100, 2),
        learning_iterations=min(5, learning_iterations),
    )

    # 汇总
    validate_status = "PASS" if fail_count == 0 else "FAIL"

    if validate_status == "PASS":
        scale_up_pct = "20%"
        stop_loss_line = f"CPI 较首窗口涨幅 >{int(cfg.cpi_increase_max_pct*100)}% 或 IPM 跌幅 >{int(cfg.ipm_drop_max_pct*100)}%"
    else:
        scale_up_pct = "10%" if fail_count <= 1 else "暂不加量"
        stop_loss_line = "收紧止损：CPI +15% 或 IPM -20% 即停"

    return ValidateGateResult(
        validate_status=validate_status,
        risk_notes=risk_notes if risk_notes else ["无显著风险"],
        scale_recommendation={
            "scale_up_step": f"建议加量步长 {scale_up_pct}",
            "stop_loss": stop_loss_line,
        },
        detail_rows=detail_rows,
        stability_metrics=stability_metrics,
    )
