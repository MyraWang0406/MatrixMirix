"""
TikTok 投放评测原型：模拟投放数据生成器
不调用任何外部 API，基于确定性伪随机生成合理波动数据。
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Literal

from pydantic import BaseModel, Field


OS = Literal["iOS", "Android"]


# -------- 输出模型 --------


class SimulatedMetrics(BaseModel):
    """单个 Variant 的模拟投放指标"""

    variant_id: str = Field(..., description="变体 ID")
    os: str = Field(..., description="iOS / Android")
    baseline: bool = Field(default=False, description="是否为基线变体")

    # 原始指标
    impressions: int = Field(..., description="曝光量")
    clicks: int = Field(..., description="点击量")
    installs: int = Field(..., description="安装量")
    spend: float = Field(..., description="花费（USD）")
    early_events: int = Field(..., description="早期事件数，如 D1_event_count")
    early_revenue: float = Field(..., description="早期收入（USD）")

    # 派生指标
    ctr: float = Field(..., description="点击率 = clicks/impressions")
    ipm: float = Field(..., description="千次曝光安装数 = installs/impressions*1000")
    cpi: float = Field(..., description="单次安装成本 = spend/installs")
    early_roas: float = Field(..., description="早期 ROAS 代理 = early_revenue/spend")

    # 电商专属：退款风险（模拟，0-1）
    refund_risk: float = Field(default=0.0, description="退款风险，ecommerce 时模拟")
    # 电商专属：转化代理（purchase/value 目标用）
    conversion_proxy: float = Field(default=0.0, description="转化率代理，ecommerce 时模拟")
    order_proxy: float = Field(default=0.0, description="下单率代理，ecommerce 时模拟")


# -------- 参数（TikTok 投放典型范围）--------

_CTR_RANGE = (0.005, 0.025)  # 0.5% - 2.5%
_IPM_RANGE = (8, 45)  # 千次曝光安装
_CPI_RANGE_IOS = (2.5, 8.0)
_CPI_RANGE_ANDROID = (1.2, 4.5)
_EVENTS_PER_INSTALL = (0.25, 0.65)
_EARLY_ROAS_RANGE = (0.0, 0.25)
_IMPRESSIONS_BASE = 50_000
_IMPRESSIONS_VARIANCE = 0.4  # ±40%


def _seeded_random(seed_str: str) -> random.Random:
    """基于字符串生成确定性随机数生成器"""
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    seed = int(h[:16], 16) % (2**32)
    return random.Random(seed)


def _variant_quality(variant_id: str) -> float:
    """根据 variant_id 得出 0.85-1.15 的质量系数（确定性）"""
    h = hashlib.sha256(f"quality_{variant_id}".encode()).hexdigest()
    v = int(h[:8], 16) / (16**8)
    return 0.85 + v * 0.3


def _motivation_bucket_factors(
    motivation_bucket: str,
    vertical: str,
) -> tuple[float, float, float]:
    """
    motivation_bucket 对 CTR/IPM、CPI、early_roas 的影响因子。
    省钱桶：CTR 更敏感（+）；体验桶：early_roas 更敏感（+）；按 vertical 微调。
    返回 (ctr_ipm_factor, cpi_factor, roas_factor)，均为 ~0.9-1.15
    """
    base = 1.0
    ctr_ipm, cpi, roas = base, base, base
    mb = (motivation_bucket or "").strip() or "其他"
    v = (vertical or "casual_game").lower()

    if mb == "省钱":
        ctr_ipm = 1.12  # CTR 更敏感
        cpi = 0.95
        roas = 0.92
    elif mb == "体验":
        ctr_ipm = 0.96
        cpi = 1.05
        roas = 1.15  # early_roas 更敏感
    elif mb == "社交":
        ctr_ipm = 1.05
        roas = 1.08
    elif mb in ("胜负欲", "成就感", "爽感"):
        ctr_ipm = 1.08
        roas = 1.02
    elif mb == "收集":
        ctr_ipm = 1.03
        roas = 0.98

    if v == "ecommerce" and mb == "省钱":
        ctr_ipm *= 1.05
        roas *= 0.95
    elif v in ("game", "casual_game") and mb in ("胜负欲", "成就感", "爽感"):
        ctr_ipm *= 1.03
        roas *= 1.02

    return (ctr_ipm, cpi, roas)


def _sell_point_factor(sell_point: str) -> float:
    """sell_point 对指标的影响因子，0.90-1.10（确定性）"""
    if not sell_point or not sell_point.strip():
        return 1.0
    h = hashlib.sha256(f"sell_point_{sell_point}".encode()).hexdigest()
    v = int(h[:8], 16) / (16**8)
    return 0.90 + v * 0.20


def _add_noise(value: float, noise_pct: float, rng: random.Random) -> float:
    """添加 ±noise_pct 的波动"""
    delta = value * noise_pct * (2 * rng.random() - 1)
    return max(value * 0.5, value + delta)


def simulate_metrics(
    variant: Any,
    os: OS,
    *,
    baseline: bool = False,
    motivation_bucket: str = "",
    vertical: str = "casual_game",
    objective: str = "",
) -> SimulatedMetrics:
    """
    为单个 Variant 模拟 TikTok 投放指标。

    - variant: 需有 variant_id 属性（如 eval_schemas.Variant）
    - os: "iOS" 或 "Android"
    - baseline: True 时作为历史基线，方差更小、ROAS 更稳定
    - motivation_bucket: 动机桶，影响 CTR/IPM/CPI/early_roas 分布
    - vertical: game / ecommerce，与 motivation_bucket 联合微调

    iOS：噪声更大、ROAS 波动更明显
    Android：相对稳定
    """
    vid = getattr(variant, "variant_id", str(variant))
    sell_point = getattr(variant, "sell_point", "") or ""
    seed = f"{vid}_{os}_baseline={baseline}"
    rng = _seeded_random(seed)

    quality = _variant_quality(vid) * _sell_point_factor(sell_point)
    ctr_ipm_f, cpi_f, roas_f = _motivation_bucket_factors(motivation_bucket, vertical)

    # 1. impressions：基线略高（更成熟）
    imp_base = _IMPRESSIONS_BASE * (1.1 if baseline else 1.0)
    imp_noise = _IMPRESSIONS_VARIANCE * (0.5 if baseline else 1.0)
    imp_noise = imp_noise * (1.3 if os == "iOS" else 0.9)  # iOS 波动更大
    impressions = int(
        _add_noise(imp_base * quality, imp_noise, rng)
    )
    impressions = max(5000, min(impressions, 200_000))

    # 2. CTR（受 motivation_bucket 影响）
    ctr_base = (rng.uniform(*_CTR_RANGE) + sum(_CTR_RANGE) / 2) / 2 * quality * ctr_ipm_f
    ctr_noise = 0.15 if baseline else (0.25 if os == "iOS" else 0.18)
    ctr = _add_noise(ctr_base, ctr_noise, rng)
    ctr = max(0.003, min(ctr, 0.04))

    # 3. clicks
    clicks = int(impressions * ctr)
    clicks = max(1, clicks)

    # 4. IPM（千次曝光安装，受 motivation_bucket 影响）
    ipm_base = rng.uniform(*_IPM_RANGE) * quality * ctr_ipm_f
    ipm_noise = 0.12 if baseline else (0.22 if os == "iOS" else 0.15)
    ipm = _add_noise(ipm_base, ipm_noise, rng)
    ipm = max(3, min(ipm, 80))

    # 5. installs
    installs = int(impressions * ipm / 1000)
    installs = max(1, installs)

    # 6. CPI：iOS 更高，受 motivation_bucket 影响
    cpi_range = _CPI_RANGE_IOS if os == "iOS" else _CPI_RANGE_ANDROID
    cpi_base = rng.uniform(*cpi_range) / quality * cpi_f
    cpi_noise = 0.1 if baseline else (0.2 if os == "iOS" else 0.12)
    cpi = _add_noise(cpi_base, cpi_noise, rng)
    cpi = max(0.8, min(cpi, 12))

    # 7. spend
    spend = round(installs * cpi, 2)
    spend = max(10, spend)

    # 8. early_events（D1 事件数）
    epi = rng.uniform(*_EVENTS_PER_INSTALL)
    early_events = int(installs * epi)
    early_events = max(0, early_events)

    # 9. early_revenue：iOS ROAS 更不稳定，体验桶等对 early_roas 更敏感
    roas_base = rng.uniform(*_EARLY_ROAS_RANGE) * roas_f
    roas_noise = 0.3 if baseline else (0.6 if os == "iOS" else 0.4)
    early_roas = _add_noise(roas_base, roas_noise, rng)
    early_roas = max(0, min(early_roas, 0.5))
    early_revenue = round(spend * early_roas, 2)

    # 有时为 0
    if rng.random() < 0.15 and not baseline:
        early_revenue = 0
        early_roas = 0

    # 10. 重算派生（保证一致）
    ctr_final = round(clicks / impressions, 6)
    ipm_final = round(installs / impressions * 1000, 2)
    cpi_final = round(spend / installs, 2)
    early_roas_final = round(early_revenue / spend, 4) if spend > 0 else 0

    # 11. 电商：退款风险 + 转化/下单代理（purchase/value 目标）
    refund_risk_val = 0.0
    conversion_proxy_val = 0.0
    order_proxy_val = 0.0
    if (vertical or "casual_game").lower() == "ecommerce":
        base_refund = 0.08 + rng.uniform(0, 0.12)
        refund_risk_val = round(max(0, min(1, base_refund - early_roas_final * 0.5 + (1 - quality) * 0.1)), 3)
        # 转化代理：与 CTR、early_roas 正相关
        conversion_proxy_val = round(ctr_final * 2.5 * (0.8 + rng.uniform(0, 0.4)), 4)
        order_proxy_val = round(early_roas_final * 3.0 * (0.7 + rng.uniform(0, 0.5)), 4)

    return SimulatedMetrics(
        variant_id=vid,
        os=os,
        baseline=baseline,
        impressions=impressions,
        clicks=clicks,
        installs=installs,
        spend=spend,
        early_events=early_events,
        early_revenue=early_revenue,
        ctr=ctr_final,
        ipm=ipm_final,
        cpi=cpi_final,
        early_roas=early_roas_final,
        refund_risk=refund_risk_val,
        conversion_proxy=conversion_proxy_val,
        order_proxy=order_proxy_val,
    )
