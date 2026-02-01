"""
评测系统：结构组合卡 + 结构级变体 + 元素标签。
主对象为 StrategyCard，Variant 为卡下变体。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# -------- 枚举定义（bucket 与 phrase 分离）--------

MotivationBucket = Literal[
    "省钱", "体验", "社交", "成就感", "收集", "爽感", "品质", "口碑", "其他",
]

# Why you 6 桶（电商+游戏通用）
WhyYouBucket = Literal[
    "更省钱", "更省事", "更强效果", "更匹配我", "更安全可信", "更好体验", "其他",
]

# Why now 7 桶（通用）
WhyNowBucket = Literal[
    "限时稀缺", "节点事件", "痛点升高", "机会出现", "社会驱动", "损失厌恶", "即时替代", "其他",
]

# 兼容：旧 key -> 新 bucket
_KEY_TO_WHY_YOU_BUCKET: dict[str, str] = {
    "price_advantage": "更省钱", "limited_offer": "更省钱", "quality_guarantee": "更安全可信",
    "word_of_mouth": "更安全可信", "need_based": "更匹配我", "experience_upgrade": "更好体验",
    "hero_easy": "更省事", "rank_easier": "更省事", "season_reward": "更省钱",
    "social_showcase": "更匹配我", "catharsis": "更强效果", "other": "其他",
}

ElementType = Literal["hook", "why_you", "why_now", "sell_point", "sell_point_copy", "cta", "asset"]


# -------- 1. StrategyCard（结构组合卡）--------


class StrategyCard(BaseModel):
    """结构组合卡：评测系统主对象"""

    card_id: str = Field(..., description="卡片唯一 ID")
    version: str = Field(default="1.0", description="版本")

    # 投放维度
    vertical: str = Field(default="casual_game", description="casual_game / ecommerce")
    country: str = Field(default="", description="投放国家/地区")
    os: str = Field(default="", description="iOS / Android / all")
    objective: str = Field(default="", description="install / purchase / lead 等")
    segment: str = Field(default="", description="人群分层")
    who_scenario_need: str = Field(default="", description="电商：什么人什么场景什么需求")

    # 动机
    motivation_bucket: MotivationBucket = Field(..., description="动机桶")

    # Why you：桶 + 具体短语
    why_you_bucket: WhyYouBucket = Field(default="其他", description="Why you 桶（6 桶通用）")
    why_you_phrase: str = Field(default="", description="Why you 具体短语，如：同款全网最低价/上手快")

    # Why now：桶 + 具体短语
    why_now_trigger_bucket: WhyNowBucket = Field(default="其他", description="Why now 桶（7 桶通用）")
    why_now_phrase: str = Field(default="", description="Why now 具体短语，如：涨价预警/新赛季开启")

    # 兼容字段（内部用）
    why_you_key: str = Field(default="other", description="兼容：评测/聚合用 key")
    why_you_label: str = Field(default="其他", description="兼容：UI 展示，= why_you_phrase 或桶名")
    why_now_trigger: str = Field(default="其他", description="兼容：= why_now_phrase 或桶名")

    root_cause_gap: str = Field(default="", description="根因/缺口解释")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy(cls, data: Any) -> Any:
        """兼容旧格式：why_you_key/label、why_now_trigger -> bucket + phrase"""
        if not isinstance(data, dict):
            return data
        d = dict(data)

        # 旧 why_you_bucket / why_you_key / why_you_label
        wyb = d.get("why_you_bucket")
        wyk = d.get("why_you_key")
        wyz = d.get("why_you_label")
        if d.get("why_you_phrase") is None or d.get("why_you_phrase") == "":
            if wyz and isinstance(wyz, str):
                d["why_you_phrase"] = wyz
            elif wyb and isinstance(wyb, str):
                d["why_you_phrase"] = str(wyb)
        if d.get("why_you_bucket") is None or d.get("why_you_bucket") == "":
            d["why_you_bucket"] = _KEY_TO_WHY_YOU_BUCKET.get(str(wyk or ""), "其他")

        # 旧 why_now_trigger -> phrase + bucket
        wnt = d.get("why_now_trigger")
        if d.get("why_now_phrase") is None or d.get("why_now_phrase") == "":
            if wnt and isinstance(wnt, str):
                d["why_now_phrase"] = str(wnt)
        if d.get("why_now_trigger_bucket") is None or d.get("why_now_trigger_bucket") == "":
            wnt_str = str(wnt or "")
            _wn_phrase_to_bucket = {
                "新赛季刚开": "节点事件", "大促来袭": "节点事件", "限时活动": "限时稀缺",
                "节日促销": "节点事件", "版本更新": "节点事件", "涨价预警": "限时稀缺",
                "限时秒杀": "限时稀缺", "新手福利": "机会出现", "首充翻倍": "机会出现",
            }
            d["why_now_trigger_bucket"] = _wn_phrase_to_bucket.get(wnt_str, "其他")

        d.setdefault("why_you_label", d.get("why_you_phrase") or d.get("why_you_bucket") or "其他")
        d.setdefault("why_now_trigger", d.get("why_now_phrase") or d.get("why_now_trigger_bucket") or "其他")
        return d


# -------- 2. Variant（结构级变体）--------


class AssetVariables(BaseModel):
    """资产层变量"""
    subtitle_template: str = Field(default="", description="字幕模板")
    bgm: str = Field(default="", description="BGM/背景音乐")
    rhythm: str = Field(default="", description="节奏/剪辑节奏")
    shot_template: str = Field(default="", description="镜头模板")


class Variant(BaseModel):
    """结构级变体：StrategyCard 下的具体创意表达"""

    variant_id: str = Field(..., description="变体唯一 ID")
    parent_card_id: str = Field(..., description="所属 card_id")

    hook_type: str = Field(default="", description="Hook 类型")
    sell_point: str = Field(..., description="说服层表达：why_you + why_now 可读组合")
    cta_type: str = Field(default="", description="CTA 类型")

    expression_template: str = Field(default="", description="叙事模板标识")
    asset_variables: AssetVariables | dict[str, Any] = Field(default_factory=AssetVariables)

    why_you_expression: str = Field(default="", description="Why you 具体表述")
    why_now_expression: str = Field(default="", description="Why now 具体表述")

    changed_field: str = Field(default="", description="OFAAT 改动字段")
    delta_desc: str = Field(default="", description="人类可读改动描述")


# -------- 3. ElementTag（元素标签）--------


class ElementTag(BaseModel):
    """元素标签：用于元素级贡献分析"""
    element_type: ElementType = Field(...)
    element_value: str = Field(default="")


def decompose_variant_to_element_tags(variant: Variant) -> list[ElementTag]:
    """将 Variant 拆解为一组 ElementTag"""
    tags: list[ElementTag] = []
    if variant.hook_type:
        tags.append(ElementTag(element_type="hook", element_value=variant.hook_type))
    why_you_val = variant.why_you_expression or variant.sell_point
    if why_you_val:
        tags.append(ElementTag(element_type="why_you", element_value=why_you_val))
    why_now_val = variant.why_now_expression or variant.sell_point
    if why_now_val:
        tags.append(ElementTag(element_type="why_now", element_value=why_now_val))
    if variant.sell_point:
        tags.append(ElementTag(element_type="sell_point", element_value=variant.sell_point))
    if variant.sell_point:
        tags.append(ElementTag(element_type="sell_point_copy", element_value=variant.sell_point))
    if variant.cta_type:
        tags.append(ElementTag(element_type="cta", element_value=variant.cta_type))
    asset = variant.asset_variables
    if isinstance(asset, dict):
        for k, v in asset.items():
            if v and isinstance(v, str):
                tags.append(ElementTag(element_type="asset", element_value=f"{k}={v}"))
    else:
        for name, val in [("subtitle_template", asset.subtitle_template), ("bgm", asset.bgm),
                          ("rhythm", asset.rhythm), ("shot_template", asset.shot_template)]:
            if val:
                tags.append(ElementTag(element_type="asset", element_value=f"{name}={val}"))
    return tags
