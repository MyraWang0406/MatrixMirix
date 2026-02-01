"""StrategyCard（结构组合卡）模型与校验"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# -------- 子模型 --------


class StoryboardShot(BaseModel):
    """分镜镜头（3-5 个）"""
    t: float = Field(default=0.0, description="时间点（秒）")
    visual: str = Field(default="", description="画面/镜头描述")
    overlay_text: str = Field(default="", description="屏幕字幕")
    voiceover: str = Field(default="", description="口播文案")
    sfx_bgm: str = Field(default="", description="音效/BGM 建议")


class ExpressionLayer(BaseModel):
    """叙事骨架"""
    narrative_arc: str = Field(default="", description="叙事弧线/结构")
    key_moments: list[str] = Field(default_factory=list, description="关键叙事节点")
    tone: str = Field(default="", description="整体叙事调性")


class AssetLayerVariables(BaseModel):
    """资产层变量：镜头模板/字幕模板/BGM/节奏等"""
    shot_template: str = Field(default="", description="镜头模板")
    subtitle_template: str = Field(default="", description="字幕模板")
    bgm: str = Field(default="", description="背景音乐")
    rhythm: str = Field(default="", description="节奏/剪辑节奏")


# -------- StrategyCard 主模型 --------


class StrategyCard(BaseModel):
    """结构组合卡：评测对象从单条素材提升为策略层组合"""

    # 投放维度
    country: str = Field(default="", description="投放国家/地区")
    os: str = Field(default="", description="操作系统：iOS / Android / all")
    objective: str = Field(default="", description="投放目标：install / purchase / lead 等")
    segment: str = Field(default="", description="人群分层")

    # 动机与触发
    motivation_bucket: str = Field(default="", description="动机桶：无聊/压力/胜负欲/收集欲等")
    why_you_bucket: str = Field(default="", description="Why you 桶")
    why_now_trigger: str = Field(default="", description="Why now 触发器")

    # 创意核心
    hook_type: str = Field(default="", description="Hook 类型")
    sell_point: str = Field(default="", description="卖点表述（why_you + why_now）")
    cta_text: str = Field(default="", description="行动号召文案")

    # 素材结构
    storyboard_shots: list[StoryboardShot] = Field(
        default_factory=list,
        description="分镜镜头 3-5 个",
    )
    expression_layer: ExpressionLayer | dict[str, Any] | str = Field(
        default_factory=lambda: ExpressionLayer(),
        description="叙事骨架",
    )
    asset_layer_variables: AssetLayerVariables | dict[str, Any] = Field(
        default_factory=lambda: AssetLayerVariables(),
        description="资产层变量：镜头模板/字幕模板/BGM/节奏等",
    )

    # 元信息
    card_id: str = Field(default="", description="卡片唯一 ID")
    version: str = Field(default="1.0", description="版本")
    tags: list[str] = Field(default_factory=list, description="标签")


# -------- 校验函数 --------


_REQUIRED_FIELDS = [
    ("country", "投放国家 country"),
    ("os", "操作系统 os"),
    ("objective", "投放目标 objective"),
    ("segment", "人群分层 segment"),
    ("motivation_bucket", "动机桶 motivation_bucket"),
    ("why_you_bucket", "Why you 桶 why_you_bucket"),
    ("why_now_trigger", "Why now 触发器 why_now_trigger"),
    ("hook_type", "Hook 类型 hook_type"),
    ("sell_point", "卖点表述 sell_point"),
    ("cta_text", "行动号召 cta_text"),
    ("card_id", "卡片 ID card_id"),
    ("version", "版本 version"),
]


def validate_strategy_card(data: dict | StrategyCard) -> list[str]:
    """
    校验 StrategyCard，缺字段返回可读错误列表。
    返回空列表表示校验通过。

    Usage:
        errors = validate_strategy_card(card_dict)
        if errors:
            for e in errors:
                print(e)
    """
    errors: list[str] = []

    if isinstance(data, StrategyCard):
        obj = data
    else:
        try:
            obj = StrategyCard.model_validate(data)
        except Exception as e:
            return [f"JSON 解析失败：{e}"]

    # 1. 必填字段非空
    for field_name, desc in _REQUIRED_FIELDS:
        val = getattr(obj, field_name, None)
        if val is None or (isinstance(val, str) and not str(val).strip()):
            errors.append(f"缺少或为空：{desc}")

    # 2. storyboard_shots 数量 3-5
    shots = obj.storyboard_shots or []
    if len(shots) < 3:
        errors.append(f"storyboard_shots 至少需 3 个镜头，当前 {len(shots)} 个")
    elif len(shots) > 5:
        errors.append(f"storyboard_shots 最多 5 个镜头，当前 {len(shots)} 个")

    # 3. expression_layer 不能为空
    exp = obj.expression_layer
    if exp is None:
        errors.append("缺少 expression_layer（叙事骨架）")
    elif isinstance(exp, dict) and not exp:
        errors.append("expression_layer 不能为空对象")
    elif isinstance(exp, str) and not exp.strip():
        errors.append("expression_layer 不能为空字符串")
    elif isinstance(exp, ExpressionLayer) and not (
        exp.narrative_arc or exp.key_moments or exp.tone
    ):
        errors.append("expression_layer 至少需填写 narrative_arc/key_moments/tone 之一")

    # 4. asset_layer_variables 不能为空
    asset = obj.asset_layer_variables
    if asset is None:
        errors.append("缺少 asset_layer_variables（资产层变量）")
    elif isinstance(asset, dict) and not asset:
        errors.append("asset_layer_variables 不能为空对象")
    elif isinstance(asset, AssetLayerVariables) and not (
        asset.shot_template or asset.subtitle_template or asset.bgm or asset.rhythm
    ):
        errors.append("asset_layer_variables 至少需填写 shot_template/subtitle_template/bgm/rhythm 之一")

    return errors
