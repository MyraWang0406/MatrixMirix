"""Pydantic 模型：创意卡片、变体、评审结果"""
from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


# -------- 创意卡片（输入）--------


class CreativeCard(BaseModel):
    """结构卡片：投放素材的输入定义"""

    vertical: Literal["game", "ecommerce"]
    product_name: str
    target_audience: str
    key_selling_points: list[str] = Field(default_factory=list)
    tone: str = "professional"
    constraints: list[str] = Field(default_factory=list)
    extra_context: str = ""
    no_exaggeration: bool = Field(
        default=True,
        description="是否禁止夸大词（命中敏感词时熔断升级）",
    )


# -------- 生成变体（输出）--------

RiskLevel = Literal["low", "medium", "high"]


class WhoWhyNow(BaseModel):
    who: str = ""
    why: str = ""
    why_now: str = ""


class Shot(BaseModel):
    t: float = 0.0
    visual: str = ""
    overlay_text: str = ""
    voiceover: str = ""
    sfx_bgm: str = ""


class ScriptShots(BaseModel):
    shots: list[Shot] = Field(default_factory=list)


class RiskFlagsObj(BaseModel):
    policy_risk: RiskLevel = "low"
    exaggeration_risk: RiskLevel = "low"
    white_traffic_risk: RiskLevel = "low"


class CreativeVariant(BaseModel):
    """单个创意变体（分镜脚本风格）"""

    variant_id: str = Field(default="v001", description="唯一ID，如 v001/v002")
    hook_type: str = Field(default="", description="触发器类型")
    who_why_now: WhoWhyNow = Field(default_factory=WhoWhyNow)
    script: ScriptShots = Field(default_factory=ScriptShots)
    cta: str = Field(default="", description="行动号召")
    risk_flags: RiskFlagsObj | list[str] = Field(
        default_factory=lambda: RiskFlagsObj(),
        description="风险标注",
    )
    notes: str = Field(default="", description="投放建议")

    # 兼容旧字段
    headline: str = Field(default="", description="主标题，兼容")
    core_message: str = Field(default="", description="核心信息，兼容")

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_risk_flags(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        rf = data.get("risk_flags")
        if isinstance(rf, list):
            data = dict(data)
            data["risk_flags"] = RiskFlagsObj()
        return data

    @property
    def title(self) -> str:
        return self.variant_id or self.headline or self.hook_type or "-"

    @property
    def hook_3s(self) -> str:
        w = self.who_why_now
        return (w.who + " " + w.why + " " + w.why_now).strip() or self.headline or self.hook_type or ""

    @property
    def script_15s(self) -> str:
        if self.script and self.script.shots:
            return " | ".join(
                f"{s.t}s: {s.voiceover or s.overlay_text or s.visual}"
                for s in self.script.shots
            )
        return self.core_message or ""

    @property
    def cta_text(self) -> str:
        return self.cta or ""


class GenerationResponse(BaseModel):
    """生成阶段返回的变体列表"""

    variants: list[CreativeVariant] = Field(default_factory=list)


# -------- 评审结果 --------


class RequiredFix(BaseModel):
    """必须修改项"""
    fix: str = ""
    why: str = ""
    how: str = ""


class FuseInfo(BaseModel):
    """熔断信息"""
    fuse_level: Literal["none", "low", "medium", "high"] = "none"
    fuse_reasons: list[str] = Field(default_factory=list)


class ReviewScores(BaseModel):
    """评测维度评分（0-100）"""

    clarity: int = Field(default=50, ge=0, le=100)
    hook_strength: int = Field(default=50, ge=0, le=100)
    sell_point_strength: int = Field(default=50, ge=0, le=100)
    cta_quality: int = Field(default=50, ge=0, le=100)
    compliance_safety: int = Field(default=50, ge=0, le=100)
    expected_test_value: int = Field(default=50, ge=0, le=100)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_scores(cls, data: object) -> object:
        """兼容旧格式 audience_match/promise_risk 等"""
        if not isinstance(data, dict):
            return data
        if "clarity" in data and "hook_strength" in data:
            return data
        # 旧 0-5 维度映射到 0-100
        def scale5(v: int) -> int:
            return min(100, max(0, int((v or 0) / 5 * 100)))
        return {
            "clarity": data.get("clarity", scale5(data.get("audience_match", 3))),
            "hook_strength": scale5(data.get("audience_match", 3)),
            "sell_point_strength": scale5(data.get("handoff_consistency", 3)),
            "cta_quality": scale5(data.get("handoff_consistency", 3)),
            "compliance_safety": 100 - scale5(max(data.get("promise_risk", 2), data.get("white_traffic_risk", 2))),
            "expected_test_value": scale5(data.get("fit_objective", data.get("audience_match", 3))),
        }

    @property
    def audience_match(self) -> int:
        return self.clarity // 20  # 0-100 -> 0-5

    @property
    def promise_risk(self) -> int:
        return (100 - self.compliance_safety) // 20

    @property
    def white_traffic_risk(self) -> int:
        return (100 - self.compliance_safety) // 20

    @property
    def handoff_consistency(self) -> int:
        return self.sell_point_strength // 20


class ReviewResult(BaseModel):
    """单个变体的评审结果"""

    variant_id: str = Field(default="")
    scores: ReviewScores = Field(default_factory=lambda: ReviewScores())
    decision: str = Field(default="SOFT_FAIL")
    key_reasons: list[str] = Field(default_factory=list)
    required_fixes: list[RequiredFix] | list[str] = Field(default_factory=list)
    fuse: FuseInfo | None = Field(default=None)
    white_traffic_risk_final: str = Field(default="low")  # low|medium|high

    # 兼容字段
    risks: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    overall_summary: str = Field(default="", validation_alias=AliasChoices("summary", "overall_summary"))
    error: str | None = Field(default=None)
    fuse_level: str = Field(default="low")
    fuse_reasons: list[str] = Field(default_factory=list)

    def _fuse_level_str(self) -> str:
        f = self.fuse
        if f:
            return f.fuse_level
        return self.fuse_level

    def _fuse_reasons_list(self) -> list[str]:
        f = self.fuse
        if f and f.fuse_reasons:
            return f.fuse_reasons
        return self.fuse_reasons or []

    @model_validator(mode="before")
    @classmethod
    def normalize_required_fixes(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        rf = data.get("required_fixes", [])
        if isinstance(rf, list) and rf and isinstance(rf[0], str):
            data = dict(data)
            data["required_fixes"] = [RequiredFix(fix=x) for x in rf]
        return data

    @property
    def required_fixes_flat(self) -> list[str]:
        """required_fixes 转为字符串列表"""
        rf = self.required_fixes
        if not rf:
            return self.fixes or []
        out = []
        for x in rf:
            if isinstance(x, RequiredFix):
                out.append(f"{x.fix} | 原因:{x.why} | 修改:{x.how}" if x.why or x.how else x.fix)
            else:
                out.append(str(x))
        return out


class ReviewResponse(BaseModel):
    """评审阶段返回"""

    overall_summary: str = Field(default="")
    results: list[ReviewResult] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reviews_to_results(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "reviews" in data and not data.get("results"):
            data = dict(data)
            data["results"] = data["reviews"]
        return data


# -------- 门禁决策 --------


Verdict = Literal["PASS", "REVISE", "KILL"]
FuseLevel = Literal["GREEN", "YELLOW", "RED"]


class VariantWithReview(BaseModel):
    """变体 + 评审 + 熔断决策"""

    variant: CreativeVariant
    review: ReviewResult
    verdict: Verdict = "REVISE"
    white_traffic_risk_final: int = Field(default=0, ge=0, le=100)
    fuse_level: FuseLevel = "YELLOW"


# -------- 投放实验建议 --------


class ExperimentSuggestion(BaseModel):
    """最小可行投放实验建议"""

    should_test: bool = True
    suggested_segment: str = Field(default="", description="建议的国家/人群")
    suggested_channel_type: str = Field(default="", description="信息流/网络型等")
    budget_range: str = Field(default="", description="建议预算区间")
    gate_metrics: list[str] = Field(default_factory=list, description="门禁指标")
    stop_loss_condition: str = Field(default="", description="不达标时的止损条件")
    experiment_goal: str = Field(default="", description="验证什么假设")
