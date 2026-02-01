"""生成 prompt 与评审 prompt，支持 game / ecommerce 两种 vertical"""
from __future__ import annotations

import json
from typing import Literal

from schemas import CreativeCard, CreativeVariant

VERTICAL_PROMPTS = {
    "game": {
        "gen_hint": (
            "游戏类素材需突出玩法、爽感、福利；hook 要有冲击力，"
            "script 要有情绪起伏，shotlist 需考虑游戏画面与真人混剪。"
        ),
        "review_hint": (
            "重点关注玩法表达是否清晰、福利/活动信息是否准确、"
            "是否存在夸大或误导性宣称、是否适合游戏平台投放。"
        ),
        "risks_fixes_priorities": (
            "【risks/fixes 优先检查项】请将以下检查点作为 risks 和 fixes 的优先输出内容："
            "① hook 是否匹配目标用户的情绪动机；"
            "② 承诺是否能在新手 3 分钟内兑现；"
            "③ 是否可能误导玩法或造成预期落差。"
        ),
    },
    "ecommerce": {
        "gen_hint": (
            "电商类素材需突出产品卖点、价格优势、使用场景；"
            "hook 要直击痛点，script 要有种草逻辑，shotlist 需展示产品细节。"
        ),
        "review_hint": (
            "重点关注卖点是否属实、价格/促销信息是否清晰、"
            "是否存在虚假宣传、是否符合电商平台广告规范。"
        ),
        "risks_fixes_priorities": (
            "【risks/fixes 优先检查项】请将以下检查点作为 risks 和 fixes 的优先输出内容："
            "① 价格/对比/优惠/物流/售后表述是否合规；"
            "② 功效宣称是否需要证据支撑；"
            "③ 是否避免夸大功效。"
        ),
    },
}


def build_generation_prompt(card: CreativeCard, n: int = 5) -> str:
    """构建生成变体的 prompt（广告投放创意生成引擎）"""
    card_json = json.dumps(card.model_dump(), ensure_ascii=False, indent=2)

    return f"""你是广告投放创意生成引擎。你的任务：根据输入的 CreativeCard，生成 {n} 条 CreativeVariant。
【强约束】
- 只允许输出一个 JSON 对象：{{"variants":[...]}}，不要 Markdown，不要解释。
- variants 数组长度必须等于 {n}
- 每条 CreativeVariant 必须包含并仅包含这些字段（多余字段也允许但不推荐）：
  variant_id, hook_type, who_why_now{{who,why,why_now}}, script{{shots:[{{t,visual,overlay_text,voiceover,sfx_bgm}}]}}, cta, risk_flags{{policy_risk,exaggeration_risk,white_traffic_risk}}, notes
- shots：3~5 个镜头；t 递增；总时长 <= 15 秒
- cta（行动号召 CTA / Call-To-Action）：必须是可执行动作短句（例：立即下载/马上开玩/领福利）
- 若 card.no_exaggeration=true：禁止夸大承诺、禁止「最强/必胜/稳赚/零成本」等；risk_flags.exaggeration_risk 需要反映风险。
- vertical=game：脚本更偏情绪/爽点/胜负/社交；vertical=ecommerce：脚本更偏痛点/对比/利益点/到手价。

【输入 CreativeCard（JSON）】
{card_json}

现在开始输出 JSON：
"""


def build_review_prompt(card: CreativeCard, variants: list[CreativeVariant]) -> str:
    """构建评审变体的 prompt（投放素材评审官）"""
    card_json = json.dumps(card.model_dump(), ensure_ascii=False, indent=2)
    variants_data = [v.model_dump() for v in variants]
    variants_json = json.dumps(variants_data, ensure_ascii=False, indent=2)

    return f"""你是投放素材评审官。你将收到：CreativeCard + 一组 CreativeVariant。
你的任务：对每条 variant 输出 ReviewResult，并给出门禁决策建议（PASS/REVISE/KILL）。

【强约束】
- 只输出一个 JSON 对象：{{"overall_summary":"...","results":[...]}}
- results 数组长度必须等于 variants 数量，且每项的 variant_id 必须与输入一致
- 每条 ReviewResult 必须包含字段：
  variant_id,
  scores{{clarity,hook_strength,sell_point_strength,cta_quality,compliance_safety,expected_test_value}},
  decision,
  key_reasons[],
  required_fixes[{{fix,why,how}}],
  fuse{{fuse_level,fuse_reasons}},
  white_traffic_risk_final
- 分数 0-100 的整数
- decision 仅允许：PASS / SOFT_FAIL / HARD_FAIL
- fuse.fuse_level 仅允许：none / low / medium / high
- white_traffic_risk_final 仅允许：low / medium / high

【评审口径】
- clarity：是否一句话说清「给谁/卖什么/为什么现在」
- hook_strength：前三秒是否抓住动机（hook_type 是否和目标人群一致）
- sell_point_strength：Why you + Why now 是否成立，是否有可验证利益点
- cta_quality：CTA 是否具体、低阻力、与脚本承接一致
- compliance_safety：是否夸大/违规/误导（no_exaggeration=true 时更严格）
- expected_test_value：是否值得小预算试投（结构是否清晰可复用）

【白量/无效试投风险（white_traffic_risk_final）判断】
- 低：信息清晰、承接自然、无诱导点击
- 中：信息不够清晰或承接弱，可能带来低质点击
- 高：强诱导/噱头/信息不完整/承诺过大，极易跑白量

【输入】
CreativeCard(JSON):
{card_json}

CreativeVariants(JSON 数组):
{variants_json}

现在开始输出 JSON：
"""


def build_experiment_prompt(card_json: str, review_json: str) -> str:
    """构建投放实验建议的 prompt（最小可行投放实验设计器）"""
    return f"""你是「最小可行投放实验设计器」。你必须**只输出一个合法 JSON 对象**，不要解释、不要 Markdown、不要代码块。

【输入：评审摘要 JSON】
{review_json}

【任务】
给出一个最小可行的投放实验建议，输出必须符合以下结构：

{{
  "should_test": true/false,
  "suggested_segment": "建议国家/人群分层（中文，越具体越好）",
  "suggested_channel_type": "信息流/搜索/网络型（中文）",
  "budget_range": "建议预算区间（中文，例如：$200-$500/天 或 2000-5000元/天）",
  "gate_metrics": ["门禁指标1","门禁指标2","门禁指标3"],
  "stop_loss_condition": "止损条件（中文，含时间窗口+阈值）",
  "experiment_goal": "本次实验要验证的假设（中文，一句话）"
}}

【规则】
1) should_test=false 仅当：大部分变体 fuse_level=RED 或 decision=KILL，且理由是合规/严重承诺风险。
2) gate_metrics 必须可执行、可衡量（例：首日点击率、安装转化率、次日留存、付费率、风控后净回收等），不要空话。
3) stop_loss_condition 必须包含：窗口（如「48小时内」）+ 阈值（如「点击率<0.8% 或 CPI>目标*1.3」）。
4) 只输出 JSON。
"""
