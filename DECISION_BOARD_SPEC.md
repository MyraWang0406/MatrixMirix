# Decision Board 页面信息结构设计

## 一、页面模块结构说明

```
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: StrategyCard 摘要                                     │
│  Why you / Why now / 人群 / 国家 / OS                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Module 2: Variant 对照表（含 baseline）                          │
│  变体横向对比：指标、Gate 状态                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Module 3: Explore / Validate Gate 状态与结论                     │
│  门禁结果、原因说明、建议                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Module 4: Element-level 贡献表                                   │
│  哪个 Hook / Why you / CTA 在拉或拖指标                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Module 5: 下一步变体建议                                         │
│  系统给出的可执行建议（可照着做素材）                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、各模块数据字段列表

### Module 1: StrategyCard 摘要

| 字段 | 类型 | 说明 |
|------|------|------|
| card_id | string | 卡片唯一 ID |
| version | string | 版本 |
| why_you_bucket | string | Why you 桶（策略层） |
| why_now_trigger | string | Why now 触发器 |
| motivation_bucket | string | 动机桶 |
| motivation_bucket | string | 动机桶（必填枚举：省钱/体验/社交/胜负欲等） |
| segment | string | 人群分层 |
| country | string | 投放国家/地区 |
| os | string | iOS / Android / all |
| objective | string | 投放目标（install / purchase 等） |
| root_cause_gap | string | 根因/缺口解释（可选展示） |

**展示建议**：一行或卡片形式，突出 Why you / Why now / 人群 / 国家 / OS。

---

### Module 2: Variant 对照表（含 baseline）

| 字段 | 类型 | 说明 |
|------|------|------|
| variant_id | string | 变体 ID |
| is_baseline | boolean | 是否为 baseline |
| hook_type | string | Hook 类型 |
| sell_point | string | 说服层表达：why_you_bucket + why_now_trigger + 可读表达 |
| cta_type | string | CTA 类型 |
| expression_template | string | 表达模板（3幕/5镜头） |
| os | string | iOS / Android（若按 os 分表则每行一条） |
| **指标** | | |
| impressions | int | 曝光量 |
| clicks | int | 点击量 |
| installs | int | 安装量 |
| spend | float | 花费（USD） |
| early_events | int | 早期事件数 |
| early_revenue | float | 早期收入 |
| ctr | float | 点击率 |
| ipm | float | 千次曝光安装数 |
| cpi | float | 单次安装成本 |
| early_roas | float | 早期 ROAS |
| **Gate 状态** | | |
| explore_gate_status | string | PASS / FAIL / INSUFFICIENT / INVALID |
| validate_gate_status | string | PASS / FAIL（若已做 Validate） |

**展示建议**：表格，行=变体（或 variant_id × os），列=创意字段 + 核心指标 + Gate 状态；baseline 行高亮或标记。

---

### Module 3: Explore / Validate Gate 状态与结论

#### 3a. Explore Gate

| 字段 | 类型 | 说明 |
|------|------|------|
| gate_status | string | PASS / FAIL / INSUFFICIENT / INVALID |
| reasons | string[] | 原因说明（中文） |
| eligible_variants | string[] | 进入验证期的 variant_id |
| variant_details | object | variant_id -> 状态（PASS/FAIL/...） |
| context | object | 评测上下文（country, os, objective, segment） |

#### 3b. Validate Gate（若有）

| 字段 | 类型 | 说明 |
|------|------|------|
| validate_status | string | PASS / FAIL |
| risk_notes | string[] | 风险提示（如「轻扩人群 IPM 劣化」） |
| scale_recommendation | object | scale_up_step, stop_loss |

**展示建议**：状态徽章 + 原因列表 + 加量/止损建议。

---

### Module 4: Element-level 贡献表

| 字段 | 类型 | 说明 |
|------|------|------|
| element_type | string | hook / why_you / why_now / sell_point / cta / asset |
| element_value | string | 元素取值 |
| avg_IPM_delta_vs_card_mean | float | IPM 与卡片均值差（正=拉，负=拖） |
| avg_CPI_delta_vs_card_mean | float | CPI 与卡片均值差（负=拉，正=拖） |
| sample_size | int | 样本数 |
| stability_flag | boolean | 样本是否足够 |
| **派生展示** | | |
| contribution_label | string | 拉 / 拖 / 中性（根据 IPM/CPI delta 计算） |

**展示建议**：表格，按 element_type 分组；用颜色/图标区分「拉」与「拖」；仅展示稳定性足够的元素。

---

### Module 5: 下一步变体建议

| 字段 | 类型 | 说明 |
|------|------|------|
| suggestions | string[] | 可读中文建议，每条 1 个元素 |
| max_suggestions | int | 最多条数（默认 3） |

**每条建议内容结构（文本内嵌）**：
- 层级：策略层 / 表达层 / 素材层
- 改动：将 X 从「A」调整为其他方案
- 依据：IPM/CPI 数据
- 预期改善：IPM / CPI / early ROAS

**展示建议**：有序列表，每条可折叠或展开为结构化字段（若后端拆分）。

---

## 三、Decision Board 聚合数据结构（API 推荐）

```json
{
  "strategy_card": { /* Module 1 字段 */ },
  "variant_table": [ /* Module 2 行数据 */ ],
  "explore_gate": { /* Module 3a 字段 */ },
  "validate_gate": { /* Module 3b 字段，可为 null */ },
  "element_scores": [ /* Module 4 行数据 */ ],
  "next_suggestions": [ /* Module 5 字符串列表 */ ]
}
```

---

## 四、数据依赖关系

```
StrategyCard + Variants
    ↓
SimulatedMetrics (per variant × os)
    ↓
ExploreGateResult, ValidateGateResult (optional)
    ↓
ElementScore (from variant_metrics + decompose)
    ↓
next_variant_suggestions (ElementScore + GateResult)
```
