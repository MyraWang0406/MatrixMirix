"""
投放实验决策系统 (Decision Support System) - 创意评测
产品化 UI，无 session_state/widget 冲突，同页 Tab 切换。
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import streamlit as st

# set_page_config 由 streamlit_app.py（入口壳）负责，此处不再调用

# 其余导入（部署失败时可在此暴露错误）
try:
    from element_scores import ElementScore, compute_element_scores
    from eval_schemas import StrategyCard, Variant
    from eval_set_generator import CardEvalRecord, generate_eval_set
    from explore_gate import evaluate_explore_gate
    from ofaat_generator import generate_ofaat_variants
    from scoring_eval import compute_card_score, compute_variant_score
    from simulate_metrics import SimulatedMetrics, simulate_metrics
    from vertical_config import (
        get_corpus,
        get_why_now_pool,
        get_why_now_strong_stimulus_penalty,
        get_why_now_strong_triggers,
        get_why_you_examples,
    )
    from validate_gate import WindowMetrics, evaluate_validate_gate
    from variant_suggestions import next_variant_suggestions
    from ui.styles import get_global_styles
except Exception as e:
    st.error(f"导入失败: {e}")
    st.code(traceback.format_exc(), language="text")
    st.stop()

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


def _render_health_page():
    """健康检查页：快速排查 key/网络/导入问题。URL: ?page=health 或 导航点 Health"""
    st.subheader("🏥 健康检查 (Health Check)")
    rows = []
    rows.append(("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))
    rows.append(("Streamlit", st.__version__))
    import_ok = True
    for name in ["pydantic", "element_scores", "eval_schemas", "decision_summary"]:
        try:
            __import__(name)
            rows.append((f"import {name}", "✓"))
        except Exception as e:
            rows.append((f"import {name}", f"✗ {str(e)[:50]}"))
            import_ok = False
    for k, v in rows:
        st.write(f"**{k}**: {v}")
    import os as _os
    api_key_set = bool(_os.environ.get("OPENROUTER_API_KEY", "").strip())
    st.write("**OPENROUTER_API_KEY**:", "✓ 已设置" if api_key_set else "○ 未设置（app_demo 模拟数据，无需 Key）")
    st.write("**OPENROUTER_MODEL**:", _os.environ.get("OPENROUTER_MODEL") or "（未设置，默认 gpt-4o-mini）")
    st.success("健康检查完成" if import_ok else "部分导入失败，请检查 requirements.txt")

# 窗口 ID → 中文可读
WINDOW_LABELS = {
    "window_1": "首测窗口（同日第1窗口）",
    "window_2": "跨天复测（跨日第2窗口）",
    "expand_segment": "轻扩人群（人群扩量阶段）",
}
WINDOW_TOOLTIP = "验证分窗策略：首测=同日首次投放；跨天复测=跨日验证稳定性；轻扩人群=轻度扩圈后表现"

# 回撤 tooltip：公式说明
IPM_DROP_TOOLTIP = "IPM回撤（相对首测窗）：(首测IPM - 最低IPM) / 首测IPM"

# 跨OS一致性
CROSS_OS_LABELS = {"pos": "一致", "neg": "一致", "mixed": "不一致"}
CROSS_OS_TOOLTIP = "pos=双端一致拉/拖；neg=双端一致；mixed=双端不一致；样本不足=样本数<6"

# OFAAT 全称
OFAAT_FULL = "单因子实验（OFAAT, One-Factor-At-A-Time）"
OFAAT_TOOLTIP = "One-Factor-At-A-Time：一次只改一个变量，便于归因"

from decision_summary import compute_decision_summary

# 实验包默认参数
DEFAULT_PLATFORMS = ["iOS", "Android"]
DEFAULT_SUGGESTED_N = 12
DEFAULT_SCALE_UP_STEP_PCT = "20%"


def build_experiment_package(
    suggestion,
    *,
    platforms: list[str] | None = None,
    suggested_n: int | None = None,
    scale_up_step: str | None = None,
) -> dict:
    """
    从 VariantSuggestion 构建下一轮实验包（OFAAT 结构化 JSON）。
    不改变业务逻辑，仅做数据包装。
    """
    alts = getattr(suggestion, "candidate_alternatives", None) or []
    return {
        "changed_field": getattr(suggestion, "changed_field", ""),
        "current_value": getattr(suggestion, "current_value", ""),
        "candidate_alternatives": [str(x) for x in alts],
        "platforms": platforms or DEFAULT_PLATFORMS.copy(),
        "suggested_n": suggested_n if suggested_n is not None else DEFAULT_SUGGESTED_N,
        "scale_up_step": scale_up_step or DEFAULT_SCALE_UP_STEP_PCT,
        "delta_desc": getattr(suggestion, "delta_desc", "") or "",
        "rationale": getattr(suggestion, "rationale", "") or "",
        "confidence_level": getattr(suggestion, "confidence_level", "medium"),
        "source": "suggestion",
    }


def _queue_item_to_export_row(item: dict) -> dict:
    """队列项转导出行（CSV/JSON 兼容）"""
    alts = item.get("candidate_alternatives", [])
    return {
        "changed_field": item.get("changed_field", ""),
        "current_value": item.get("current_value", ""),
        "candidate_alternatives": " | ".join(str(x) for x in alts),
        "platforms": ", ".join(item.get("platforms", [])),
        "suggested_n": item.get("suggested_n", DEFAULT_SUGGESTED_N),
        "scale_up_step": item.get("scale_up_step", DEFAULT_SCALE_UP_STEP_PCT),
        "delta_desc": item.get("delta_desc", ""),
        "source": item.get("source", "unknown"),
    }


def export_queue_json(queue: list) -> str:
    """导出实验队列为 JSON 字符串"""
    out = [dict(item) for item in queue]
    return json.dumps(out, ensure_ascii=False, indent=2)


def export_queue_csv(queue: list) -> str:
    """导出实验队列为 CSV 字符串"""
    import io
    import csv
    if not queue:
        return "changed_field,current_value,candidate_alternatives,platforms,suggested_n,scale_up_step,delta_desc,source\n"
    rows = [_queue_item_to_export_row(item) for item in queue]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _render_decision_summary_card(summary: dict):
    """渲染决策结论卡片：红仅 FAIL，其余蓝/灰，分析型呈现。"""
    status = summary.get("status", "yellow")
    status_text = summary.get("status_text", "🟡 小步复测(20%)")
    reason = summary.get("reason", "")
    risk = summary.get("risk", "")
    next_step = summary.get("next_step", "复测")
    insufficient = summary.get("insufficient", False)
    # 语义 class：fail=红，pass=蓝，warn=灰
    status_class = "status-fail" if status == "red" else ("status-pass" if status == "green" else "status-warn")

    html = f"""
    <div class="decision-summary-hero {status_class}">
        <div class="summary-label">📌 决策结论 Summary</div>
        <div class="summary-status">{status_text}</div>
        <div class="summary-row"><b>原因：</b>{reason}</div>
        <div class="summary-row"><b>风险：</b>{risk}</div>
        <div class="summary-row"><b>下一步：</b>{next_step}{"（样本不足，建议补足数据后复测）" if insufficient else ""}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    # 下一步按钮（占位，不写库）
    btn_cols = st.columns([1, 1, 1, 5])
    with btn_cols[0]:
        if st.button("🔄 复测", key="next_retest"):
            st.toast("复测（占位，未写入数据库）")
    with btn_cols[1]:
        if st.button("📈 放量", key="next_scale", disabled=(next_step != "放量")):
            st.toast("放量（占位，未写入数据库）")
    with btn_cols[2]:
        if st.button("➕ 加入实验队列", key="next_queue"):
            st.toast("加入实验队列（占位，未写入数据库）")
    st.divider()


def _render_experiment_queue_sidebar():
    """侧边栏实验队列：展示已加入实验、导出 JSON/CSV"""
    q = st.session_state.get("experiment_queue", [])
    st.markdown("**📋 实验队列**")
    if not q:
        st.caption("暂无实验，从「变体建议」或「元素贡献」加入")
    else:
        for idx, item in enumerate(q):
            field = item.get("changed_field", "-")
            curr = (item.get("current_value", "") or "")[:12]
            alts = item.get("candidate_alternatives", [])[:2]
            st.caption(f"{idx + 1}. {field}: {curr} → {', '.join(str(a) for a in alts) or '-'}")
            if st.button("移除", key=f"queue_rm_{idx}"):
                q2 = [x for i, x in enumerate(q) if i != idx]
                st.session_state["experiment_queue"] = q2
                st.rerun()
        if st.button("清空队列", key="queue_clear"):
            st.session_state["experiment_queue"] = []
            st.rerun()
        st.divider()
        st.caption("导出")
        json_str = export_queue_json(q)
        csv_str = export_queue_csv(q)
        st.download_button("⬇ JSON", data=json_str, file_name="experiment_queue.json", mime="application/json", key="dl_json")
        st.download_button("⬇ CSV", data=csv_str, file_name="experiment_queue.csv", mime="text/csv", key="dl_csv")


def _init_session_state():
    """【关键】统一初始化 session_state，避免 widget 与 session_state 冲突。"""
    st.session_state.setdefault("view_radio", "决策看板")
    st.session_state.setdefault("vertical_select", "休闲游戏")
    st.session_state.setdefault("show_help", False)
    st.session_state.setdefault("nav_section", "sec-1")
    st.session_state.setdefault("use_generated", False)
    st.session_state.setdefault("generated_variants", None)
    st.session_state.setdefault("experiment_queue", [])  # 实验队列
    st.session_state.setdefault("elem_selected_key", None)  # 元素贡献选中卡片
    st.session_state.setdefault("eval_set_records", [])
    st.session_state.setdefault("eval_n_cards", 75)
    st.session_state.setdefault("eval_status_filter", ["未测", "探索中", "进验证", "可放量"])
    # 多选：用 setdefault 初始化，widget 只传 key 不传 default
    # 注意：multiselect 的 key 会绑定 session_state，全选/清空按钮先改 state 再 rerun


def load_mock_data(
    variants: list[Variant] | None = None,
    vertical_override: str | None = None,
    motivation_bucket_override: str | None = None,
):
    """加载 StrategyCard、Variants，并生成所有模拟数据。"""
    vert = (vertical_override or "casual_game").lower()
    if vert not in ("ecommerce", "casual_game"):
        vert = "casual_game"

    card_path = SAMPLES_DIR / f"eval_strategy_card_{vert}.json"
    variant_path = SAMPLES_DIR / f"eval_variants_{vert}.json"
    if not card_path.exists():
        card_path = SAMPLES_DIR / "eval_strategy_card.json"
    if not variant_path.exists():
        variant_path = SAMPLES_DIR / "eval_variants.json"

    with open(card_path, "r", encoding="utf-8") as f:
        card = StrategyCard.model_validate(json.load(f))
    from vertical_config import get_sample_strategy_card, get_root_cause_gap

    sample = get_sample_strategy_card(vert)
    if sample:
        card = card.model_copy(
            update={
                "vertical": vert,
                "motivation_bucket": motivation_bucket_override
                or sample.get("motivation_bucket")
                or card.motivation_bucket,
                "why_you_bucket": sample.get("why_you_bucket") or card.why_you_bucket,
                "why_you_phrase": sample.get("why_you_phrase") or card.why_you_phrase,
                "why_now_trigger_bucket": sample.get("why_now_trigger_bucket")
                or card.why_now_trigger_bucket,
                "why_now_phrase": sample.get("why_now_phrase") or card.why_now_phrase,
                "why_you_label": sample.get("why_you_phrase")
                or sample.get("why_you_label")
                or card.why_you_label,
                "why_now_trigger": sample.get("why_now_phrase")
                or sample.get("why_now_trigger")
                or card.why_now_trigger,
                "segment": sample.get("segment") or card.segment,
                "who_scenario_need": sample.get("who_scenario_need") or getattr(card, "who_scenario_need", "") or "",
                "objective": sample.get("objective") or card.objective,
                "root_cause_gap": sample.get("root_cause_gap")
                or get_root_cause_gap(vert)
                or card.root_cause_gap,
            }
        )

    if variants is None:
        with open(variant_path, "r", encoding="utf-8") as f:
            variants = [Variant.model_validate(v) for v in json.load(f)]
        variants = [
            v.model_copy(update={"parent_card_id": card.card_id})
            if v.parent_card_id != card.card_id
            else v
            for v in variants
        ]

    mb = getattr(card, "motivation_bucket", "") or (
        "省钱" if vert == "ecommerce" else "成就感"
    )
    metrics = []
    metrics.append(
        simulate_metrics(variants[0], "iOS", baseline=True, motivation_bucket=mb, vertical=vert)
    )
    metrics.append(
        simulate_metrics(variants[0], "Android", baseline=True, motivation_bucket=mb, vertical=vert)
    )
    for v in variants[1:]:
        metrics.append(
            simulate_metrics(v, "iOS", baseline=False, motivation_bucket=mb, vertical=vert)
        )
        metrics.append(
            simulate_metrics(v, "Android", baseline=False, motivation_bucket=mb, vertical=vert)
        )

    baseline_list = [m for m in metrics if m.baseline]
    variant_list = [m for m in metrics if not m.baseline]
    obj = (card.objective or "").strip() or ("purchase" if vert == "ecommerce" else "install")
    ctx_base = {
        "country": "CN",
        "objective": obj,
        "segment": card.segment,
        "motivation_bucket": mb,
    }
    explore_ios = evaluate_explore_gate(
        variant_list, baseline_list, context={**ctx_base, "os": "iOS"}
    )
    explore_android = evaluate_explore_gate(
        variant_list, baseline_list, context={**ctx_base, "os": "Android"}
    )

    element_scores = compute_element_scores(variant_metrics=metrics, variants=variants)

    from eval_schemas import decompose_variant_to_element_tags

    variant_to_tags = {v.variant_id: decompose_variant_to_element_tags(v) for v in variants}
    suggestions = next_variant_suggestions(
        element_scores,
        gate_result=explore_android,
        max_suggestions=3,
        variant_metrics=metrics,
        variant_to_tags=variant_to_tags,
        variants=variants,
        vertical=vert,
    )

    windowed = [
        WindowMetrics(
            window_id="window_1",
            impressions=50000,
            clicks=800,
            installs=2000,
            spend=6000,
            early_events=1200,
            early_revenue=480,
            ipm=40.0,
            cpi=3.0,
            early_roas=0.08,
        ),
        WindowMetrics(
            window_id="window_2",
            impressions=55000,
            clicks=880,
            installs=2090,
            spend=6270,
            early_events=1250,
            early_revenue=500,
            ipm=38.0,
            cpi=3.0,
            early_roas=0.08,
        ),
    ]
    light_exp = WindowMetrics(
        window_id="expand_segment",
        impressions=20000,
        clicks=288,
        installs=720,
        spend=2160,
        early_events=430,
        early_revenue=172,
        ipm=36.0,
        cpi=3.0,
        early_roas=0.08,
    )
    validate_result = evaluate_validate_gate(windowed, light_exp)

    variant_scores_by_row: dict[tuple[str, str], float] = {}
    for m in metrics:
        cohort = [x for x in metrics if x.os == m.os]
        variant_scores_by_row[(m.variant_id, m.os)] = compute_variant_score(
            m, cohort, os=m.os, vertical=vert
        )
    by_vid: dict[str, list[float]] = defaultdict(list)
    for (vid, _), s in variant_scores_by_row.items():
        by_vid[vid].append(s)
    variant_scores_agg = {vid: sum(s) / len(s) for vid, s in by_vid.items()}

    eligible_all = list(
        dict.fromkeys(
            (explore_ios.eligible_variants or [])
            + (explore_android.eligible_variants or [])
        )
    )
    stab_penalty = 5.0 if validate_result.validate_status == "FAIL" else 0.0
    why_now_penalty = 0.0
    strong_triggers = get_why_now_strong_triggers(vert)
    wn_trigger = getattr(card, "why_now_trigger", "") or ""
    if wn_trigger in strong_triggers:
        why_now_penalty = get_why_now_strong_stimulus_penalty(vert)
    elif any(
        "why now" in n.lower() or "虚高" in n or "强刺激" in n
        for n in validate_result.risk_notes
    ):
        why_now_penalty = get_why_now_strong_stimulus_penalty(vert) * 0.5
    card_score_result = compute_card_score(
        eligible_variants=eligible_all,
        variant_scores=variant_scores_agg,
        top_k=5,
        stability_penalty=stab_penalty,
        why_now_strong_stimulus_penalty=why_now_penalty,
    )

    return {
        "card": card,
        "vertical": vert,
        "variants": variants,
        "metrics": metrics,
        "explore_ios": explore_ios,
        "explore_android": explore_android,
        "element_scores": element_scores,
        "suggestions": suggestions,
        "validate_result": validate_result,
        "variant_scores_by_row": variant_scores_by_row,
        "card_score_result": card_score_result,
    }


def render_eval_set_view():
    """评测集视图"""
    st.session_state.setdefault("eval_n_cards", 75)
    col_n, col_btn, _ = st.columns([1, 1, 4])
    with col_n:
        n_cards = st.number_input(
            "卡片数量",
            min_value=50,
            max_value=100,
            step=5,
            key="eval_n_cards",
        )
    with col_btn:
        if st.button("生成 / 重新生成评测集", type="primary", key="eval_gen_btn"):
            try:
                with st.spinner("生成评测集中..."):
                    records = generate_eval_set(n_cards=n_cards, variants_per_card=12)
                    st.session_state["eval_set_records"] = records
                    st.session_state.pop("eval_set_error", None)
                st.rerun()
            except Exception as e:
                st.session_state["eval_set_error"] = str(e)
                st.session_state["eval_set_trace"] = traceback.format_exc()
                st.rerun()

    records: list[CardEvalRecord] = st.session_state.get("eval_set_records", [])
    if st.session_state.get("eval_set_error"):
        st.error(f"生成评测集时出错：{st.session_state['eval_set_error']}")
        with st.expander("错误详情", expanded=False):
            st.code(st.session_state.get("eval_set_trace", ""), language="text")
        if st.button("清除错误", key="clear_eval_err"):
            del st.session_state["eval_set_error"]
            st.rerun()
        return

    if not records:
        st.info("暂无数据，请点击「生成 / 重新生成评测集」")
        return

    tab1, tab2, tab3 = st.tabs([
        "结构评测集 (Structure Eval Set)",
        "探索评测集 (Explore Eval Set)",
        "验证评测集 (Validate Eval Set)",
    ])

    with tab1:
        st.subheader("结构评测集：卡片列表")
        # 【关键】status_filter 用 setdefault，multiselect 不传 default
        st.session_state.setdefault(
            "eval_status_filter", ["未测", "探索中", "进验证", "可放量"]
        )
        status_filter = st.multiselect(
            "筛选状态",
            ["未测", "探索中", "进验证", "可放量"],
            key="eval_status_filter",
            placeholder="选择状态",
        )
        filtered = [r for r in records if r.status in status_filter] if status_filter else records
        rows = [
            {
                "卡片ID": r.card.card_id,
                "分数": f"{r.card_score:.1f}",
                "状态": r.status,
                "动机桶": r.card.motivation_bucket,
                "行业": "休闲游戏" if r.card.vertical == "casual_game" else "电商",
                "人群": (r.card.segment[:20] + "…" if len(r.card.segment) > 20 else r.card.segment),
            }
            for r in filtered
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            f"共 {len(filtered)} 张 | 未测:{sum(1 for r in records if r.status=='未测')} "
            f"探索中:{sum(1 for r in records if r.status=='探索中')} "
            f"进验证:{sum(1 for r in records if r.status=='进验证')} "
            f"可放量:{sum(1 for r in records if r.status=='可放量')}"
        )

    with tab2:
        st.subheader("探索评测集：Explore 结果汇总")
        rows = []
        for r in records:
            e_ios, e_android = r.explore_ios, r.explore_android
            rows.append({
                "卡片 (card_id)": r.card.card_id,
                "状态 (status)": r.status,
                "变体数": len(r.variants),
                "iOS 通过数": len(e_ios.eligible_variants or []),
                "Android 通过数": len(e_android.eligible_variants or []),
                "iOS 门禁": "✓" if e_ios.gate_status == "PASS" else "✗",
                "Android 门禁": "✓" if e_android.gate_status == "PASS" else "✗",
            })
        st.dataframe(rows, width="stretch", hide_index=True)

    with tab3:
        st.subheader("验证评测集：Validate 明细")
        validate_records = [
            r for r in records
            if r.status in ("进验证", "可放量") and r.validate_result
        ]
        if not validate_records:
            st.info("暂无进入验证阶段的卡片")
        else:
            for r in validate_records[:20]:
                with st.expander(
                    f"{r.card.card_id} | 状态:{r.status} | Validate:{r.validate_result.validate_status}"
                ):
                    if r.validate_result.detail_rows:
                        detail_data = [
                            {
                                "窗口": WINDOW_LABELS.get(row.window_id, row.window_id),
                                "千次展示安装(IPM)": f"{row.ipm:.2f}",
                                "单次安装成本(CPI)": f"{row.cpi:.2f}",
                                "早期回报率(early_ROAS)": f"{row.early_roas:.2%}",
                            }
                            for row in r.validate_result.detail_rows
                        ]
                        st.dataframe(detail_data, width="stretch", hide_index=True)
                    sm = getattr(r.validate_result, "stability_metrics", None)
                    if sm:
                        st.caption(
                            f"波动(ipm_cv)={sm.ipm_cv:.2%} | "
                            f"{IPM_DROP_TOOLTIP}: {sm.ipm_drop_pct:.1f}% | "
                            f"CPI涨幅={sm.cpi_increase_pct:.1f}% | "
                            f"学习反复={sm.learning_iterations}"
                        )
                    for n in r.validate_result.risk_notes:
                        st.caption(f"• {n}")
            if len(validate_records) > 20:
                st.caption(f"仅展示前 20 张，共 {len(validate_records)} 张进入验证")


def _multiselect_safe(label: str, options: list[str], key: str, default_all: bool = True):
    """
    【关键】多选：setdefault 初始化，widget 不传 default。
    全选/清空按钮只改 session_state 再 rerun，不与此 widget 冲突。
    """
    if not options:
        return []
    widget_key = f"{key}_ms"
    default_val = options if default_all else options[:3]
    st.session_state.setdefault(widget_key, default_val)

    col_sel, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("全选", key=f"{key}_all"):
            st.session_state[widget_key] = options
            st.rerun()
        if st.button("清空", key=f"{key}_clear"):
            st.session_state[widget_key] = []
            st.rerun()
    with col_sel:
        selected = st.multiselect(
            label,
            options=options,
            key=widget_key,
            placeholder="选 1 项以上…",
        )
    return selected


def main():
    _init_session_state()

    # 【关键】每次 rerun 都注入样式，防止点击后颜色版式丢失
    st.markdown(get_global_styles(), unsafe_allow_html=True)

    # 健康检查：URL ?page=health 或 ?health=1 时优先显示（需 Streamlit>=1.30）
    try:
        q = getattr(st, "query_params", None)
        if q and (q.get("page") == "health" or q.get("health") == "1"):
            _render_health_page()
            return
    except Exception:
        pass

    # 右下角联系
    st.markdown(
        '<div class="contact-footer">联系作者 <a href="mailto:myrawzm0406@163.com">myrawzm0406@163.com</a></div>',
        unsafe_allow_html=True,
    )

    view = st.session_state["view_radio"]
    vert_idx = st.session_state["vertical_select"]
    vertical_choice = "casual_game" if vert_idx == "休闲游戏" else "ecommerce"
    show_help = st.session_state["show_help"]

    # ===== 顶部 Header：固定蓝系 + Tabs 同页切换 =====
    main_title = (
        "评测集 (Eval Set)" if view == "评测集"
        else "Health Check" if view == "Health"
        else "投放实验决策系统 (Decision Support System)"
    )
    st.markdown(
        f'<div id="main-header" class="title-banner">'
        f'<span class="title-text">{main_title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 顶部 Tabs
    tab_cols = st.columns([1, 1, 1, 1, 1, 4])
    with tab_cols[0]:
        if st.button("决策看板", key="nav_board", type="primary" if view == "决策看板" else "secondary"):
            st.session_state["view_radio"] = "决策看板"
            st.rerun()
    with tab_cols[1]:
        if st.button("评测集", key="nav_eval", type="primary" if view == "评测集" else "secondary"):
            st.session_state["view_radio"] = "评测集"
            st.rerun()
    with tab_cols[2]:
        if st.button("Health", key="nav_health", type="primary" if view == "Health" else "secondary"):
            st.session_state["view_radio"] = "Health"
            st.rerun()
    with tab_cols[3]:
        if st.button("休闲游戏", key="nav_game", type="primary" if vert_idx == "休闲游戏" else "secondary"):
            st.session_state["vertical_select"] = "休闲游戏"
            st.session_state["use_generated"] = False
            st.session_state["generated_variants"] = None
            st.rerun()
    with tab_cols[4]:
        if st.button("电商", key="nav_ec", type="primary" if vert_idx == "电商" else "secondary"):
            st.session_state["vertical_select"] = "电商"
            st.session_state["use_generated"] = False
            st.session_state["generated_variants"] = None
            st.rerun()
    with tab_cols[5]:
        if st.button("帮助", key="nav_help"):
            st.session_state["show_help"] = not st.session_state["show_help"]
            st.rerun()

    if show_help:
        st.info(
            "选择「决策看板」或「评测集」。决策看板：筛选 Hook/卖点/CTA 后点「生成并评测」。"
            "切换行业后语料自动切换。"
        )

    # ===== 左侧：电梯导航（锚点链接，不触发 rerun，保持锚定+样式） + 实验队列 =====
    with st.sidebar:
        st.markdown('<div class="elevator-title">📌 电梯导航</div>', unsafe_allow_html=True)
        for label, sid in [
            ("0 决策结论", "sec-0"),
            ("1 结构卡片", "sec-1"),
            ("2 实验对照表", "sec-2"),
            ("3 门禁状态", "sec-3"),
            ("4 元素贡献", "sec-4"),
            ("5 变体建议", "sec-5"),
        ]:
            st.markdown(
                f'<a href="#{sid}" class="elevator-link">{label}</a>',
                unsafe_allow_html=True,
            )

        st.divider()
        _render_experiment_queue_sidebar()

    if view == "评测集":
        render_eval_set_view()
        return

    if view == "Health":
        _render_health_page()
        return

    # ===== 决策看板主内容 =====
    corp = get_corpus(vertical_choice)
    hook_opts = corp.get("hook_type") or ["反差(Before/After)", "冲突", "结果先行", "痛点", "爽点"]
    sell_opts = corp.get("sell_point") or ["示例卖点"]
    cta_opts = corp.get("cta") or ["立即下载", "现在试试", "立即下单", "立刻试玩"]
    mb_opts = corp.get("motivation_bucket") or ["成就感", "爽感", "其他"]

    st.session_state.setdefault("filter_mb", mb_opts[0])
    st.session_state.setdefault("filter_n_gen", 12)
    mb_selected = st.session_state.get("filter_mb") or mb_opts[0]
    if mb_selected not in mb_opts:
        mb_selected = mb_opts[0]

    # 先加载数据，用于顶部决策结论（第一屏最突出）
    variants_arg = st.session_state["generated_variants"] if st.session_state["use_generated"] else None
    data = load_mock_data(
        variants=variants_arg,
        vertical_override=vertical_choice,
        motivation_bucket_override=mb_selected,
    )
    card = data["card"]
    metrics = data["metrics"]
    variants = data["variants"]
    vert = data.get("vertical", getattr(card, "vertical", "casual_game") or "casual_game")

    # ----- 0 决策结论 Summary：第一屏最突出 -----
    st.markdown('<span id="sec-0"></span>', unsafe_allow_html=True)
    summary = compute_decision_summary(data)
    _render_decision_summary_card(summary)

    st.caption("筛选与生成")
    who_scenario_opts = corp.get("who_scenario_need") or []
    # 第一行：Hook、卖点、（电商则加 人/场景/需求）、CTA
    if vertical_choice == "ecommerce" and who_scenario_opts:
        r1a, r1b, r1c, r1d = st.columns([1, 1, 1, 1])
        with r1a:
            hooks = _multiselect_safe("Hook", hook_opts, f"filter_hook_{vertical_choice}")
        with r1b:
            sells = _multiselect_safe("卖点", sell_opts, f"filter_sell_{vertical_choice}")
        with r1c:
            who_scenario = _multiselect_safe("人/场景/需求", who_scenario_opts, f"filter_who_{vertical_choice}")
        with r1d:
            ctas = _multiselect_safe("CTA", cta_opts, f"filter_cta_{vertical_choice}")
    else:
        r1a, r1b, r1c = st.columns(3)
        with r1a:
            hooks = _multiselect_safe("Hook", hook_opts, f"filter_hook_{vertical_choice}")
        with r1b:
            sells = _multiselect_safe("卖点", sell_opts, f"filter_sell_{vertical_choice}")
        with r1c:
            ctas = _multiselect_safe("CTA", cta_opts, f"filter_cta_{vertical_choice}")
        who_scenario = []
    # 第二行：动机桶、N、恢复示例、生成并评测
    r2a, r2b, r2c, r2d = st.columns([1, 0.5, 0.5, 1.5])
    with r2a:
        if st.session_state.get("filter_mb") not in mb_opts:
            st.session_state["filter_mb"] = mb_opts[0]
        mb_selected = st.selectbox("动机桶", mb_opts, key="filter_mb")
    with r2b:
        n_gen = st.number_input(
            "N",
            min_value=1,
            max_value=24,
            step=1,
            key="filter_n_gen",
            help="生成变体数量",
        )
    with r2c:
        if st.session_state["use_generated"] and st.button("恢复示例"):
            st.session_state["use_generated"] = False
            st.session_state["generated_variants"] = None
            st.rerun()
    with r2d:
        if st.button("生成并评测", type="primary"):
            if not hooks or not sells or not ctas:
                st.error("请至少各选 1 项 hook、卖点、CTA")
            else:
                sell_points_for_gen = list(sells)
                if vertical_choice == "ecommerce" and who_scenario:
                    suffix = " | " + "、".join(who_scenario)
                    sell_points_for_gen = [s + suffix for s in sells]
                card_path = SAMPLES_DIR / f"eval_strategy_card_{vertical_choice}.json"
                if not card_path.exists():
                    card_path = SAMPLES_DIR / "eval_strategy_card.json"
                with open(card_path, "r", encoding="utf-8") as f:
                    card = StrategyCard.model_validate(json.load(f))
                asset_pool = corp.get("asset_var") or {}
                vs = generate_ofaat_variants(
                    card.card_id, hooks, sell_points_for_gen, ctas, n=n_gen, asset_pool=asset_pool
                )
                st.session_state["generated_variants"] = vs
                st.session_state["use_generated"] = True
                st.success(f"已生成 {len(vs)} 个变体")
                st.rerun()

    st.divider()

    # ----- 1 结构卡片 -----
    st.markdown('<span id="sec-1"></span>', unsafe_allow_html=True)
    st.subheader("1️⃣ 结构卡片摘要")
    cols = st.columns(7 if vert == "ecommerce" else 6)
    with cols[0]:
        st.metric("动机桶", getattr(card, "motivation_bucket", "-") or "成就感")
    with cols[1]:
        st.metric("Sell Point-Why you", card.why_you_phrase or card.why_you_label)
    with cols[2]:
        st.metric("Sell Point-Why now", card.why_now_phrase or card.why_now_trigger)
    with cols[3]:
        st.metric("人群", card.segment[:18] + "…" if len(card.segment) > 18 else card.segment)
    with cols[4]:
        st.metric("行业", "休闲游戏" if vert == "casual_game" else "电商")
    with cols[5]:
        st.metric("投放目标", card.objective)
    if vert == "ecommerce":
        with cols[6]:
            wsn = getattr(card, "who_scenario_need", "") or ""
            st.metric("人/场景/需求", wsn[:18] + "…" if len(wsn) > 18 else (wsn or "-"))
    st.caption(f"国家/OS: {card.country or '-'} / {card.os or '-'}")
    if vert == "ecommerce":
        st.caption("电商：early_ROAS 权重大，含退款风险")
    if card.root_cause_gap:
        st.info(card.root_cause_gap)

    st.divider()
    st.markdown('<span id="sec-2"></span>', unsafe_allow_html=True)
    st.subheader("2️⃣ 实验对照表")
    st.caption(f"{OFAAT_FULL}")
    st.caption(f"💡 {OFAAT_TOOLTIP}")

    var_map = {v.variant_id: v for v in variants}
    explore_by_os = {"iOS": data["explore_ios"], "Android": data["explore_android"]}
    scores_by_row = data.get("variant_scores_by_row", {})

    rows = []
    for m in metrics:
        v = var_map.get(m.variant_id)
        exp = explore_by_os.get(m.os)
        status = exp.variant_details.get(m.variant_id, "-") if exp else "-"
        score_val = scores_by_row.get((m.variant_id, m.os), 0.0)
        row = {
            "变体ID": m.variant_id,
            "基线": "✓" if m.baseline else "",
            "OS": m.os,
            "分数": f"{score_val:.1f}",
            "Hook": v.hook_type if v else "-",
            "Sell Point-Why you": (
                v.why_you_expression
                or (v.sell_point[:18] + "…" if v and len(v.sell_point) > 18 else (v.sell_point if v else "-"))
            ),
            "Sell Point-Why now": (v.why_now_expression or "-"),
            "CTA": v.cta_type if v else "-",
            "曝光": f"{m.impressions:,}",
            "安装": m.installs,
            "花费": f"${m.spend:,.0f}",
            "千次展示安装(IPM)": f"{m.ipm:.1f}",
            "单次安装成本(CPI)": f"{m.cpi:.2f}",
            "早期回报率(early_ROAS)": f"{m.early_roas:.2%}",
            "门禁": status,
        }
        if vert == "ecommerce":
            row["退款风险"] = f"{getattr(m, 'refund_risk', 0):.2%}"
            row["转化代理"] = f"{getattr(m, 'conversion_proxy', 0):.2%}"
            row["下单代理"] = f"{getattr(m, 'order_proxy', 0):.2%}"
        rows.append(row)

    st.caption("💡 变体ID：同一变体在 iOS/Android 各有一行")
    st.dataframe(rows, width="stretch", hide_index=True)

    st.divider()
    st.markdown('<span id="sec-3"></span>', unsafe_allow_html=True)
    st.subheader("3️⃣ 门禁状态与结论")

    card_score_result = data.get("card_score_result", {})
    card_score_val = card_score_result.get("card_score", 0.0)
    st.metric("卡片总分", f"{card_score_val:.1f}")

    t1, t2 = st.tabs(["探索门禁", "验证门禁"])

    with t1:
        baseline_list = [m for m in metrics if m.baseline]
        baseline_by_os = {m.os: m for m in baseline_list}
        exp_ios, exp_android = data["explore_ios"], data["explore_android"]
        os_tabs = st.tabs(["iOS", "Android"])
        for tab, os_name, exp in [
            (os_tabs[0], "iOS", exp_ios),
            (os_tabs[1], "Android", exp_android),
        ]:
            with tab:
                status_icon = "✓" if exp.gate_status == "PASS" else "✗" if exp.gate_status == "FAIL" else "○"
                st.write(f"**{os_name}** {status_icon} {exp.gate_status}")
                bl = baseline_by_os.get(os_name)
                if bl:
                    variant_metrics_os = [
                        m for m in metrics if m.os == os_name and not m.baseline
                    ]
                    gate_rows = []
                    for m in variant_metrics_os:
                        better = sum([m.ctr > bl.ctr, m.ipm > bl.ipm, m.cpi < bl.cpi])
                        beat_baseline = "是" if better >= 2 else "否"
                        status = exp.variant_details.get(m.variant_id, "-")
                        gate_rows.append({
                            "变体ID": m.variant_id,
                            "千次展示安装(IPM)": f"{m.ipm:.1f}",
                            "单次安装成本(CPI)": f"{m.cpi:.2f}",
                            "早期回报率(early_ROAS)": f"{m.early_roas:.2%}",
                            "≥2指标超baseline": beat_baseline,
                            "结论": status,
                        })
                    if gate_rows:
                        st.dataframe(gate_rows, width="stretch", hide_index=True)
                with st.expander("📋 门禁说明与详情"):
                    if exp.eligible_variants:
                        st.success(f"通过: {', '.join(exp.eligible_variants)}")
                    for r in exp.reasons:
                        st.caption(f"• {r}")

    with t2:
        v = data["validate_result"]
        val_ios, val_android = st.tabs(["iOS", "Android"])
        for vt in [val_ios, val_android]:
            with vt:
                st.write(
                    "**Validate**",
                    "✓ PASS" if v.validate_status == "PASS" else "✗ FAIL",
                )
                if getattr(v, "detail_rows", None) and v.detail_rows:
                    st.caption("**明细表**（窗口含义见 tooltip）")
                    detail_data = []
                    for r in v.detail_rows:
                        wl = WINDOW_LABELS.get(r.window_id, r.window_id)
                        detail_data.append({
                            "窗口": wl,
                            "千次展示安装(IPM)": f"{r.ipm:.2f}",
                            "单次安装成本(CPI)": f"{r.cpi:.2f}",
                            "早期回报率(early_ROAS)": f"{r.early_roas:.2%}",
                            "曝光": r.impressions,
                            "花费": f"${r.spend:,.0f}",
                        })
                    st.dataframe(detail_data, width="stretch", hide_index=True)
                    st.caption(f"💡 {WINDOW_TOOLTIP}")

                if getattr(v, "stability_metrics", None):
                    sm = v.stability_metrics
                    with st.expander("📋 稳定性指标"):
                        st.write(
                            f"波动(ipm_cv): {sm.ipm_cv:.2%} | "
                            f"{IPM_DROP_TOOLTIP}: {sm.ipm_drop_pct:.1f}% | "
                            f"CPI涨幅: {sm.cpi_increase_pct:.1f}% | "
                            f"学习反复: {sm.learning_iterations}"
                        )

                with st.expander("📋 风险提示与放量建议"):
                    for n in v.risk_notes:
                        st.caption(f"• {n}")
                    for k, val in v.scale_recommendation.items():
                        st.write(f"**{k}**: {val}")

    # ----- 4 元素贡献：卡片列表 + 右侧详情 -----
    st.divider()
    st.markdown('<span id="sec-4"></span>', unsafe_allow_html=True)
    st.subheader("4️⃣ 元素级贡献表")
    st.caption(f"跨OS一致性：{CROSS_OS_TOOLTIP}")

    scores = data["element_scores"]
    dim_opts = [
        ("Hook", "hook"),
        ("Sell Point-Why you", "why_you"),
        ("Sell Point-Why now", "why_now"),
        ("CTA", "cta"),
    ]
    opts_labels = [x[0] for x in dim_opts]
    st.session_state.setdefault("elem_dim", opts_labels[0])
    if st.session_state.get("elem_dim") not in opts_labels:
        st.session_state["elem_dim"] = opts_labels[0]
    st.session_state.setdefault("elem_search", "")

    c_dim, c_search = st.columns([2, 3])
    with c_dim:
        dim_sel = st.selectbox("选择维度", opts_labels, key="elem_dim")
    with c_search:
        search_term = st.text_input(
            "搜索元素值",
            key="elem_search",
            placeholder="输入关键词过滤...",
        )

    et_map = {x[0]: x[1] for x in dim_opts}
    et = et_map.get(dim_sel, "hook")
    subset = [s for s in scores if s.element_type == et]
    if search_term:
        subset = [
            s
            for s in subset
            if search_term.lower() in (s.element_value or "").lower()
        ]

    col_cards, col_detail = st.columns([2, 1])
    with col_cards:
        if subset:
            for i, s in enumerate(subset):
                conf = getattr(s, "confidence_level", "low")
                cross_os = getattr(s, "cross_os_consistency", "mixed")
                if conf == "low":
                    tendency, ipm_d, cpi_d, action = "不确定", "-", "-", "建议复测"
                    cross_os_label = "样本不足"
                else:
                    is_pull = (
                        s.avg_IPM_delta_vs_card_mean > 0
                        or s.avg_CPI_delta_vs_card_mean < 0
                    )
                    tendency = "拉" if is_pull else "拖"
                    ipm_d = f"{s.avg_IPM_delta_vs_card_mean:+.1f}"
                    cpi_d = f"{s.avg_CPI_delta_vs_card_mean:+.2f}"
                    action = "稳定结论" if conf == "high" else "倾向"
                    cross_os_label = CROSS_OS_LABELS.get(cross_os, cross_os)

                card_cls = "pull" if tendency == "拉" else ("drag" if tendency == "拖" else "unknown")
                with st.expander(
                    f"**{s.element_value[:28]}{'…' if len(s.element_value) > 28 else ''}** | "
                    f"倾向:{tendency} | IPMΔ:{ipm_d} CPIΔ:{cpi_d} | n={s.sample_size} | {cross_os_label}",
                ):
                    st.caption(f"维度: {dim_sel} | 建议: {action}")
                    # iOS/Android 对比（简化）
                    ios_rows = [m for m in metrics if m.os == "iOS"]
                    and_rows = [m for m in metrics if m.os == "Android"]
                    st.write("**含此元素的变体**：")
                    for v in variants[:5]:
                        if s.element_value and s.element_value in str(getattr(v, et, "")):
                            st.caption(f"- {v.variant_id}")
                    if st.button("复制 Prompt", key=f"elem_copy_{i}"):
                        st.toast("已复制到剪贴板（占位）")
                    if st.button("加入实验队列", key=f"elem_queue_{i}"):
                        _ef = {"hook": "hook_type", "why_you": "why_you_bucket", "why_now": "why_now_trigger"}.get(et, et)
                        q = st.session_state.get("experiment_queue", [])
                        q.append({
                            "changed_field": _ef,
                            "current_value": s.element_value or "",
                            "candidate_alternatives": [],
                            "platforms": DEFAULT_PLATFORMS.copy(),
                            "suggested_n": DEFAULT_SUGGESTED_N,
                            "scale_up_step": DEFAULT_SCALE_UP_STEP_PCT,
                            "delta_desc": f"{_ef}: {s.element_value or ''}",
                            "rationale": "",
                            "source": "element",
                        })
                        st.session_state["experiment_queue"] = q
                        st.toast(f"已加入队列，当前 {len(q)} 项")
                        st.rerun()
        else:
            st.caption("该维度暂无数据")

    # ----- 5 下一步变体建议：表格 + 展开 + 复制/加入队列 -----
    st.divider()
    st.markdown('<span id="sec-5"></span>', unsafe_allow_html=True)
    st.subheader("5️⃣ 下一步变体建议")

    suggestions = data.get("suggestions", [])
    if not suggestions:
        st.caption("样本不足或当前元素表现均不低于卡片均值，暂无优化建议")
    else:
        sug_rows = []
        for i, s in enumerate(suggestions, 1):
            if hasattr(s, "change_layer"):
                conf_label = {"high": "高", "medium": "中", "low": "低"}.get(
                    getattr(s, "confidence_level", "low"), "低"
                )
                exp_metric = getattr(s, "expected_metric", "") or getattr(
                    s, "expected_improvement", ""
                )
                alts = getattr(s, "candidate_alternatives", None) or []
                delta = getattr(s, "delta_desc", "") or (
                    f"{getattr(s, 'changed_field', '')}: {s.current_value} → "
                    f"{', '.join(str(x) for x in alts[:2])}"
                )
                sug_rows.append({
                    "改动字段": getattr(s, "changed_field", ""),
                    "当前→候选": f"{s.current_value} → {', '.join(str(x) for x in alts[:2])}",
                    "预期提升(IPM)": exp_metric,
                    "置信度": conf_label,
                    "样本": getattr(s, "sample_size", "-"),
                    "推荐动作": "复测" if conf_label == "低" else "替换",
                })

        st.dataframe(sug_rows, width="stretch", hide_index=True)

        for i, s in enumerate(suggestions, 1):
            if hasattr(s, "change_layer"):
                alts = getattr(s, "candidate_alternatives", None) or []
                delta = getattr(s, "delta_desc", "") or ""
                with st.expander(f"实验工单 {i} 详情"):
                    st.write("**改动（只改一变量）**：", delta)
                    st.write("**候选替代**：", ", ".join(str(x) for x in alts))
                    st.write("**依据**：", getattr(s, "rationale", "") or "")
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("复制 Prompt", key=f"sug_copy_{i}"):
                        st.toast("已复制到剪贴板（占位）")
                with b2:
                    if st.button("加入实验队列", key=f"sug_queue_{i}"):
                        pkg = build_experiment_package(s)
                        q = st.session_state.get("experiment_queue", [])
                        q.append(pkg)
                        st.session_state["experiment_queue"] = q
                        st.toast(f"已加入队列，当前 {len(q)} 项")
                        st.rerun()
                with b3:
                    pkg = build_experiment_package(s)
                    st.download_button(
                        "一键生成下一轮实验",
                        data=json.dumps(pkg, ensure_ascii=False, indent=2),
                        file_name=f"experiment_round_{i}_{getattr(s, 'changed_field', 'x')}.json",
                        mime="application/json",
                        key=f"sug_gen_{i}",
                    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"运行错误: {e}")
        st.code(traceback.format_exc(), language="text")
