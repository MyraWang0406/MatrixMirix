"""Streamlit 界面：输入结构卡片 JSON，生成变体并评审，展示表格并支持导出"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# 本地可用 .env，云端依赖环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from exporters import export_csv, export_markdown
from openrouter_client import JsonParseError, chat_completion_json
from prompts import build_experiment_prompt, build_generation_prompt, build_review_prompt
from schemas import CreativeCard, CreativeVariant, ExperimentSuggestion, ReviewResponse, ReviewResult, VariantWithReview
from scoring import compute_fuse_decision

st.set_page_config(page_title="创意素材生成与评审", layout="wide")

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


def load_sample(name: str) -> str:
    p = SAMPLES_DIR / name
    if not p.exists():
        return "{}"
    return p.read_text(encoding="utf-8")


def parse_card(raw: str) -> CreativeCard | None:
    try:
        data = json.loads(raw)
        return CreativeCard.model_validate(data)
    except Exception as e:
        st.error(f"解析结构卡片失败: {e}")
        return None


def run_generation(card: CreativeCard, n: int) -> list[CreativeVariant]:
    prompt = build_generation_prompt(card, n=n)
    try:
        out, raw = chat_completion_json(
            [{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=8192,
            return_raw=True,
        )
        st.session_state["raw_generation"] = raw

        if isinstance(out, list):
            variants_data = out
        else:
            variants_data = out.get("variants", [])

        variants = [CreativeVariant.model_validate(v) for v in variants_data]
        return variants

    except JsonParseError as e:
        st.session_state["raw_generation"] = getattr(e, "raw_content", "")
        st.error(f"生成 JSON 解析失败（已重试）：{e}")
        return []

    except Exception as e:
        st.error(f"生成失败: {e}")
        return []


def run_review(card: CreativeCard, variants: list[CreativeVariant]) -> list[ReviewResult]:
    if not variants:
        return []

    prompt = build_review_prompt(card, variants)
    try:
        out, raw = chat_completion_json(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=8192,
            return_raw=True,
        )
        st.session_state["raw_review"] = raw

        resp = ReviewResponse.model_validate(out)
        results = resp.results
        if resp.overall_summary:
            st.session_state["review_overall_summary"] = resp.overall_summary

        # 按 variant_id 或索引对齐
        out_list: list[ReviewResult] = []
        for i, v in enumerate(variants):
            rid = getattr(v, "variant_id", "") or f"v{i+1:03d}"
            r = next((r for r in results if (r.variant_id or "").strip() == rid), None)
            if r is None and i < len(results):
                r = results[i]
            if r is None:
                r = ReviewResult(variant_id=rid)
            out_list.append(r)

        while len(out_list) < len(variants):
            out_list.append(
                ReviewResult(
                    variant_id=getattr(variants[len(out_list)], "variant_id", "")
                    or f"v{len(out_list)+1:03d}"
                )
            )
        return out_list[: len(variants)]

    except JsonParseError as e:
        st.session_state["raw_review"] = getattr(e, "raw_content", "")
        st.warning(f"评审 JSON 解析失败（已重试），返回 KILL：{e}")
        return [ReviewResult(error=f"LLM 评审结果 JSON 解析失败: {e}") for _ in variants]

    except Exception as e:
        st.error(f"评审失败: {e}")
        return [ReviewResult() for _ in variants]


def build_experiment_inputs(card: CreativeCard, rows: list[VariantWithReview]) -> tuple[str, str]:
    """构建供实验建议使用的 card_json 与 review_json"""
    card_json = json.dumps(card.model_dump(), ensure_ascii=False, indent=2)
    results = []
    for i, rw in enumerate(rows, 1):
        s = rw.review.scores
        decision = rw.review.decision or (
            "HARD_FAIL" if rw.verdict == "KILL" else "SOFT_FAIL" if rw.verdict == "REVISE" else "PASS"
        )
        fuse_level = getattr(rw.review.fuse, "fuse_level", None) if rw.review.fuse else rw.fuse_level
        fuse_map = {"GREEN": "none", "YELLOW": "low", "RED": "high"}
        fuse_level_str = fuse_map.get(str(fuse_level), str(fuse_level).lower() if fuse_level else "none")
        results.append({
            "variant_id": rw.variant.variant_id or f"v{i:03d}",
            "decision": decision,
            "scores": {
                "clarity": s.clarity,
                "hook_strength": s.hook_strength,
                "sell_point_strength": s.sell_point_strength,
                "cta_quality": s.cta_quality,
                "compliance_safety": s.compliance_safety,
                "expected_test_value": s.expected_test_value,
            },
            "fuse": {
                "fuse_level": fuse_level_str,
                "fuse_reasons": rw.review._fuse_reasons_list() if hasattr(rw.review, "_fuse_reasons_list") else (rw.review.fuse_reasons or []),
            },
            "white_traffic_risk_final": (
                rw.review.white_traffic_risk_final
                if isinstance(getattr(rw.review, "white_traffic_risk_final", None), str)
                else ("high" if rw.white_traffic_risk_final >= 67 else "medium" if rw.white_traffic_risk_final >= 34 else "low")
            ),
        })
    review_json = json.dumps(
        {"overall_summary": st.session_state.get("review_overall_summary", ""), "results": results},
        ensure_ascii=False,
        indent=2,
    )
    return card_json, review_json


def run_experiment_suggestion(card: CreativeCard, rows: list[VariantWithReview]) -> ExperimentSuggestion | None:
    """生成投放实验建议"""
    card_json, review_json = build_experiment_inputs(card, rows)
    prompt = build_experiment_prompt(card_json, review_json)
    try:
        out = chat_completion_json(
            [{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2048,
        )
        if isinstance(out, dict) and "should_test" in out:
            return ExperimentSuggestion.model_validate(out)
        return None
    except Exception as e:
        st.error(f"实验建议生成失败: {e}")
        return None


def main() -> None:
    st.title("自动化投放素材生成与评审")
    st.caption("输入结构卡片 JSON → 生成变体 → 评审 → 门禁决策（PASS/REVISE/KILL）")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("结构卡片")
        if "raw_json" not in st.session_state:
            st.session_state["raw_json"] = load_sample("game_card.json")
        if "vertical" not in st.session_state:
            st.session_state["vertical"] = "game"

        def on_vertical_change():
            v = st.session_state.get("vertical", "game")
            st.session_state["raw_json"] = load_sample(f"{v}_card.json")
            st.session_state["sample"] = f"{v}_card.json"

        def on_sample_change():
            s = st.session_state.get("sample")
            if s in ("game_card.json", "ecommerce_card.json"):
                st.session_state["raw_json"] = load_sample(s)
                st.session_state["vertical"] = "game" if s == "game_card.json" else "ecommerce"

        st.selectbox(
            "vertical",
            ["game", "ecommerce"],
            format_func=lambda x: "游戏 (game)" if x == "game" else "电商 (ecommerce)",
            key="vertical",
            on_change=on_vertical_change,
        )
        st.selectbox(
            "加载示例",
            ["game_card.json", "ecommerce_card.json", "自定义输入"],
            key="sample",
            on_change=on_sample_change,
        )
        st.text_area(
            "结构卡片 JSON",
            value=st.session_state["raw_json"],
            height=280,
            key="raw_json",
        )
        n_variants = st.number_input("生成变体数量", min_value=1, max_value=10, value=5)

        if st.button("生成并评审", type="primary"):
            card = parse_card(st.session_state["raw_json"])
            if card:
                st.session_state.pop("review_overall_summary", None)
                st.session_state.pop("experiment_suggestion", None)

                with st.spinner("生成变体中..."):
                    variants = run_generation(card, n=int(n_variants))

                if variants:
                    with st.spinner("评审中..."):
                        reviews = run_review(card, variants)

                    rows: list[VariantWithReview] = []
                    for v, r in zip(variants, reviews):
                        verdict, wt_risk, fuse = compute_fuse_decision(card, v, r)
                        rows.append(
                            VariantWithReview(
                                variant=v,
                                review=r,
                                verdict=verdict,
                                white_traffic_risk_final=wt_risk,
                                fuse_level=fuse,
                            )
                        )
                    st.session_state["results"] = rows
                    st.session_state["card"] = card
                    st.rerun()

    with col_right:
        st.subheader("评审结果")

        # ✅ Step2：Raw Output 就在这里
        with st.expander("🔍 Raw Output（调试用：模型原始返回）", expanded=False):
            rg = st.session_state.get("raw_generation", "")
            rr = st.session_state.get("raw_review", "")
            st.markdown("**生成阶段 Raw Output**")
            st.code(rg or "(empty)", language="text")
            st.markdown("**评审阶段 Raw Output**")
            st.code(rr or "(empty)", language="text")

        rows: list[VariantWithReview] = st.session_state.get("results", [])
        if not rows:
            st.info("请在左侧输入结构卡片并点击「生成并评审」")
        else:
            overall = st.session_state.get("review_overall_summary", "")
            if overall:
                st.caption("**整体总结:** " + overall)

            table_data = []
            for i, rw in enumerate(rows, 1):
                s = rw.review.scores
                summary = rw.review.error or rw.review.overall_summary or "-"
                title = (rw.variant.variant_id or rw.variant.headline or rw.variant.title) or "-"
                table_data.append({
                    "#": i,
                    "headline": title[:25] + ("..." if len(title) > 25 else ""),
                    "decision": rw.verdict,
                    "fuse_level": rw.fuse_level,
                    "white_traffic_risk_final": rw.white_traffic_risk_final,
                    "clarity": s.clarity,
                    "hook_strength": s.hook_strength,
                    "compliance_safety": s.compliance_safety,
                    "expected_test_value": s.expected_test_value,
                    "总结": summary[:40] + ("..." if len(summary) > 40 else ""),
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            st.divider()

            if "experiment_suggestion" not in st.session_state:
                st.session_state["experiment_suggestion"] = None
            if st.button("生成投放实验建议"):
                card = st.session_state.get("card")
                if card:
                    with st.spinner("生成实验建议中..."):
                        suggestion = run_experiment_suggestion(card, rows)
                        if suggestion:
                            st.session_state["experiment_suggestion"] = suggestion
                            st.rerun()

            exp = st.session_state.get("experiment_suggestion")
            if exp:
                with st.expander("📋 最小可行投放实验建议", expanded=True):
                    st.write("**是否建议试投:**", "✅ 是" if exp.should_test else "❌ 否")
                    st.write("**建议人群/地区:**", exp.suggested_segment)
                    st.write("**渠道类型:**", exp.suggested_channel_type)
                    st.write("**预算区间:**", exp.budget_range)
                    st.write("**门禁指标:**", exp.gate_metrics)
                    st.write("**止损条件:**", exp.stop_loss_condition)
                    st.write("**实验目标:**", exp.experiment_goal)

            st.divider()

            md = export_markdown(rows)
            csv_str = export_csv(rows)
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                st.download_button("下载 Markdown", md, file_name="creative_review.md", mime="text/markdown")
            with c2:
                st.download_button("下载 CSV", csv_str, file_name="creative_review.csv", mime="text/csv")

            st.divider()
            with st.expander("查看变体详情"):
                for i, rw in enumerate(rows, 1):
                    st.markdown(
                        f"**变体 {i}: {rw.variant.title}** — `{rw.verdict}` | fuse={rw.fuse_level} | white_traffic_risk={rw.white_traffic_risk_final}"
                    )
                    if rw.review.error:
                        st.error(rw.review.error)
                    else:
                        st.write("**variant_id:**", rw.variant.variant_id)
                        st.write("**hook_type:**", rw.variant.hook_type)
                        w = rw.variant.who_why_now
                        if w and (w.who or w.why or w.why_now):
                            st.write("**who_why_now:**", f"who={w.who} | why={w.why} | why_now={w.why_now}")
                        st.write("**cta:**", rw.variant.cta)
                        if rw.variant.notes:
                            st.write("**notes:**", rw.variant.notes)

                        rf = rw.variant.risk_flags
                        if hasattr(rf, "policy_risk"):
                            st.caption(
                                f"risk_flags: policy={rf.policy_risk} | exaggeration={rf.exaggeration_risk} | white_traffic={rf.white_traffic_risk}"
                            )

                        if rw.variant.script and rw.variant.script.shots:
                            with st.expander("查看分镜 shots"):
                                for s in rw.variant.script.shots:
                                    vo = (s.voiceover or "")[:80] + ("..." if len(s.voiceover or "") > 80 else "")
                                    st.write(f"{s.t}s | {s.visual} | 字幕:{s.overlay_text} | 口播:{vo}")

                        freasons = rw.review._fuse_reasons_list() if hasattr(rw.review, "_fuse_reasons_list") else rw.review.fuse_reasons
                        if freasons:
                            st.warning("fuse_reasons: " + ", ".join(freasons))

                        rfix = rw.review.required_fixes_flat if hasattr(rw.review, "required_fixes_flat") else rw.review.fixes
                        if rfix:
                            st.warning("required_fixes: " + " | ".join(rfix[:5]))

                        if rw.review.key_reasons:
                            st.write("key_reasons:", rw.review.key_reasons)
                        elif rw.review.risks:
                            st.write("风险:", rw.review.risks)

                        if rw.review.fixes:
                            st.write("修改建议:", rw.review.fixes)

                    st.write("---")


if __name__ == "__main__":
    main()
