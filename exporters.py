"""导出 Markdown / CSV"""
from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import VariantWithReview


def export_markdown(rows: list["VariantWithReview"]) -> str:
    """导出为 Markdown 表格"""
    lines = [
        "# 创意变体评审报告",
        "",
        "| # | headline | decision | fuse_level | wt_risk_final | clarity | hook_strength | compliance_safety | expected_test_value | 总结 |",
        "|---|------|----------|--------------------------|------------|--------|----------|------|--------|----------|----------|------|",
    ]

    for i, rw in enumerate(rows, 1):
        s = rw.review.scores
        summary = (rw.review.error or rw.review.overall_summary or "-")[:30]
        if len(rw.review.error or rw.review.overall_summary or "") > 30:
            summary += "..."
        lines.append(
            f"| {i} | {_esc(rw.variant.variant_id or rw.variant.headline or rw.variant.title)} | {rw.verdict} | {rw.fuse_level} | {rw.white_traffic_risk_final} | "
            f"{s.clarity} | {s.hook_strength} | {s.compliance_safety} | {s.expected_test_value} | {_esc(summary)} |"
        )

    lines.append("")
    lines.append("## 详情")
    lines.append("")

    for i, rw in enumerate(rows, 1):
        lines.append(f"### 变体 {i}: {_esc(rw.variant.variant_id or rw.variant.headline or rw.variant.title)} — {rw.verdict} | fuse={rw.fuse_level} | white_traffic_risk={rw.white_traffic_risk_final}")
        lines.append("")
        freasons = getattr(rw.review, "_fuse_reasons_list", lambda: rw.review.fuse_reasons)()
        if freasons:
            lines.append("**fuse_reasons:** " + "; ".join(freasons))
            lines.append("")
        rfix = getattr(rw.review, "required_fixes_flat", lambda: rw.review.fixes or [])()
        if rfix:
            lines.append("**required_fixes:** " + " | ".join(rfix))
            lines.append("")
        if rw.review.error:
            lines.append("**错误:** " + _esc(rw.review.error))
            lines.append("")
        else:
            lines.append("**variant_id:** " + _esc(rw.variant.variant_id))
            lines.append("")
            lines.append("**hook_type:** " + _esc(rw.variant.hook_type))
            w = rw.variant.who_why_now
            if w:
                lines.append("**who_why_now:** who=" + _esc(w.who) + " | why=" + _esc(w.why) + " | why_now=" + _esc(w.why_now))
            lines.append("")
            lines.append("**cta:** " + _esc(rw.variant.cta))
            if rw.variant.notes:
                lines.append("")
                lines.append("**notes:** " + _esc(rw.variant.notes))
            rf = rw.variant.risk_flags
            if hasattr(rf, "policy_risk"):
                lines.append("**risk_flags:** policy=" + str(rf.policy_risk) + " | exaggeration=" + str(rf.exaggeration_risk) + " | white_traffic=" + str(rf.white_traffic_risk))
            if rw.variant.script and rw.variant.script.shots:
                lines.append("")
                lines.append("**shots:**")
                for s in rw.variant.script.shots:
                    lines.append(f"  - {s.t}s: {_esc(s.visual)} | 字幕:{_esc(s.overlay_text)} | 口播:{_esc((s.voiceover or '')[:100])}")
            lines.append("")
            if rw.review.key_reasons:
                lines.append("**key_reasons:** " + "; ".join(rw.review.key_reasons))
            elif rw.review.risks:
                lines.append("**风险:** " + "; ".join(rw.review.risks))
                lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _rf_to_str(rf) -> str:
    """risk_flags 转为字符串"""
    if hasattr(rf, "policy_risk"):
        return f"policy={rf.policy_risk}|exag={rf.exaggeration_risk}|wt={rf.white_traffic_risk}"
    if isinstance(rf, list):
        return "; ".join(rf)
    return str(rf) if rf else ""


def _esc(s: str) -> str:
    """转义 Markdown 表格中的特殊字符"""
    if not s:
        return "-"
    return str(s).replace("|", "\\|").replace("\n", " ")


def export_csv(rows: list["VariantWithReview"]) -> str:
    """导出为 CSV"""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "序号", "headline", "hook_type", "decision", "fuse_level", "white_traffic_risk_final",
        "clarity", "hook_strength", "compliance_safety", "expected_test_value",
        "core_message", "cta", "risk_flags", "fuse_reasons", "required_fixes", "风险", "修改建议", "总结", "错误",
    ])
    for i, rw in enumerate(rows, 1):
        s = rw.review.scores
        risks = "; ".join(rw.review.risks) if rw.review.risks else ""
        fixes = "; ".join(rw.review.fixes) if rw.review.fixes else ""
        freasons = getattr(rw.review, "_fuse_reasons_list", lambda: rw.review.fuse_reasons or [])()
        fuse_reasons = "; ".join(freasons) if freasons else ""
        rfix = getattr(rw.review, "required_fixes_flat", lambda: rw.review.fixes or [])()
        required_fixes = " | ".join(rfix) if rfix else ""
        w.writerow([
            i,
            rw.variant.variant_id or rw.variant.headline or rw.variant.title,
            rw.variant.hook_type or "",
            rw.verdict,
            rw.fuse_level,
            rw.white_traffic_risk_final,
            s.clarity,
            s.hook_strength,
            s.compliance_safety,
            s.expected_test_value,
            (rw.variant.core_message or rw.variant.script_15s or "")[:500],
            rw.variant.cta or rw.variant.cta_text,
            (_rf_to_str(rw.variant.risk_flags)),
            fuse_reasons,
            required_fixes,
            risks,
            fixes,
            rw.review.overall_summary or "",
            rw.review.error or "",
        ])
    return buf.getvalue()
