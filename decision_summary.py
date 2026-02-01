"""
30 秒决策结论：综合 iOS/Android Explore + Validate 状态。
无 Streamlit 依赖，可单独测试。
"""
from __future__ import annotations

MIN_SAMPLES = 6
MIN_WINDOWS = 3
IPM_CV_THRESHOLD_FOR_SCALE = 0.05
DEFAULT_SCALE_UP_STEP = "20%"


def compute_decision_summary(results: dict) -> dict:
    """
    30 秒决策结论：综合 iOS/Android Explore + Validate 状态。
    返回: status(red/yellow/green), status_text, reason, risk, next_step, insufficient
    """
    explore_ios = results.get("explore_ios")
    explore_android = results.get("explore_android")
    validate_result = results.get("validate_result")
    metrics = results.get("metrics", [])
    scale_up_step = DEFAULT_SCALE_UP_STEP

    n_samples = len([m for m in metrics if not m.baseline])
    detail_rows = getattr(validate_result, "detail_rows", None) or []
    n_windows = len(detail_rows)
    insufficient = n_samples < MIN_SAMPLES or n_windows < MIN_WINDOWS

    exp_ios_pass = explore_ios.gate_status == "PASS" if explore_ios else False
    exp_android_pass = explore_android.gate_status == "PASS" if explore_android else False
    val_pass = validate_result.validate_status == "PASS" if validate_result else False
    sm = getattr(validate_result, "stability_metrics", None)
    ipm_cv = getattr(sm, "ipm_cv", 1.0) if sm else 1.0

    # 原因
    reason_parts = []
    if exp_ios_pass:
        reason_parts.append("iOS Explore PASS")
    else:
        reason_parts.append("iOS Explore FAIL")
    if exp_android_pass:
        reason_parts.append("Android Explore PASS")
    else:
        reason_parts.append("Android Explore FAIL")
    if val_pass:
        reason_parts.append("Validate PASS")
    else:
        reason_parts.append("Validate FAIL")
    if insufficient:
        reason_parts.append("样本不足（n<6 或窗口<3）")
    reason_str = "；".join(reason_parts)

    # 风险
    risk_parts = list(getattr(validate_result, "risk_notes", None) or [])[:2]
    baseline_list = [m for m in metrics if m.baseline]
    variant_list = [m for m in metrics if not m.baseline]
    if baseline_list and variant_list and len(baseline_list) > 0:
        bl_cpi = sum(m.cpi for m in baseline_list) / len(baseline_list)
        var_cpi = sum(m.cpi for m in variant_list) / len(variant_list)
        if bl_cpi > 0:
            cpi_delta = (var_cpi - bl_cpi) / bl_cpi
            if cpi_delta > 0.05:
                risk_parts.append(f"CPI +{cpi_delta:.1%} 高于 baseline")
    risk_str = "；".join(risk_parts) if risk_parts else "暂无显著风险"

    # 状态与下一步
    if insufficient:
        status = "yellow"
        status_text = f"🟡 小步复测({scale_up_step})"
        reason_str += "（样本不足，建议补足数据后复测）"
        next_step = "复测"
    elif val_pass and ipm_cv < IPM_CV_THRESHOLD_FOR_SCALE:
        status = "green"
        status_text = f"🟢 建议放量({scale_up_step})"
        next_step = "放量"
    elif not val_pass:
        status = "red"
        status_text = "🔴 不建议放量"
        next_step = "复测"
    else:
        status = "yellow"
        status_text = f"🟡 小步复测({scale_up_step})"
        next_step = "复测"

    return {
        "status": status,
        "status_text": status_text,
        "reason": reason_str,
        "risk": risk_str,
        "next_step": next_step,
        "insufficient": insufficient,
    }
