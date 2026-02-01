"""
compute_decision_summary 示例：3 种模拟输入与输出。
运行：python run_decision_summary_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_gate import ExploreGateResult
from validate_gate import ValidateDetailRow, ValidateGateResult, ValidateStabilityMetrics

from decision_summary import compute_decision_summary


class _MockMetric:
    def __init__(self, os: str, baseline: bool, cpi: float):
        self.os = os
        self.baseline = baseline
        self.cpi = cpi


def _detail_rows(*ids: str):
    return [ValidateDetailRow(window_id=w) for w in ids]


# ---------- 示例 1：PASS，建议放量 ----------
results_pass = {
    "explore_ios": ExploreGateResult(
        gate_status="PASS",
        reasons=["≥2 指标超 baseline"],
        eligible_variants=["v1", "v2"],
        variant_details={"v1": "PASS", "v2": "PASS"},
    ),
    "explore_android": ExploreGateResult(
        gate_status="PASS",
        reasons=["≥2 指标超 baseline"],
        eligible_variants=["v1", "v2"],
        variant_details={"v1": "PASS", "v2": "PASS"},
    ),
    "validate_result": ValidateGateResult(
        validate_status="PASS",
        risk_notes=[],
        detail_rows=_detail_rows("window_1", "window_2", "expand_segment"),
        stability_metrics=ValidateStabilityMetrics(
            ipm_cv=0.03, ipm_drop_pct=5, cpi_increase_pct=8, learning_iterations=0
        ),
    ),
    "metrics": (
        [_MockMetric("iOS", True, 3.0), _MockMetric("Android", True, 3.0)]
        + [_MockMetric("iOS", False, 2.8), _MockMetric("Android", False, 2.9)] * 4
    ),
    "variants": [{}] * 5,
}
out1 = compute_decision_summary(results_pass)
print("=== 示例 1：PASS，建议放量 ===")
print(out1)
# 期望: status=green, status_text="🟢 建议放量(20%)", next_step="放量"

# ---------- 示例 2：FAIL，不建议放量 ----------
results_fail = {
    "explore_ios": ExploreGateResult(gate_status="PASS", reasons=[], eligible_variants=["v1"], variant_details={"v1": "PASS"}),
    "explore_android": ExploreGateResult(gate_status="FAIL", reasons=["≥2 指标未达 baseline"], eligible_variants=[], variant_details={"v1": "FAIL"}),
    "validate_result": ValidateGateResult(
        validate_status="FAIL",
        risk_notes=["IPM 波动过大，结构稳定性存疑", "CPI 回撤，成本抬升明显"],
        detail_rows=_detail_rows("window_1", "window_2", "expand_segment"),
        stability_metrics=ValidateStabilityMetrics(ipm_cv=0.40, ipm_drop_pct=25, cpi_increase_pct=28, learning_iterations=2),
    ),
    "metrics": (
        [_MockMetric("iOS", True, 3.0), _MockMetric("Android", True, 3.0)]
        + [_MockMetric("iOS", False, 3.5), _MockMetric("Android", False, 3.6)] * 4
    ),
    "variants": [{}] * 5,
}
out2 = compute_decision_summary(results_fail)
print("\n=== 示例 2：FAIL，不建议放量 ===")
print(out2)
# 期望: status=red, status_text="🔴 不建议放量", next_step="复测"

# ---------- 示例 3：样本不足 ----------
results_insufficient = {
    "explore_ios": ExploreGateResult(gate_status="PASS", reasons=[], eligible_variants=["v1"], variant_details={"v1": "PASS"}),
    "explore_android": ExploreGateResult(gate_status="PASS", reasons=[], eligible_variants=["v1"], variant_details={"v1": "PASS"}),
    "validate_result": ValidateGateResult(
        validate_status="PASS",
        risk_notes=[],
        detail_rows=_detail_rows("window_1"),  # 仅 1 窗口 < 3
        stability_metrics=ValidateStabilityMetrics(ipm_cv=0.02, ipm_drop_pct=2, cpi_increase_pct=3, learning_iterations=0),
    ),
    "metrics": (
        [_MockMetric("iOS", True, 3.0), _MockMetric("Android", True, 3.0)]
        + [_MockMetric("iOS", False, 2.9), _MockMetric("Android", False, 2.9)]  # 仅 2 个 variant-OS < 6
    ),
    "variants": [{}] * 2,
}
out3 = compute_decision_summary(results_insufficient)
print("\n=== 示例 3：样本不足 ===")
print(out3)
# 期望: status=yellow, insufficient=True, next_step="复测"
