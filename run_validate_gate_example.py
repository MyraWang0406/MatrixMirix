"""
Validate Gate 示例：构造 ≥2 时间窗口 + 轻扩人群的模拟数据，输出评测结果。
"""
import json
from pathlib import Path

from validate_gate import (
    ValidateGateConfig,
    ValidateGateResult,
    WindowMetrics,
    evaluate_validate_gate,
)

SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_PATH = SAMPLES_DIR / "validate_gate_example_output.json"


def main() -> None:
    # 1. 模拟 ≥2 时间窗口的 metrics（同一结构组合）
    windowed = [
        WindowMetrics(
            window_id="T1",
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
            window_id="T2",
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
        WindowMetrics(
            window_id="T3",
            impressions=60000,
            clicks=900,
            installs=2040,
            spend=6528,
            early_events=1180,
            early_revenue=460,
            ipm=34.0,
            cpi=3.2,
            early_roas=0.07,
        ),
    ]

    # 2. 轻扩人群 variant 的 metrics
    light_expansion = WindowMetrics(
        window_id="T2_light_expand",
        impressions=20000,
        clicks=280,
        installs=520,
        spend=1872,
        early_events=280,
        early_revenue=112,
        ipm=26.0,
        cpi=3.6,
        early_roas=0.06,
    )

    # 3. 评测
    result = evaluate_validate_gate(
        windowed_metrics=windowed,
        light_expansion_metrics=light_expansion,
        config=ValidateGateConfig(
            ipm_cv_max=0.35,
            ipm_drop_max_pct=0.30,
            cpi_increase_max_pct=0.25,
            light_expansion_ipm_drop_max=0.20,
            light_expansion_cpi_increase_max=0.30,
        ),
    )

    # 4. 再跑一个 PASS 场景（轻扩表现尚可）
    windowed_pass = [
        WindowMetrics(window_id="T1", impressions=50000, clicks=800, installs=2000,
                      spend=6000, early_events=1200, early_revenue=480,
                      ipm=40.0, cpi=3.0, early_roas=0.08),
        WindowMetrics(window_id="T2", impressions=55000, clicks=880, installs=2200,
                      spend=6600, early_events=1320, early_revenue=528,
                      ipm=40.0, cpi=3.0, early_roas=0.08),
    ]
    light_expansion_pass = WindowMetrics(
        window_id="T2_light", impressions=20000, clicks=288, installs=720,
        spend=2160, early_events=430, early_revenue=172,
        ipm=36.0, cpi=3.0, early_roas=0.08,
    )
    result_pass = evaluate_validate_gate(
        windowed_metrics=windowed_pass,
        light_expansion_metrics=light_expansion_pass,
    )

    # 5. 输出
    output = {
        "description": "Validate Gate 评测示例",
        "scenarios": {
            "FAIL_轻扩劣化": {
                "input": {
                    "windowed_metrics": [w.model_dump() for w in windowed],
                    "light_expansion_metrics": light_expansion.model_dump(),
                },
                "result": result.model_dump(),
            },
            "PASS_稳定加量": {
                "input": {
                    "windowed_metrics": [w.model_dump() for w in windowed_pass],
                    "light_expansion_metrics": light_expansion_pass.model_dump(),
                },
                "result": result_pass.model_dump(),
            },
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Validate Gate 示例结果")
    print("=" * 60)
    for name, data in output["scenarios"].items():
        r = data["result"]
        print(f"\n【{name}】")
        print(f"  validate_status: {r['validate_status']}")
        print("  risk_notes:", r["risk_notes"])
        print("  scale_recommendation:", r["scale_recommendation"])
    print(f"\n已写入: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
