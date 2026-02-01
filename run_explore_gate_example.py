"""
Explore Gate 示例：加载模拟数据，按 iOS/Android 分别评测，输出 gate 结果。
"""
import json
from pathlib import Path

from explore_gate import (
    ExploreGateConfig,
    ExploreGateResult,
    evaluate_explore_gate,
)
from simulate_metrics import SimulatedMetrics

SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_PATH = SAMPLES_DIR / "explore_gate_example_output.json"


def main() -> None:
    # 1. 加载模拟数据
    with open(SAMPLES_DIR / "simulated_metrics_example.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = [SimulatedMetrics.model_validate(m) for m in data["metrics"]]
    baseline_list = [m for m in metrics if m.baseline]
    variant_list = [m for m in metrics if not m.baseline]

    # 2. 评测配置
    config = ExploreGateConfig(
        min_spend=500,
        min_better_metrics=2,
        improvement_pct=0,
    )

    # 3. 按 iOS 评测（context 含 motivation_bucket 以在 reasons 中引用）
    context_ios = {
        "country": "CN",
        "os": "iOS",
        "objective": "install",
        "segment": "18-30岁手游玩家，MOBA/竞技偏好",
        "motivation_bucket": "胜负欲",
    }
    result_ios = evaluate_explore_gate(
        variant_metrics=variant_list,
        baseline_metrics=baseline_list,
        context=context_ios,
        config=config,
    )

    # 4. 按 Android 评测
    context_android = {
        "country": "CN",
        "os": "Android",
        "objective": "install",
        "segment": "18-30岁手游玩家，MOBA/竞技偏好",
        "motivation_bucket": "胜负欲",
    }
    result_android = evaluate_explore_gate(
        variant_metrics=variant_list,
        baseline_metrics=baseline_list,
        context=context_android,
        config=config,
    )

    # 5. 输出
    output = {
        "description": "Explore Gate 评测示例",
        "input_summary": {
            "baseline_variant": "v001",
            "test_variants": ["v002", "v003"],
            "context_ios": context_ios,
            "context_android": context_android,
        },
        "config": {
            "min_spend": config.min_spend,
            "min_better_metrics": config.min_better_metrics,
        },
        "results": {
            "iOS": result_ios.model_dump(),
            "Android": result_android.model_dump(),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Explore Gate 示例输入 & 输出")
    print("=" * 60)
    print("\n【示例输入】")
    print("baseline: v001 (iOS/Android 各一条)")
    print("variants: v002, v003")
    print("context: CN, install, 18-30岁手游玩家")
    print("config: min_spend=500, min_better_metrics=2")
    print("\n【iOS 评测结果】")
    print(f"  gate_status: {result_ios.gate_status}")
    print(f"  eligible_variants: {result_ios.eligible_variants}")
    for r in result_ios.reasons:
        print(f"  - {r}")
    print("\n【Android 评测结果】")
    print(f"  gate_status: {result_android.gate_status}")
    print(f"  eligible_variants: {result_android.eligible_variants}")
    for r in result_android.reasons:
        print(f"  - {r}")
    print(f"\n已写入: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
