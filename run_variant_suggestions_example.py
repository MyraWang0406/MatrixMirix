"""
变体优化建议示例：基于 ElementScore + GateResult，输出可执行的 next_variant_suggestions。
"""
import json
from pathlib import Path

from element_scores import ElementScore, compute_element_scores
from eval_schemas import Variant
from explore_gate import evaluate_explore_gate
from simulate_metrics import SimulatedMetrics
from variant_suggestions import next_variant_suggestions

SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_PATH = SAMPLES_DIR / "variant_suggestions_example_output.json"


def main() -> None:
    # 1. 加载 Variants + metrics
    with open(SAMPLES_DIR / "eval_variants.json", "r", encoding="utf-8") as f:
        variants = [Variant.model_validate(v) for v in json.load(f)]
    with open(SAMPLES_DIR / "simulated_metrics_example.json", "r", encoding="utf-8") as f:
        metrics = [SimulatedMetrics.model_validate(m) for m in json.load(f)["metrics"]]

    # 2. 计算 ElementScore
    element_scores = compute_element_scores(
        variant_metrics=metrics,
        variants=variants,
    )

    # 3. Explore Gate 结果（按 Android，因为 Android 全 FAIL）
    baseline_list = [m for m in metrics if m.baseline]
    variant_list = [m for m in metrics if not m.baseline]
    gate_result = evaluate_explore_gate(
        variant_metrics=variant_list,
        baseline_metrics=baseline_list,
        context={"country": "CN", "os": "Android", "objective": "install", "segment": "18-30手游"},
    )

    # 4. 生成建议（结构化）
    suggestions = next_variant_suggestions(
        element_scores=element_scores,
        gate_result=gate_result,
        max_suggestions=3,
        variant_metrics=metrics,
        variants=variants,
    )

    # 5. 输出
    output = {
        "description": "变体优化建议 - 结构化输出",
        "gate_context": {"os": "Android", "gate_status": gate_result.gate_status},
        "suggestions": [s.model_dump() if hasattr(s, "model_dump") else str(s) for s in suggestions],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("变体优化建议（结构化）")
    print("=" * 70)
    for i, s in enumerate(suggestions, 1):
        if hasattr(s, "change_layer"):
            print(f"\n实验单 {i}：{getattr(s, 'suggestion_type', '')} | 层级={s.change_layer} | changed_field={getattr(s, 'changed_field', '')}")
            print(f"  delta_desc: {getattr(s, 'delta_desc', '')}")
            print(f"  当前: {s.current_value}")
            print(f"  候选: {s.candidate_alternatives}")
            print(f"  依据: {s.rationale}")
            print(f"  预期指标: {getattr(s, 'expected_metric', '')}")
        else:
            print(f"\n建议 {i}：{s}")
    print(f"\n已写入: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
