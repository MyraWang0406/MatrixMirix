"""
元素级贡献分析示例：加载 Variant + 模拟 metrics，计算 ElementScore 表。
"""
import json
from pathlib import Path

from element_scores import ElementScore, compute_element_scores
from eval_schemas import Variant
from simulate_metrics import SimulatedMetrics

SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_PATH = SAMPLES_DIR / "element_scores_example_output.json"


def main() -> None:
    # 1. 加载 Variants
    with open(SAMPLES_DIR / "eval_variants.json", "r", encoding="utf-8") as f:
        variants_data = json.load(f)
    variants = [Variant.model_validate(v) for v in variants_data]

    # 2. 加载 metrics（含 baseline，都参与分析）
    with open(SAMPLES_DIR / "simulated_metrics_example.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = [SimulatedMetrics.model_validate(m) for m in data["metrics"]]

    # 3. 计算 ElementScore
    scores = compute_element_scores(
        variant_metrics=metrics,
        variants=variants,
        min_sample_size=2,
    )

    # 4. 输出
    output = {
        "description": "元素级贡献分析 - ElementScore 表",
        "card": "sc_game_install_001",
        "element_scores": [s.model_dump() for s in scores],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 表格打印
    print("=" * 80)
    print("ElementScore 表")
    print("=" * 80)
    print(f"{'element_type':<12} {'element_value':<35} {'IPM_delta':>10} {'CPI_delta':>10} {'n':>4} {'stable':>6}")
    print("-" * 80)
    for s in sorted(scores, key=lambda x: (x.element_type, x.element_value)):
        ev_short = (s.element_value[:32] + "…") if len(s.element_value) > 33 else s.element_value
        print(
            f"{s.element_type:<12} {ev_short:<35} "
            f"{s.avg_IPM_delta_vs_card_mean:>+10.2f} {s.avg_CPI_delta_vs_card_mean:>+10.2f} "
            f"{s.sample_size:>4} {str(s.stability_flag):>6}"
        )
    print(f"\n已写入: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
