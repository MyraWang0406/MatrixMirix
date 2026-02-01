"""
生成模拟投放数据示例：加载 3 个 Variant，为 iOS/Android 各生成指标，
包含 1 个 baseline 变体。
"""
import json
from pathlib import Path

from eval_schemas import StrategyCard, Variant
from simulate_metrics import SimulatedMetrics, simulate_metrics

SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_PATH = SAMPLES_DIR / "simulated_metrics_example.json"


def main() -> None:
    with open(SAMPLES_DIR / "eval_strategy_card.json", "r", encoding="utf-8") as f:
        card = StrategyCard.model_validate(json.load(f))
    with open(SAMPLES_DIR / "eval_variants.json", "r", encoding="utf-8") as f:
        variants = [Variant.model_validate(v) for v in json.load(f)]

    mb = card.motivation_bucket
    vert = getattr(card, "vertical", "game") or "game"
    results: list[dict] = []

    # v001 作为 baseline（历史对照组）
    results.append(
        simulate_metrics(variants[0], "iOS", baseline=True, motivation_bucket=mb, vertical=vert).model_dump()
    )
    results.append(
        simulate_metrics(variants[0], "Android", baseline=True, motivation_bucket=mb, vertical=vert).model_dump()
    )

    # v002、v003 为测试变体
    for v in variants[1:]:
        results.append(simulate_metrics(v, "iOS", baseline=False, motivation_bucket=mb, vertical=vert).model_dump())
        results.append(simulate_metrics(v, "Android", baseline=False, motivation_bucket=mb, vertical=vert).model_dump())

    output = {
        "description": "TikTok 投放评测原型 - 模拟数据示例",
        "metrics": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"已写入: {OUTPUT_PATH}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
