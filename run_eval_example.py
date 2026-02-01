"""
评测系统原型示例：加载 StrategyCard、3 个 Variant，并输出每个 Variant 的 ElementTag 拆解。
不调用任何模型 API。
"""
import json
from pathlib import Path

from eval_schemas import (
    StrategyCard,
    Variant,
    decompose_variant_to_element_tags,
)

SAMPLES_DIR = Path(__file__).parent / "samples"


def main() -> None:
    # 1. 加载 StrategyCard
    with open(SAMPLES_DIR / "eval_strategy_card.json", "r", encoding="utf-8") as f:
        card_data = json.load(f)
    card = StrategyCard.model_validate(card_data)
    print("=" * 60)
    print("1. StrategyCard 示例")
    print("=" * 60)
    print(json.dumps(card.model_dump(), ensure_ascii=False, indent=2))
    print()

    # 2. 加载 3 个 Variant
    with open(SAMPLES_DIR / "eval_variants.json", "r", encoding="utf-8") as f:
        variants_data = json.load(f)
    variants = [Variant.model_validate(v) for v in variants_data]
    print("=" * 60)
    print("2. Variant 示例（3 个）")
    print("=" * 60)
    for v in variants:
        print(f"\n--- {v.variant_id} ---")
        print(json.dumps(v.model_dump(), ensure_ascii=False, indent=2))
    print()

    # 3. 每个 Variant 拆解后的 ElementTag 列表
    print("=" * 60)
    print("3. 每个 Variant 拆解后的 ElementTag 列表")
    print("=" * 60)
    for v in variants:
        tags = decompose_variant_to_element_tags(v)
        print(f"\n--- {v.variant_id} 的 ElementTags ---")
        for t in tags:
            print(f"  [{t.element_type}] {t.element_value}")
        print(f"  共 {len(tags)} 个元素")


if __name__ == "__main__":
    main()
