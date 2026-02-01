"""
OFAAT（One Factor At A Time）变体生成器。
从 hook/sell_point/cta/asset 候选池生成变体，每个变体仅改一个元素。
"""
from __future__ import annotations

from typing import Any

from eval_schemas import AssetVariables, Variant


def generate_ofaat_variants(
    parent_card_id: str,
    hook_types: list[str],
    sell_points: list[str],
    ctas: list[str],
    *,
    n: int = 12,
    asset_pool: dict[str, list[str]] | None = None,
) -> list[Variant]:
    """
    OFAAT 生成 N 个变体，每个变体仅改一个元素（hook / sell_point / cta / 素材变量）。

    输入：
    - parent_card_id: 所属 StrategyCard 的 card_id
    - hook_types: Hook 候选池
    - sell_points: sell_point 候选池
    - ctas: CTA 候选池
    - n: 生成数量，默认 12
    - asset_pool: 可选，{"subtitle_template": [...], "bgm": [...], "rhythm": [...], "shot_template": [...]}

    规则：基线为各池首个；后续变体轮流改 hook/sell_point/cta/asset 中一个，保证 OFAAT。
    """
    hook_types = [h.strip() for h in hook_types if h and str(h).strip()]
    sell_points = [s.strip() for s in sell_points if s and str(s).strip()]
    ctas = [c.strip() for c in ctas if c and str(c).strip()]

    if not hook_types:
        hook_types = [""]
    if not sell_points:
        sell_points = [""]
    if not ctas:
        ctas = [""]

    asset_pool = asset_pool or {}
    ap = {
        "subtitle_template": [x.strip() for x in asset_pool.get("subtitle_template", []) if x],
        "bgm": [x.strip() for x in asset_pool.get("bgm", []) if x],
        "rhythm": [x.strip() for x in asset_pool.get("rhythm", []) if x],
        "shot_template": [x.strip() for x in asset_pool.get("shot_template", []) if x],
    }
    # 默认资产（取各池首个，否则用默认）
    default_asset = AssetVariables(
        subtitle_template=ap["subtitle_template"][0] if ap["subtitle_template"] else "大字+高亮关键词",
        bgm=ap["bgm"][0] if ap["bgm"] else "电子/节奏感",
        rhythm=ap["rhythm"][0] if ap["rhythm"] else "快切，3秒一镜",
        shot_template=ap["shot_template"][0] if ap["shot_template"] else "游戏画面+字幕叠加",
    )

    baseline_hook = hook_types[0]
    baseline_sell = sell_points[0]
    baseline_cta = ctas[0]

    variants: list[Variant] = []
    vid = 1

    def make_asset(
        sub: str | None = None,
        bgm: str | None = None,
        rh: str | None = None,
        shot: str | None = None,
    ) -> AssetVariables:
        return AssetVariables(
            subtitle_template=sub if sub is not None else default_asset.subtitle_template,
            bgm=bgm if bgm is not None else default_asset.bgm,
            rhythm=rh if rh is not None else default_asset.rhythm,
            shot_template=shot if shot is not None else default_asset.shot_template,
        )

    # 基线（v001）
    variants.append(
        Variant(
            variant_id="v001",
            parent_card_id=parent_card_id,
            hook_type=baseline_hook,
            sell_point=baseline_sell,
            cta_type=baseline_cta,
            expression_template="5镜头",
            asset_variables=default_asset,
            why_you_expression=baseline_sell,
            why_now_expression=baseline_sell,
            changed_field="",
            delta_desc="基线",
        )
    )
    vid += 1

    # OFAAT：轮流改 hook / sell_point / cta / asset
    hook_idx = 1
    sell_idx = 1
    cta_idx = 1
    asset_keys = [k for k, v in ap.items() if len(v) > 1]
    asset_key_idx = 0
    asset_val_idx = {k: 1 for k in asset_keys}

    while len(variants) < n:
        variant_id = f"v{vid:03d}"

        # 轮转：优先 hook -> sell_point -> cta -> asset
        if hook_idx < len(hook_types):
            new_hook = hook_types[hook_idx]
            v = Variant(
                variant_id=variant_id,
                parent_card_id=parent_card_id,
                hook_type=new_hook,
                sell_point=baseline_sell,
                cta_type=baseline_cta,
                expression_template="5镜头",
                asset_variables=default_asset,
                why_you_expression=baseline_sell,
                why_now_expression=baseline_sell,
                changed_field="hook_type",
                delta_desc=f"Hook: {baseline_hook} -> {new_hook}",
            )
            variants.append(v)
            hook_idx += 1
        elif sell_idx < len(sell_points):
            new_sell = sell_points[sell_idx]
            v = Variant(
                variant_id=variant_id,
                parent_card_id=parent_card_id,
                hook_type=baseline_hook,
                sell_point=new_sell,
                cta_type=baseline_cta,
                expression_template="5镜头",
                asset_variables=default_asset,
                why_you_expression=new_sell,
                why_now_expression=new_sell,
                changed_field="sell_point",
                delta_desc=f"卖点: {baseline_sell[:20]}{'…' if len(baseline_sell) > 20 else ''} -> {new_sell[:20]}{'…' if len(new_sell) > 20 else ''}",
            )
            variants.append(v)
            sell_idx += 1
        elif cta_idx < len(ctas):
            new_cta = ctas[cta_idx]
            v = Variant(
                variant_id=variant_id,
                parent_card_id=parent_card_id,
                hook_type=baseline_hook,
                sell_point=baseline_sell,
                cta_type=new_cta,
                expression_template="5镜头",
                asset_variables=default_asset,
                why_you_expression=baseline_sell,
                why_now_expression=baseline_sell,
                changed_field="cta",
                delta_desc=f"CTA: {baseline_cta} -> {new_cta}",
            )
            variants.append(v)
            cta_idx += 1
        elif asset_keys:
            any_added = False
            for _ in range(len(asset_keys)):
                key = asset_keys[asset_key_idx % len(asset_keys)]
                vals = ap[key]
                idx = asset_val_idx.get(key, 1)
                if idx < len(vals):
                    val = vals[idx]
                    kw: dict[str, str] = {}
                    if key == "subtitle_template":
                        kw["sub"] = val
                    elif key == "bgm":
                        kw["bgm"] = val
                    elif key == "rhythm":
                        kw["rh"] = val
                    elif key == "shot_template":
                        kw["shot"] = val
                    asset = make_asset(**kw) if kw else default_asset
                    old_val = getattr(default_asset, key, "") if hasattr(default_asset, key) else (ap[key][0] if ap[key] else "")
                    v = Variant(
                        variant_id=variant_id,
                        parent_card_id=parent_card_id,
                        hook_type=baseline_hook,
                        sell_point=baseline_sell,
                        cta_type=baseline_cta,
                        expression_template="5镜头",
                        asset_variables=asset,
                        why_you_expression=baseline_sell,
                        why_now_expression=baseline_sell,
                        changed_field="asset_var",
                        delta_desc=f"素材({key}): {str(old_val)[:16]}{'…' if len(str(old_val)) > 16 else ''} -> {str(val)[:16]}{'…' if len(str(val)) > 16 else ''}",
                    )
                    variants.append(v)
                    asset_val_idx[key] = idx + 1
                    asset_key_idx += 1
                    any_added = True
                    break
                asset_key_idx += 1
            if not any_added:
                break
        else:
            break

        vid += 1

    return variants[:n]
