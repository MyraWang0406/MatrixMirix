"""
评测集批量模拟数据生成：50-100 张 StrategyCard，每张卡状态流转与 Explore/Validate 汇总。
bucket + phrase 分离，保证 pydantic 校验不崩。
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from eval_schemas import StrategyCard, Variant

from explore_gate import evaluate_explore_gate
from ofaat_generator import generate_ofaat_variants
from simulate_metrics import simulate_metrics
from validate_gate import WindowMetrics, evaluate_validate_gate
from vertical_config import (
    get_corpus,
    get_why_now_buckets,
    get_why_now_phrases,
    get_why_you_buckets,
    get_why_you_phrases,
)

CARD_STATUSES = ("未测", "探索中", "进验证", "可放量")

_MOTIVATION_ALLOWED = frozenset(["省钱", "体验", "社交", "成就感", "收集", "爽感", "品质", "口碑", "其他"])
_WHY_YOU_BUCKETS_ALLOWED = frozenset(["更省钱", "更省事", "更强效果", "更匹配我", "更安全可信", "更好体验", "其他"])
_WHY_NOW_BUCKETS_ALLOWED = frozenset(["限时稀缺", "节点事件", "痛点升高", "机会出现", "社会驱动", "损失厌恶", "即时替代", "其他"])


def _seeded(seed: str) -> random.Random:
    h = hashlib.sha256(seed.encode()).hexdigest()
    return random.Random(int(h[:16], 16) % (2**32))


def _normalize_mb(mb: str) -> str:
    return mb if mb in _MOTIVATION_ALLOWED else "其他"


def _normalize_wyb(wyb: str) -> str:
    return wyb if wyb in _WHY_YOU_BUCKETS_ALLOWED else "其他"


def _normalize_wnb(wnb: str) -> str:
    return wnb if wnb in _WHY_NOW_BUCKETS_ALLOWED else "其他"


@dataclass
class CardEvalRecord:
    card: StrategyCard
    card_score: float
    status: str
    variants: list[Variant]
    explore_ios: Any
    explore_android: Any
    validate_result: Any | None = None
    window_metrics: list[WindowMetrics] = None
    expand_segment_metrics: WindowMetrics | None = None

    def __post_init__(self):
        if self.window_metrics is None:
            self.window_metrics = []


def generate_eval_set(
    n_cards: int = 75,
    variants_per_card: int = 12,
    *,
    status_dist: dict[str, float] | None = None,
) -> list[CardEvalRecord]:
    status_dist = status_dist or {"未测": 0.25, "探索中": 0.30, "进验证": 0.25, "可放量": 0.20}
    rng = _seeded("eval_set_v2")
    records: list[CardEvalRecord] = []

    wy_buckets = get_why_you_buckets()
    wn_buckets = get_why_now_buckets()

    for i in range(n_cards):
        cid = f"sc_{i+1:03d}"
        seed = f"card_{cid}"
        r = _seeded(seed)

        vert = "casual_game" if r.random() < 0.7 else "ecommerce"
        corp = get_corpus(vert)
        mb_pool = corp.get("motivation_bucket") or ["成就感", "爽感", "其他"]
        seg_pool = corp.get("segment") or ["18-45岁手游玩家"]

        mb_raw = r.choice(mb_pool) if mb_pool else "其他"
        mb = _normalize_mb(mb_raw)

        wy_phrases = get_why_you_phrases(vert)
        wn_phrases = get_why_now_phrases(vert)
        wyb_raw = r.choice(wy_buckets) if wy_buckets else "其他"
        wnb_raw = r.choice(wn_buckets) if wn_buckets else "其他"
        wyb = _normalize_wyb(wyb_raw)
        wnb = _normalize_wnb(wnb_raw)

        wy_ph_list = wy_phrases.get(wyb, [])
        wn_ph_list = wn_phrases.get(wnb, [])
        wy_phrase = r.choice(wy_ph_list) if wy_ph_list else wyb
        wn_phrase = r.choice(wn_ph_list) if wn_ph_list else wnb

        seg = r.choice(seg_pool) if seg_pool else "默认人群"
        obj = "purchase" if vert == "ecommerce" else "install"

        card = StrategyCard(
            card_id=cid,
            version="1.0",
            vertical=vert,
            country="CN",
            os="all",
            objective=obj,
            segment=seg,
            motivation_bucket=mb,
            why_you_bucket=wyb,
            why_you_phrase=wy_phrase,
            why_now_trigger_bucket=wnb,
            why_now_phrase=wn_phrase,
            why_you_key="other",
            why_you_label=wy_phrase,
            why_now_trigger=wn_phrase,
            root_cause_gap="",
        )

        hooks = corp.get("hook_type") or ["冲突", "利益前置", "社交"]
        sells = corp.get("sell_point") or ["上手快爽点前置", "福利多登录即送"]
        ctas = corp.get("cta") or ["立即下载", "领福利", "马上开玩"]
        vs = generate_ofaat_variants(
            cid,
            list(hooks)[:8],
            list(sells)[:8],
            list(ctas)[:5],
            n=variants_per_card,
        )

        metrics = []
        metrics.append(simulate_metrics(vs[0], "iOS", baseline=True, motivation_bucket=mb, vertical=vert))
        metrics.append(simulate_metrics(vs[0], "Android", baseline=True, motivation_bucket=mb, vertical=vert))
        for v in vs[1:]:
            metrics.append(simulate_metrics(v, "iOS", baseline=False, motivation_bucket=mb, vertical=vert))
            metrics.append(simulate_metrics(v, "Android", baseline=False, motivation_bucket=mb, vertical=vert))

        ctx = {"country": "CN", "objective": obj, "segment": seg, "motivation_bucket": mb}
        exp_ios = evaluate_explore_gate(
            [m for m in metrics if not m.baseline],
            [m for m in metrics if m.baseline],
            context={**ctx, "os": "iOS"},
        )
        exp_android = evaluate_explore_gate(
            [m for m in metrics if not m.baseline],
            [m for m in metrics if m.baseline],
            context={**ctx, "os": "Android"},
        )

        eligible = list(dict.fromkeys((exp_ios.eligible_variants or []) + (exp_android.eligible_variants or [])))
        base_score = min(100.0, 40.0 + len(eligible) * 4.0 + r.uniform(0, 25))
        card_score = round(base_score, 1)

        rv = rng.random()
        cum = 0.0
        status = "未测"
        for s, p in status_dist.items():
            cum += p
            if rv <= cum:
                status = s
                break

        window_metrics: list[WindowMetrics] = []
        expand_metrics: WindowMetrics | None = None
        validate_result = None

        if status in ("进验证", "可放量"):
            w1_ipm = metrics[0].ipm * (0.95 + r.uniform(0, 0.1))
            w1_cpi = metrics[0].cpi * (0.98 + r.uniform(0, 0.06))
            w1_roas = metrics[0].early_roas * (0.9 + r.uniform(0, 0.2))
            w2_ipm = w1_ipm * (0.85 + r.uniform(0, 0.2))
            w2_cpi = w1_cpi * (1.0 + r.uniform(-0.05, 0.15))
            w2_roas = w1_roas * (0.95 + r.uniform(-0.1, 0.2))
            imp1, imp2 = 50000, 52000
            inst1 = max(100, int(imp1 * w1_ipm / 1000))
            inst2 = max(100, int(imp2 * w2_ipm / 1000))
            window_metrics = [
                WindowMetrics(window_id="window_1", impressions=imp1, clicks=800, installs=inst1,
                              spend=6000, early_events=1200, early_revenue=480, ipm=round(w1_ipm, 2),
                              cpi=round(w1_cpi, 2), early_roas=round(w1_roas, 4)),
                WindowMetrics(window_id="window_2", impressions=imp2, clicks=840, installs=inst2,
                              spend=6240, early_events=1250, early_revenue=500, ipm=round(w2_ipm, 2),
                              cpi=round(w2_cpi, 2), early_roas=round(w2_roas, 4)),
            ]
            exp_ipm = w2_ipm * (0.80 + r.uniform(0, 0.15))
            exp_cpi = w2_cpi * (1.0 + r.uniform(0, 0.2))
            exp_roas = w2_roas * (0.9 + r.uniform(-0.1, 0.15))
            exp_inst = max(50, int(20000 * exp_ipm / 1000))
            expand_metrics = WindowMetrics(
                window_id="expand_segment", impressions=20000, clicks=320, installs=exp_inst,
                spend=2400, early_events=400, early_revenue=160, ipm=round(exp_ipm, 2),
                cpi=round(exp_cpi, 2), early_roas=round(exp_roas, 4),
            )
            validate_result = evaluate_validate_gate(window_metrics, expand_metrics)

        records.append(CardEvalRecord(
            card=card,
            card_score=card_score,
            status=status,
            variants=vs,
            explore_ios=exp_ios,
            explore_android=exp_android,
            validate_result=validate_result,
            window_metrics=window_metrics,
            expand_segment_metrics=expand_metrics,
        ))

    return records
