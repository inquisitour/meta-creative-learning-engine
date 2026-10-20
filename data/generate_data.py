"""Generate dummy Meta ad performance data with *planted* ground-truth effects.

The planted effects let us verify the analysis engine recovers real drivers
(and does not invent drivers that aren't there). Run:

    python data/generate_data.py            # writes data/ads.csv
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

HOOKS = ["question", "social_proof", "problem_solution", "offer_led", "story"]
VISUALS = ["lifestyle", "product_closeup", "unboxing", "gift_scene"]
CTAS = ["shop_now", "gift_now", "learn_more"]
FORMATS = ["image", "video", "carousel"]

# Ground truth: additive effects on the logit of each funnel stage.
# Anything not listed has effect 0 (i.e. a "null" level that must NOT be flagged).
TRUE_CTR = {
    ("hook", "social_proof"): 0.35,
    ("hook", "story"): 0.15,
    ("hook", "offer_led"): 0.10,
    ("format", "video"): 0.30,
    ("format", "carousel"): 0.05,
    ("cta", "learn_more"): 0.10,
}
TRUE_CVR = {
    ("hook", "offer_led"): 0.25,
    ("visual", "gift_scene"): 0.30,
    ("cta", "gift_now"): 0.20,
    ("cta", "learn_more"): -0.30,
    ("format", "carousel"): 0.10,
}

BASE_CTR = 0.011   # 1.1% click-through
BASE_CVR = 0.025   # 2.5% click -> purchase
AOV = 89.0         # EUR
CPM = 9.0          # EUR per 1000 impressions


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def generate(n_ads: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    rows = []
    # Deliberately unbalanced sampling: production teams over-use some levels.
    hook_w = [0.15, 0.30, 0.20, 0.15, 0.20]
    visual_w = [0.35, 0.25, 0.20, 0.20]
    cta_w = [0.5, 0.2, 0.3]
    fmt_w = [0.35, 0.45, 0.20]

    for i in range(n_ads):
        hook = rng.choices(HOOKS, hook_w)[0]
        visual = rng.choices(VISUALS, visual_w)[0]
        cta = rng.choices(CTAS, cta_w)[0]
        fmt = rng.choices(FORMATS, fmt_w)[0]
        attrs = {"hook": hook, "visual": visual, "cta": cta, "format": fmt}

        # Ad-level random effect => overdispersion, as in real data.
        re_ctr = nprng.normal(0, 0.15)
        re_cvr = nprng.normal(0, 0.15)
        lo_ctr = _logit(BASE_CTR) + re_ctr + sum(TRUE_CTR.get((k, v), 0) for k, v in attrs.items())
        lo_cvr = _logit(BASE_CVR) + re_cvr + sum(TRUE_CVR.get((k, v), 0) for k, v in attrs.items())

        impressions = int(nprng.lognormal(mean=math.log(60_000), sigma=0.6))
        clicks = int(nprng.binomial(impressions, _sigmoid(lo_ctr)))
        purchases = int(nprng.binomial(max(clicks, 0), _sigmoid(lo_cvr)))
        spend = round(impressions / 1000 * CPM * nprng.normal(1.0, 0.05), 2)
        revenue = round(purchases * AOV * nprng.normal(1.0, 0.08), 2)

        rows.append(
            {
                "ad_id": f"AD-{i + 1:03d}",
                "hook": hook,
                "visual": visual,
                "cta": cta,
                "format": fmt,
                "impressions": impressions,
                "clicks": clicks,
                "purchases": purchases,
                "spend": spend,
                "revenue": revenue,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "ads.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} ads -> {out}")
    print(df[["impressions", "clicks", "purchases", "spend", "revenue"]].describe().round(1))
