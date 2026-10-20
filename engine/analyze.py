"""Turn raw ad results into structured, statistically honest learnings."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from .glm import Z95, fit_quasibinomial

ATTRIBUTES = ["hook", "visual", "cta", "format"]
MIN_ADS_PER_LEVEL = 4  # below this we refuse to call a winner or loser


@dataclass
class Learning:
    attribute: str
    level: str
    n_ads: int
    impressions: int
    ctr_lift_pct: float       # effect on click-through vs the attribute's average level
    cvr_lift_pct: float       # effect on click->purchase conversion
    net_lift_pct: float       # combined effect on purchases per impression
    net_ci_low_pct: float
    net_ci_high_pct: float
    status: str               # winner | loser | inconclusive | insufficient_data
    team: str                 # who should act on it
    statement: str            # human-readable sentence


TEAM_BY_ATTRIBUTE = {
    "hook": "content",
    "cta": "content",
    "visual": "visual",
    "format": "visual",
}


def _pct(logodds: float) -> float:
    # At click/purchase rates of a few percent, odds ratio ~ rate ratio.
    return round((math.exp(logodds) - 1) * 100, 1)


def analyze(df: pd.DataFrame) -> dict:
    ctr = fit_quasibinomial(df, ATTRIBUTES, "clicks", "impressions")
    cvr = fit_quasibinomial(df, ATTRIBUTES, "purchases", "clicks")
    ctr_by = {(e.attribute, e.level): e for e in ctr.effects}
    cvr_by = {(e.attribute, e.level): e for e in cvr.effects}

    learnings: list[Learning] = []
    for (attr, level), e1 in ctr_by.items():
        e2 = cvr_by[(attr, level)]
        sub = df[df[attr] == level]
        n_ads = len(sub)
        net = e1.coef + e2.coef
        # the two stages are modelled separately; treat their errors as independent
        se = math.sqrt(e1.se**2 + e2.se**2)
        # Decide on the *reported* (rounded) numbers, so what a reader sees can never
        # contradict the verdict (e.g. an upper bound of -0.0004% printed as 0.0%).
        net_pct = _pct(net)
        lo_pct, hi_pct = _pct(net - Z95 * se), _pct(net + Z95 * se)

        if n_ads < MIN_ADS_PER_LEVEL:
            status = "insufficient_data"
        elif lo_pct > 0:
            status = "winner"
        elif hi_pct < 0:
            status = "loser"
        else:
            status = "inconclusive"

        learnings.append(
            Learning(
                attribute=attr,
                level=level,
                n_ads=n_ads,
                impressions=int(sub["impressions"].sum()),
                ctr_lift_pct=_pct(e1.coef),
                cvr_lift_pct=_pct(e2.coef),
                net_lift_pct=net_pct,
                net_ci_low_pct=lo_pct,
                net_ci_high_pct=hi_pct,
                status=status,
                team=TEAM_BY_ATTRIBUTE[attr],
                statement=_statement(attr, level, status, net_pct, lo_pct, hi_pct, n_ads),
            )
        )

    learnings.sort(key=lambda l: l.net_lift_pct, reverse=True)
    overall_ctr = df["clicks"].sum() / df["impressions"].sum()
    overall_cvr = df["purchases"].sum() / df["clicks"].sum()
    return {
        "learnings": [asdict(l) for l in learnings],
        "meta": {
            "n_ads": int(len(df)),
            "total_impressions": int(df["impressions"].sum()),
            "overall_ctr": round(float(overall_ctr), 5),
            "overall_cvr": round(float(overall_cvr), 5),
            "dispersion_ctr": round(ctr.dispersion, 2),
            "dispersion_cvr": round(cvr.dispersion, 2),
            "interval": "95% (quasi-binomial)",
            "min_ads_per_level": MIN_ADS_PER_LEVEL,
        },
    }


def _statement(attr, level, status, net, lo, hi, n_ads) -> str:
    name = f"{attr} = {level}"
    rng = f"{net:+.1f}% (95% CI {lo:+.1f}% to {hi:+.1f}%)"
    if status == "winner":
        return f"{name} drives more purchases per impression: {rng}, across {n_ads} ads."
    if status == "loser":
        return f"{name} underperforms: {rng}, across {n_ads} ads."
    if status == "insufficient_data":
        return f"{name}: only {n_ads} ads so far, too few to judge."
    return f"{name}: no reliable effect yet. Estimate {rng}, {n_ads} ads."
