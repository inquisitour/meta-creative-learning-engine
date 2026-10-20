"""Decide what to test next: exploit proven winners, explore the most promising unknowns.

Test design rules (enforced later by validate.py):
  * cell A  = incumbent (best existing ad)              -> the control
  * cell B  = "champion": the incumbent with every proven winner swapped in
  * cells C.. = champion with exactly ONE attribute changed to an unresolved
                level -> a single-variable test, so any difference is attributable
  * budget shares sum to 1; each cell gets a minimum sample size and duration
"""
from __future__ import annotations

import math

import pandas as pd

from .analyze import ATTRIBUTES, MIN_ADS_PER_LEVEL

Z_ALPHA = 1.96    # two-sided 5%
Z_POWER = 0.8416  # 80% power
CTR_MDE = 0.15    # smallest relative CTR lift we care to detect
CVR_MDE = 0.20    # smallest relative CVR lift we care to detect
MAX_EXPLORE_CELLS = 2


def _n_per_cell(p: float, rel_mde: float) -> int:
    """Observations per cell to detect a relative lift of rel_mde on base rate p."""
    delta = p * rel_mde
    return math.ceil(2 * (Z_ALPHA + Z_POWER) ** 2 * p * (1 - p) / delta**2)


def plan_next_test(df: pd.DataFrame, analysis: dict, daily_budget: float = 400.0) -> dict:
    learnings = analysis["learnings"]
    meta = analysis["meta"]

    # --- incumbent: best existing ad (ignore tiny-impression ads: pure noise) -----
    pool = df[df["impressions"] >= df["impressions"].median()].copy()
    pool["ppm"] = pool["purchases"] / pool["impressions"] * 1000
    inc_row = pool.sort_values("ppm", ascending=False).iloc[0]
    incumbent = {a: inc_row[a] for a in ATTRIBUTES}

    # --- champion: swap in the best proven winner per attribute -------------------
    champion = dict(incumbent)
    swapped = []
    for attr in ATTRIBUTES:
        winners = [l for l in learnings if l["attribute"] == attr and l["status"] == "winner"]
        if winners:
            best = max(winners, key=lambda l: l["net_lift_pct"])
            if best["level"] != champion[attr]:
                swapped.append({"attribute": attr, "from": champion[attr], "to": best["level"]})
                champion[attr] = best["level"]

    cells = [
        {
            "cell_id": "A_incumbent",
            "role": "control",
            "attributes": incumbent,
            "hypothesis": f"Baseline: best existing ad ({inc_row['ad_id']}).",
        }
    ]
    if champion != incumbent:
        cells.append(
            {
                "cell_id": "B_champion",
                "role": "exploit",
                "attributes": champion,
                "changes_vs_control": swapped,
                "hypothesis": "Stacking all proven winners beats the incumbent "
                "(checks that effects add up instead of interfering).",
            }
        )

    # --- explore: unresolved levels with the highest plausible upside -------------
    base_for_explore = champion
    estimate = {(l["attribute"], l["level"]): l["net_lift_pct"] for l in learnings}
    # A challenger is only worth budget if its best case could beat the level it
    # replaces (otherwise the test cannot plausibly produce a new winner).
    candidates = [
        l
        for l in learnings
        if l["status"] == "inconclusive"
        and l["n_ads"] >= MIN_ADS_PER_LEVEL
        and base_for_explore[l["attribute"]] != l["level"]
        and l["net_ci_high_pct"] > estimate[(l["attribute"], base_for_explore[l["attribute"]])]
    ]
    candidates.sort(key=lambda l: l["net_ci_high_pct"], reverse=True)  # most upside first
    used_attrs: set[str] = set()
    existing = {tuple(cell["attributes"][a] for a in ATTRIBUTES) for cell in cells}
    explore = []
    for c in candidates:
        if c["attribute"] in used_attrs:
            continue
        combo = tuple(c["level"] if a == c["attribute"] else base_for_explore[a] for a in ATTRIBUTES)
        if combo in existing:  # would just recreate the incumbent (or another cell)
            continue
        existing.add(combo)
        used_attrs.add(c["attribute"])
        explore.append(c)
        if len(explore) == MAX_EXPLORE_CELLS:
            break
    for i, c in enumerate(explore):
        attrs = dict(base_for_explore)
        attrs[c["attribute"]] = c["level"]
        cells.append(
            {
                "cell_id": f"{chr(ord('C') + i)}_explore_{c['attribute']}_{c['level']}",
                "role": "explore",
                "attributes": attrs,
                "single_variable": {
                    "attribute": c["attribute"],
                    "from": base_for_explore[c["attribute"]],
                    "to": c["level"],
                },
                "hypothesis": f"{c['attribute']}={c['level']} could beat "
                f"{base_for_explore[c['attribute']]} (current estimate "
                f"{c['net_lift_pct']:+.1f}%, upper bound {c['net_ci_high_pct']:+.1f}%).",
            }
        )

    # --- budget: control and champion get fixed shares, explores split the rest ---
    shares = {"control": 0.20, "exploit": 0.30}
    fixed = sum(shares[c["role"]] for c in cells if c["role"] in shares)
    n_explore = sum(1 for c in cells if c["role"] == "explore")
    for c in cells:
        if c["role"] in shares:
            c["budget_share"] = shares[c["role"]]
        else:
            c["budget_share"] = (1 - fixed) / n_explore
    total = sum(c["budget_share"] for c in cells)
    for c in cells:  # renormalise (e.g. no champion cell, or no explore cells)
        c["budget_share"] = round(c["budget_share"] / total, 4)
    drift = round(1 - sum(c["budget_share"] for c in cells), 4)
    cells[-1]["budget_share"] = round(cells[-1]["budget_share"] + drift, 4)

    # --- minimum sample size / duration ------------------------------------------
    # Primary metric per cell is CTR: it is frequent enough to test in days.
    # Purchases are ~40x rarer, so per-cell purchase tests would take months at this
    # budget; we report that number honestly instead of pretending otherwise.
    imp_ctr = _n_per_cell(meta["overall_ctr"], CTR_MDE)
    clicks_cvr = _n_per_cell(meta["overall_cvr"], CVR_MDE)
    imp_cvr = math.ceil(clicks_cvr / meta["overall_ctr"])
    min_impr = int(math.ceil(imp_ctr / 1000) * 1000)
    cpm = float(df["spend"].sum() / df["impressions"].sum() * 1000)
    for c in cells:
        daily_impr = daily_budget * c["budget_share"] / cpm * 1000
        c["min_impressions"] = min_impr
        c["min_days"] = max(3, math.ceil(min_impr / daily_impr))
        c["purchase_test_days_if_run_alone"] = math.ceil(imp_cvr / daily_impr)

    return {
        "daily_budget_eur": daily_budget,
        "assumed_cpm_eur": round(cpm, 2),
        "primary_metric": "ctr",
        "sample_size_basis": {
            "alpha": 0.05,
            "power": 0.80,
            "ctr_relative_mde": CTR_MDE,
            "cvr_relative_mde": CVR_MDE,
            "impressions_needed_for_ctr": imp_ctr,
            "impressions_needed_for_purchase_level": imp_cvr,
        },
        "stopping_rule": "Do not read results before every cell reaches min_impressions "
        "and min_days. Decide on CTR: adopt a variant only if its 95% CI vs the "
        "champion excludes 0; otherwise keep the champion. Purchase effects are not "
        "tested per cell (see purchase_test_days_if_run_alone); they stay estimated by "
        "the pooled model, which is refit whenever new results arrive.",
        "cells": cells,
    }
