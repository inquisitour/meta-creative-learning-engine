"""Invariant checker: formal guarantees the output must satisfy before it is released.

The statistics can be wrong in subtle ways and an LLM step can drift, so nothing
is sent to a team unless every invariant holds. Failures are collected, not
raised one at a time, so a reviewer sees the whole picture.
"""
from __future__ import annotations

from .analyze import ATTRIBUTES, MIN_ADS_PER_LEVEL


class ValidationError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__(f"{len(errors)} invariant violation(s):\n- " + "\n- ".join(errors))
        self.errors = errors


def check(analysis: dict, plan: dict, briefs: list[dict]) -> list[str]:
    errs: list[str] = []

    # ---- learnings --------------------------------------------------------------
    seen = set()
    for l in analysis["learnings"]:
        key = (l["attribute"], l["level"])
        tag = f"{l['attribute']}={l['level']}"
        if key in seen:
            errs.append(f"duplicate learning {tag}")
        seen.add(key)
        if not (l["net_ci_low_pct"] <= l["net_lift_pct"] <= l["net_ci_high_pct"]):
            errs.append(f"{tag}: point estimate outside its own confidence interval")
        if l["status"] == "winner" and not l["net_ci_low_pct"] > 0:
            errs.append(f"{tag}: marked winner but CI includes 0")
        if l["status"] == "loser" and not l["net_ci_high_pct"] < 0:
            errs.append(f"{tag}: marked loser but CI includes 0")
        if l["status"] in ("winner", "loser") and l["n_ads"] < MIN_ADS_PER_LEVEL:
            errs.append(f"{tag}: verdict '{l['status']}' on only {l['n_ads']} ads")
        if l["status"] not in ("winner", "loser", "inconclusive", "insufficient_data"):
            errs.append(f"{tag}: unknown status {l['status']!r}")

    # ---- test plan --------------------------------------------------------------
    cells = plan["cells"]
    ids = [c["cell_id"] for c in cells]
    if len(set(ids)) != len(ids):
        errs.append("duplicate cell ids in test plan")
    controls = [c for c in cells if c["role"] == "control"]
    if len(controls) != 1:
        errs.append(f"expected exactly 1 control cell, found {len(controls)}")
    if abs(sum(c["budget_share"] for c in cells) - 1.0) > 1e-3:
        errs.append(f"budget shares sum to {sum(c['budget_share'] for c in cells):.4f}, not 1")
    combos = [tuple(c["attributes"][a] for a in ATTRIBUTES) for c in cells]
    if len(set(combos)) != len(combos):
        errs.append("two cells have identical creative attributes (wasted budget)")

    champion = next((c for c in cells if c["role"] == "exploit"), None)
    for c in cells:
        if set(c["attributes"]) != set(ATTRIBUTES):
            errs.append(f"{c['cell_id']}: missing attributes")
        if c["min_impressions"] <= 0 or c["min_days"] <= 0:
            errs.append(f"{c['cell_id']}: non-positive sample size or duration")
        if c["role"] == "explore":
            ref = champion["attributes"] if champion else controls[0]["attributes"]
            diff = [a for a in ATTRIBUTES if c["attributes"][a] != ref[a]]
            if len(diff) != 1:
                errs.append(f"{c['cell_id']}: explore cell changes {len(diff)} attributes, must be exactly 1")
            else:
                attr = diff[0]
                est = {(l["attribute"], l["level"]): l for l in analysis["learnings"]}
                challenger = est[(attr, c["attributes"][attr])]
                incumbent_level = est[(attr, ref[attr])]
                if challenger["net_ci_high_pct"] <= incumbent_level["net_lift_pct"]:
                    errs.append(
                        f"{c['cell_id']}: challenger {attr}={challenger['level']} cannot plausibly beat "
                        f"{incumbent_level['level']} (upper bound {challenger['net_ci_high_pct']:+.1f}% "
                        f"<= {incumbent_level['net_lift_pct']:+.1f}%)"
                    )

    # ---- briefs -----------------------------------------------------------------
    if {b["cell_id"] for b in briefs} != set(ids):
        errs.append("briefs do not cover exactly the cells in the plan")
    for b in briefs:
        if not b["copy_options"]:
            errs.append(f"{b['cell_id']}: no copy options")
        if any(len(o) > 125 for o in b["copy_options"]):
            errs.append(f"{b['cell_id']}: copy option exceeds 125 characters")
        if b["status"] != "needs_brand_approval":
            errs.append(f"{b['cell_id']}: generated asset not routed to brand approval")
    return errs


def validate(analysis: dict, plan: dict, briefs: list[dict]) -> None:
    errs = check(analysis, plan, briefs)
    if errs:
        raise ValidationError(errs)
