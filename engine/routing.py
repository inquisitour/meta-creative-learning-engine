"""Send each learning to the team that can act on it (as ready-to-post markdown)."""
from __future__ import annotations

from pathlib import Path

ACTION = {
    "winner": "Make 3 more variants in this direction.",
    "loser": "Stop producing this. Retire running ads that use it.",
    "inconclusive": "No decision yet. Keep it in rotation so the test can finish.",
    "insufficient_data": "Produce at least 4 ads with this before judging it.",
}


def _team_report(title: str, learnings: list[dict], meta: dict) -> str:
    lines = [f"# {title}", "", f"Based on {meta['n_ads']} ads, {meta['total_impressions']:,} impressions "
             f"(each effect is vs the average level of its attribute, shown with a 95% confidence interval).", ""]
    for status in ("winner", "loser", "inconclusive", "insufficient_data"):
        group = [l for l in learnings if l["status"] == status]
        if not group:
            continue
        lines.append(f"## {status.replace('_', ' ').title()}")
        for l in group:
            lines.append(f"- {l['statement']} **Action:** {ACTION[status]}")
        lines.append("")
    return "\n".join(lines)


def route(analysis: dict, plan: dict, briefs: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = analysis["meta"]
    L = analysis["learnings"]
    paths = []

    p = out_dir / "content_team.md"
    p.write_text(_team_report("Learnings for the Content team (hooks and CTAs)",
                              [l for l in L if l["team"] == "content"], meta))
    paths.append(p)

    p = out_dir / "visual_team.md"
    p.write_text(_team_report("Learnings for the Visual team (concepts and formats)",
                              [l for l in L if l["team"] == "visual"], meta))
    paths.append(p)

    perf = ["# Next test plan for the Performance team", "",
            f"Daily budget: EUR {plan['daily_budget_eur']:.0f}, assumed CPM EUR {plan['assumed_cpm_eur']}.", ""]
    for c in plan["cells"]:
        a = c["attributes"]
        perf.append(f"## {c['cell_id']} ({c['role']}, {c['budget_share']:.0%} of budget)")
        perf.append(f"- Creative: hook={a['hook']}, visual={a['visual']}, cta={a['cta']}, format={a['format']}")
        perf.append(f"- Hypothesis: {c['hypothesis']}")
        perf.append(f"- Run at least {c['min_impressions']:,} impressions and {c['min_days']} days before reading CTR results.")
        perf.append(f"- Purchase-level confirmation alone would need ~{c['purchase_test_days_if_run_alone']} days at this budget, "
                    "so purchases are tracked through the pooled model instead.")
        perf.append("")
    perf.append(f"**Stopping rule:** {plan['stopping_rule']}")
    perf.append("")
    perf.append(f"**Data quality:** ad-to-ad variance is {meta['dispersion_ctr']}x (CTR) and "
                f"{meta['dispersion_cvr']}x (CVR) the level pure chance would give; "
                "intervals are widened accordingly.")
    perf.append("")
    p = out_dir / "performance_team.md"
    p.write_text("\n".join(perf))
    paths.append(p)

    brief_lines = ["# Generation briefs for the next test cells (all need brand approval)", ""]
    for b in briefs:
        brief_lines.append(f"## {b['cell_id']}")
        brief_lines.append(f"- Hook: {b['hook']['type']} - {b['hook']['instruction']}")
        brief_lines.append(f"- Visual: {b['visual']['concept']} as {b['visual']['format']}")
        brief_lines.append(f"- CTA button: {b['cta']['button_text']}")
        for o in b["copy_options"]:
            brief_lines.append(f"  - copy: {o}")
        brief_lines.append("")
    p = out_dir / "creative_briefs.md"
    p.write_text("\n".join(brief_lines))
    paths.append(p)
    return paths
