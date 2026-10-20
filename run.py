"""Run the full loop: results -> learnings -> next test plan -> briefs -> validation -> routing.

    python run.py                     # uses data/ads.csv, offline, template copy
    python run.py --data my_ads.csv   # your own export (same columns)
    python run.py --llm               # copy via an OpenAI-compatible API (needs LLM_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from engine.analyze import analyze
from engine.briefs import make_briefs, openai_compatible
from engine.planner import plan_next_test
from engine.routing import route
from engine.validate import ValidationError, validate

ROOT = Path(__file__).parent
REQUIRED = ["ad_id", "hook", "visual", "cta", "format", "impressions", "clicks", "purchases", "spend", "revenue"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing columns: {missing}")
    if (df["clicks"] > df["impressions"]).any() or (df["purchases"] > df["clicks"]).any():
        sys.exit("Data error: clicks > impressions or purchases > clicks in some rows")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ads.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument("--budget", type=float, default=400.0, help="daily test budget in EUR")
    ap.add_argument("--llm", action="store_true", help="generate copy with an LLM")
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"{args.data} not found. Run: python data/generate_data.py")
    df = load(args.data)

    analysis = analyze(df)
    plan = plan_next_test(df, analysis, daily_budget=args.budget)

    generator = None
    if args.llm:
        key = os.environ.get("LLM_API_KEY")
        if not key:
            sys.exit("--llm needs LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL)")
        generator = openai_compatible(
            os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            key,
            os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        )
    briefs = make_briefs(plan, generator)

    try:
        validate(analysis, plan, briefs)  # nothing is released unless all invariants hold
    except ValidationError as exc:
        sys.exit(f"REFUSING TO RELEASE OUTPUT\n{exc}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "learnings.json").write_text(json.dumps(analysis, indent=2))
    (args.out / "test_plan.json").write_text(json.dumps(plan, indent=2))
    (args.out / "briefs.json").write_text(json.dumps(briefs, indent=2))
    route(analysis, plan, briefs, args.out / "reports")

    print(f"{analysis['meta']['n_ads']} ads, {analysis['meta']['total_impressions']:,} impressions\n")
    print(f"{'attribute=level':<28}{'ads':>4}{'net lift':>10}{'95% CI':>20}  status")
    for l in analysis["learnings"]:
        ci = f"[{l['net_ci_low_pct']:+.1f}%, {l['net_ci_high_pct']:+.1f}%]"
        print(f"{l['attribute'] + '=' + l['level']:<28}{l['n_ads']:>4}{l['net_lift_pct']:>+9.1f}%{ci:>20}  {l['status']}")
    print("\nNext test plan:")
    for c in plan["cells"]:
        a = c["attributes"]
        print(f"  {c['cell_id']:<34}{c['budget_share']:>5.0%}  "
              f"{a['hook']}/{a['visual']}/{a['cta']}/{a['format']}  "
              f"(>= {c['min_impressions']:,} impr, {c['min_days']} days)")
    print(f"\nAll invariants passed. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
