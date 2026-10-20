"""Run with:  python -m pytest -q   (or just: python tests/test_engine.py)"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.analyze import ATTRIBUTES, analyze  # noqa: E402
from engine.briefs import make_briefs, openai_compatible  # noqa: E402
from engine.planner import plan_next_test  # noqa: E402
from engine.validate import check  # noqa: E402

_spec = importlib.util.spec_from_file_location("generate_data", ROOT / "data" / "generate_data.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

LEVELS = {"hook": gen.HOOKS, "visual": gen.VISUALS, "cta": gen.CTAS, "format": gen.FORMATS}


def true_net_pct(attr: str, level: str) -> float:
    """Ground-truth net effect (% on purchases/impression) vs the attribute's mean level."""
    eff = {lv: gen.TRUE_CTR.get((attr, lv), 0) + gen.TRUE_CVR.get((attr, lv), 0) for lv in LEVELS[attr]}
    mean = sum(eff.values()) / len(eff)
    return (math.exp(eff[level] - mean) - 1) * 100


def _full_run(seed=7, n=80):
    df = gen.generate(n_ads=n, seed=seed)
    analysis = analyze(df)
    plan = plan_next_test(df, analysis)
    briefs = make_briefs(plan)
    return df, analysis, plan, briefs


def test_recovers_planted_effects():
    _, analysis, _, _ = _full_run()
    ls = analysis["learnings"]
    # never call the wrong direction on a level that has a real effect (|true| >= 3%)
    # and allow at most one false positive among true-null levels (95% intervals over
    # 15 levels give ~0.75 expected false calls; the "champion" test cell exists to
    # catch exactly these before they are rolled out)
    null_calls = 0
    for l in ls:
        t = true_net_pct(l["attribute"], l["level"])
        if abs(t) < 3:
            null_calls += l["status"] in ("winner", "loser")
            continue
        if l["status"] == "winner":
            assert t > 0, f"wrong-direction winner {l['attribute']}={l['level']} (true {t:.1f}%)"
        if l["status"] == "loser":
            assert t < 0, f"wrong-direction loser {l['attribute']}={l['level']} (true {t:.1f}%)"
    assert null_calls <= 1, f"{null_calls} false verdicts on true-null levels"
    # intervals should cover the truth most of the time
    covered = sum(l["net_ci_low_pct"] <= true_net_pct(l["attribute"], l["level"]) <= l["net_ci_high_pct"] for l in ls)
    assert covered / len(ls) >= 0.8, f"only {covered}/{len(ls)} intervals cover the truth"
    # the biggest planted drivers must be found
    status = {(l["attribute"], l["level"]): l["status"] for l in ls}
    assert status[("cta", "gift_now")] == "winner"
    assert status[("cta", "learn_more")] == "loser"


def test_null_world_stays_quiet():
    """If nothing truly matters, the engine must not invent drivers."""
    saved = (gen.TRUE_CTR, gen.TRUE_CVR)
    gen.TRUE_CTR, gen.TRUE_CVR = {}, {}
    try:
        calls = tests = 0
        for seed in range(1, 9):
            for l in analyze(gen.generate(n_ads=80, seed=seed))["learnings"]:
                tests += 1
                calls += l["status"] in ("winner", "loser")
        assert calls / tests < 0.10, f"{calls}/{tests} false verdicts in a null world"
    finally:
        gen.TRUE_CTR, gen.TRUE_CVR = saved


def test_clean_run_passes_all_invariants():
    _, analysis, plan, briefs = _full_run()
    assert check(analysis, plan, briefs) == []


def test_validator_catches_corruption():
    _, analysis, plan, briefs = _full_run()

    bad = copy.deepcopy(analysis)
    inc = next(l for l in bad["learnings"] if l["status"] == "inconclusive")
    inc["status"] = "winner"  # claim a winner whose CI includes 0
    assert any("CI includes 0" in e for e in check(bad, plan, briefs))

    bad_plan = copy.deepcopy(plan)
    bad_plan["cells"][0]["budget_share"] += 0.2
    assert any("budget shares" in e for e in check(analysis, bad_plan, briefs))

    bad_plan = copy.deepcopy(plan)
    explore = next(c for c in bad_plan["cells"] if c["role"] == "explore")
    explore["attributes"]["format"] = next(f for f in gen.FORMATS if f != explore["attributes"]["format"])
    explore["attributes"]["cta"] = next(c for c in gen.CTAS if c != explore["attributes"]["cta"])
    assert any("must be exactly 1" in e or "identical" in e for e in check(analysis, bad_plan, briefs))

    bad_briefs = copy.deepcopy(briefs)
    bad_briefs[0]["copy_options"] = ["x" * 200]
    assert any("125 characters" in e for e in check(analysis, plan, bad_briefs))


def test_validator_rejects_hopeless_challenger():
    _, analysis, plan, briefs = _full_run()
    bad = copy.deepcopy(analysis)
    cell = next(c for c in plan["cells"] if c["role"] == "explore")
    sv = cell["single_variable"]
    for l in bad["learnings"]:  # make the challenger's best case worse than the level it replaces
        if (l["attribute"], l["level"]) == (sv["attribute"], sv["to"]):
            l["net_ci_high_pct"] = -50.0
            l["net_lift_pct"] = -60.0
            l["net_ci_low_pct"] = -70.0
    assert any("cannot plausibly beat" in e for e in check(bad, plan, briefs))


def test_pipeline_is_robust_across_datasets():
    """Many datasets, including tiny and effect-free ones: never crash, never emit an invalid plan."""
    saved = (gen.TRUE_CTR, gen.TRUE_CVR)
    try:
        for seed in range(1, 41):
            if seed % 4 == 0:
                gen.TRUE_CTR, gen.TRUE_CVR = {}, {}
            else:
                gen.TRUE_CTR, gen.TRUE_CVR = saved
            n = 30 if seed % 5 == 0 else 80
            _, analysis, plan, briefs = _full_run(seed=seed, n=n)
            assert check(analysis, plan, briefs) == [], f"seed {seed}"
    finally:
        gen.TRUE_CTR, gen.TRUE_CVR = saved


def test_explore_cells_are_single_variable():
    _, _, plan, _ = _full_run()
    champ = next(c for c in plan["cells"] if c["role"] == "exploit")["attributes"]
    for c in plan["cells"]:
        if c["role"] == "explore":
            assert sum(c["attributes"][a] != champ[a] for a in ATTRIBUTES) == 1


class _MockLLM(BaseHTTPRequestHandler):
    reply = json.dumps(["Option one", "Option two", "y" * 300])

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": type(self).reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _MockLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_llm_adapter_parses_and_enforces_length_guardrail():
    srv = _serve()
    try:
        _MockLLM.reply = json.dumps(["Option one", "Option two", "y" * 300])
        gen_fn = openai_compatible(f"http://127.0.0.1:{srv.server_port}", "k", "m")
        _, _, plan, _ = _full_run()
        briefs = make_briefs(plan, gen_fn)
        assert briefs[0]["copy_options"] == ["Option one", "Option two"]  # the 300-char one is dropped in code
    finally:
        srv.shutdown()


def test_llm_garbage_falls_back_to_templates():
    srv = _serve()
    try:
        _MockLLM.reply = "sorry, I cannot do that"
        gen_fn = openai_compatible(f"http://127.0.0.1:{srv.server_port}", "k", "m")
        _, analysis, plan, _ = _full_run()
        briefs = make_briefs(plan, gen_fn)
        assert all(b["copy_options"] and "fell back" in b["generator_error"] for b in briefs)
        assert check(analysis, plan, briefs) == []
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
