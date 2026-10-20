"""Turn test cells into generation briefs, and generate copy options.

This is the "AI makes the next iteration itself" step. The brief is structured,
so a downstream generator (LLM for copy, image/video model for visuals) gets
unambiguous instructions. Copy generation is pluggable:

  * default: deterministic templates, so the prototype runs offline;
  * LLM mode: any OpenAI-compatible chat endpoint (see openai_compatible()).

Humans stay in the loop for brand approval - briefs carry guardrails and every
generated asset is marked `status: needs_brand_approval`.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Callable

HOOK_GUIDE = {
    "question": "Open with a question the buyer already asks themselves.",
    "social_proof": "Open with a concrete customer number, quote or rating.",
    "problem_solution": "Name a familiar annoyance, then show the product as the fix.",
    "offer_led": "Open with the concrete offer (price, bundle, free engraving).",
    "story": "Open with a short origin or craftsmanship story.",
}

TEMPLATE_COPY = {
    "question": [
        "Still wearing a watch that looks like everyone else's?",
        "What would a watch made from real wood say about you?",
        "Looking for a gift that is not another gadget?",
    ],
    "social_proof": [
        "Loved by 100,000+ customers who wanted something different.",
        "'The first watch people actually ask about.' - verified buyer",
        "4.8 stars from thousands of happy wrists.",
    ],
    "problem_solution": [
        "Tired of cold, metal watches? Meet the one that feels alive.",
        "No more generic gifts: give something made from real wood.",
        "Fed up with watches that look the same? Try natural grain.",
    ],
    "offer_led": [
        "Free engraving on every watch this week.",
        "Bundle a watch and a strap and save, today only.",
        "Gift-ready packaging included, no extra cost.",
    ],
    "story": [
        "Every grain is different, and so is every watch we make.",
        "From a small workshop to your wrist: how one watch is made.",
        "Why we started building watches from wood.",
    ],
}

CTA_TEXT = {"shop_now": "Shop now", "gift_now": "Find the perfect gift", "learn_more": "Learn more"}

GUARDRAILS = [
    "Brand tone: warm, natural, understated. No fake urgency.",
    "No unverifiable claims (numbers must come from the approved fact sheet).",
    "Max 125 characters primary text; headline max 40 characters.",
]


def build_brief(cell: dict) -> dict:
    a = cell["attributes"]
    return {
        "cell_id": cell["cell_id"],
        "role": cell["role"],
        "hook": {"type": a["hook"], "instruction": HOOK_GUIDE[a["hook"]]},
        "visual": {"concept": a["visual"], "format": a["format"]},
        "cta": {"type": a["cta"], "button_text": CTA_TEXT[a["cta"]]},
        "hypothesis": cell["hypothesis"],
        "guardrails": GUARDRAILS,
        "copy_options": [],
        "status": "needs_brand_approval",
    }


def template_copy(brief: dict) -> list[str]:
    return list(TEMPLATE_COPY[brief["hook"]["type"]])


def openai_compatible(base_url: str, api_key: str, model: str) -> Callable[[dict], list[str]]:
    """Return a copy generator that calls an OpenAI-compatible /chat/completions API."""

    def generate(brief: dict) -> list[str]:
        prompt = (
            "Write 3 Meta ad primary-text options as a JSON array of strings, nothing else.\n"
            f"Hook type: {brief['hook']['type']} - {brief['hook']['instruction']}\n"
            f"Visual concept: {brief['visual']['concept']} ({brief['visual']['format']})\n"
            f"Call to action: {brief['cta']['button_text']}\n"
            "Rules:\n- " + "\n- ".join(brief["guardrails"])
        )
        body = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
        ).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = json.load(resp)["choices"][0]["message"]["content"]
        options = json.loads(content)
        if not (isinstance(options, list) and all(isinstance(o, str) and o for o in options)):
            raise ValueError("LLM did not return a JSON array of non-empty strings")
        return [o for o in options if len(o) <= 125]  # enforce the guardrail in code, not in the prompt

    return generate


def make_briefs(plan: dict, generator: Callable[[dict], list[str]] | None = None) -> list[dict]:
    generator = generator or template_copy
    briefs = []
    for cell in plan["cells"]:
        brief = build_brief(cell)
        try:
            brief["copy_options"] = generator(brief)
        except Exception as exc:  # never block the pipeline on the generator
            brief["copy_options"] = template_copy(brief)
            brief["generator_error"] = f"{type(exc).__name__}: {exc}; fell back to templates"
        briefs.append(brief)
    return briefs
