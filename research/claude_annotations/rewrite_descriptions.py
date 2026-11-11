"""Rewrite the 13 look descriptions to be contrastive, from the measured numbers only, and measure whether a reader
can then tell the looks apart (`lutdesc` with the new descriptions, different scenes, a different reading model).

  python -m research.claude_annotations.rewrite_descriptions generate --no-dry-run --confirm-paid   # one Sonnet request
  python -m research.claude_annotations.pipeline submit --task lutdesc2 ... ; collect; python -m research.claude_annotations.lut_eval

The current descriptions are templated from each look's measured colour numbers in isolation, and a reader picks the
right one out of four only 29 % of the time (chance 25 %). Here the writer sees all 13 looks' numbers at once and is
asked to write, for each, one or two sentences that say what distinguishes it from the others, in English and Chinese,
without naming any film, brand or preset. Output: research/claude_annotations/look_descriptions_rewrite.json. The
numbers stay the only source; the text is a presentation of them, so the website's "LUT-derived aesthetic" basis holds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import lut_tasks as L  # noqa: E402
from research.claude_annotations import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "look_descriptions_rewrite.json"
SYSTEM = """You write short, contrastive descriptions of colour grades for a photography website, from measured numbers.

You are given 13 grades. For each: the colour bias of midtones, shadows and highlights as CIELAB (a, b) offsets
(a > 0 red, a < 0 green; b > 0 yellow, b < 0 blue), the chroma ratio (< 1 muted, > 1 more saturated), the black lift in
8-bit levels, and the midtone contrast slope (< 1 flatter, > 1 punchier). Write for each grade one or two plain
sentences in English and the same in Simplified Chinese that say what a viewer would notice that distinguishes THIS
grade from the other twelve: name the direction and the strength of each cast in everyday words (a faint, a clear, a
strong warm/yellow/green/magenta/blue cast), say where it sits (shadows, midtones, highlights), and say how muted or
punchy it is compared with the others. Do not name any film, brand, preset or product. Do not use the words "vintage",
"cinematic", "nostalgic" or "dreamy". Do not use dashes. Chinese: short sentences, no 不仅…而且, a space between
Chinese and Latin characters or numbers. Return JSON only: {"<style_id>": {"en": "...", "zh": "..."}, ...} with
exactly the 13 ids given."""


def generate(paid: bool) -> None:
    cat = json.loads(L.CATALOG.read_text())["styles"]
    table = {k: v["description"]["numbers"] for k, v in cat.items()}
    user = "Measured numbers per grade id:\n" + json.dumps(table, indent=1) + "\n\nWrite the contrastive descriptions."
    if not paid:
        print("dry run; would send", len(user), "characters for", len(table), "grades"); return
    client = P.client_or_none(); assert client is not None, "no key"
    P.guard(0.10, P.cfg(), "film")
    msg = client.messages.create(model=P.cfg()["models"]["check"], max_tokens=6000, system=SYSTEM, messages=[{"role": "user", "content": user}],
                                 output_config={"effort": "medium", "format": {"type": "json_schema", "schema": {"type": "object", "properties": {k: {"type": "object", "properties": {"en": {"type": "string"}, "zh": {"type": "string"}}, "required": ["en", "zh"], "additionalProperties": False} for k in table}, "required": list(table), "additionalProperties": False}}})
    text = next(b.text for b in msg.content if b.type == "text"); data = json.loads(text)
    u = msg.usage; usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    cost = P.price(usage, msg.model, batch=False)
    P.append_jsonl(P.LEDGER, {"ts": P.now(), "task": "rewrite_descriptions", "bucket": "film", "model": msg.model, "mode": "sync", "requests": 1, **usage, "cost_usd": round(cost, 6)})
    OUT.write_text(json.dumps({"model": msg.model, "source": "catalog_map.json numbers", "descriptions": data}, ensure_ascii=False, indent=1))
    print(json.dumps(data, ensure_ascii=False, indent=1)[:2500]); print("cost", round(cost, 4))


if __name__ == "__main__":
    generate(paid="--no-dry-run" in sys.argv and "--confirm-paid" in sys.argv)
