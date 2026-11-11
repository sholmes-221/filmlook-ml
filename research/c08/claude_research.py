"""C08e: Claude as an offline research assistant (owner briefs of 2026-11-11). Never called by the website.

    python -m research.c08.claude_research datasheets --no-dry-run --confirm-paid   # one Sonnet request per datasheet PDF
    python -m research.c08.claude_research lutsig --no-dry-run --confirm-paid       # hypotheses from measured LUT signatures

Every request goes through the campaign's budget guard (cap, bucket `film`, expiry) and is priced into the same ledger
(data/annotations/claude_research/usage.jsonl). Responses are cached in data/c08/claude/<task>/<item>.json and never
requested twice. Datasheet extraction separates what the document states, what was read approximately from a graph,
and the manufacturer's own descriptive (subjective) language. Hypotheses are labelled as hypotheses: they are tested
against measurements before anything uses them.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from research.claude_annotations import pipeline as P

OUT = Path("data/c08/claude")
SHEETS = Path("data/c08/datasheets")
MODEL = "claude-sonnet-5-5"

PROPS = ["granularity", "resolving_power", "mtf", "characteristic_curve_shape", "contrast_gamma", "density_range", "exposure_latitude",
         "push_pull", "spectral_sensitivity_peaks", "spectral_sensitivity_notes", "colour_balance", "saturation", "skin_tones",
         "recommended_lighting", "reciprocity", "processing", "scanning", "dye_or_mask_notes", "sharpness", "other"]

DATASHEET_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["stock_name", "manufacturer", "document_code", "document_date", "film_type", "iso", "process", "light_balance", "documented",
                 "manufacturer_descriptions", "absent_from_document"],
    "properties": {
        "stock_name": {"type": "string"}, "manufacturer": {"type": "string"}, "document_code": {"type": "string"}, "document_date": {"type": "string"},
        "film_type": {"type": "string", "enum": ["colour_negative", "colour_reversal", "bw_negative", "bw_chromogenic", "other"]},
        "iso": {"type": "string"}, "process": {"type": "string"}, "light_balance": {"type": "string"},
        "documented": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["property", "value", "basis", "page", "quote"],
                                                  "properties": {"property": {"type": "string", "enum": PROPS}, "value": {"type": "string"},
                                                                 "basis": {"type": "string", "enum": ["stated_in_text", "table", "read_from_graph"]},
                                                                 "page": {"type": "integer"}, "quote": {"type": "string"}}}},
        "manufacturer_descriptions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["phrase", "page"],
                                                                 "properties": {"phrase": {"type": "string"}, "page": {"type": "integer"}}}},
        "absent_from_document": {"type": "array", "items": {"type": "string", "enum": PROPS}},
    },
}

DATASHEET_SYSTEM = """You extract facts from one photographic film manufacturer's technical data sheet (attached PDF) into a fixed JSON schema.
Rules:
- Use only this document. Never add knowledge from elsewhere, even if you believe it.
- `documented`: one entry per property the document gives. `basis` is `stated_in_text` or `table` when the value is printed, and
  `read_from_graph` when you estimate it from a plotted curve (characteristic curves, spectral sensitivity, MTF, spectral dye density).
  Graph read-offs are approximate: give values with "about" and the axis units (e.g. "blue-sensitive layer peaks at about 440 nm").
  `quote`: up to 20 words copied from the page that supports the entry, or the figure caption for a graph read-off.
- For characteristic curves, describe shape (toe length, straight-line slope relative to the other layers, shoulder), the density range
  covered, and whether the three colour layers run parallel. For spectral sensitivity, give each layer's approximate peak and any overlap.
- `manufacturer_descriptions`: the document's evaluative marketing language (e.g. "natural skin tones", "vivid colours"), verbatim, short.
  These are subjective claims, kept separate from measurements.
- `absent_from_document`: properties from the enum that this document does not address.
- Page numbers are 1-based pages of the attached PDF."""

LUTSIG_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["discriminative_features", "confusable_pairs", "hard_negatives", "datasheet_links", "experiments"],
    "properties": {
        "discriminative_features": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["feature", "why", "status"],
                                                               "properties": {"feature": {"type": "string"}, "why": {"type": "string"},
                                                                              "status": {"type": "string", "enum": ["hypothesis", "supported_by_given_measurements"]}}}},
        "confusable_pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["a", "b", "distinguishing_measurement", "status"],
                                                        "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "distinguishing_measurement": {"type": "string"},
                                                                       "status": {"type": "string", "enum": ["hypothesis", "supported_by_given_measurements"]}}}},
        "hard_negatives": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["anchor", "negative", "reason"],
                                                      "properties": {"anchor": {"type": "string"}, "negative": {"type": "string"}, "reason": {"type": "string"}}}},
        "datasheet_links": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["stock", "datasheet_fact", "lut_measurement", "defensible", "caveat"],
                                                       "properties": {"stock": {"type": "string"}, "datasheet_fact": {"type": "string"}, "lut_measurement": {"type": "string"},
                                                                      "defensible": {"type": "boolean"}, "caveat": {"type": "string"}}}},
        "experiments": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["question", "procedure", "prediction"],
                                                   "properties": {"question": {"type": "string"}, "procedure": {"type": "string"}, "prediction": {"type": "string"}}}},
    },
}

LUTSIG_SYSTEM = """You are helping design falsifiable experiments for a film-look matching system. You receive (1) measured signatures of film-emulation
LUTs grouped by the stock their authors claim (numbers computed by applying each LUT to standard patches: grey ramp, ColorChecker, hue wheel),
(2) facts extracted from manufacturer datasheets with their basis, and (3) measured confusion between LUT classes.
Rules: every claim you make is a hypothesis unless the given numbers directly show it; say `supported_by_given_measurements` only then.
A LUT is its author's subjective approximation; a datasheet describes the negative or slide, not a scanned and inverted digital image,
so link datasheet facts to LUT measurements only where the physics carries through scanning and inversion, and give the caveat.
Prefer measurements that can be computed from a single finished photograph without the original. Be concrete and brief."""


def _ledger(task: str, message, cost: float, item: str) -> None:
    u = message.usage
    usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    P.append_jsonl(P.LEDGER, {"ts": P.now(), "task": task, "phase": "c08", "tier": "check", "bucket": "film", "model": MODEL, "mode": "sync", "batch_id": None,
                              "requests": 1, **usage, "cost_usd": round(cost, 6), "outcome": {"item": item, "stop": message.stop_reason}, "estimate_usd": None})


def _call(task: str, item: str, system: str, content: list, schema: dict, est: float, max_tokens: int = 6000) -> dict:
    path = OUT / task / f"{item}.json"
    if path.exists():
        return json.loads(path.read_text())
    c = P.cfg()
    P.guard(est, c, "film")
    client = P.client_or_none()
    if client is None:
        raise SystemExit("no ANTHROPIC_API_KEY reachable")
    msg = client.messages.create(model=MODEL, max_tokens=max_tokens, system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                 output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
                                 messages=[{"role": "user", "content": content}])
    u = msg.usage
    usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    cost = P.price(usage, MODEL, batch=False, c=c)
    _ledger(f"c08_{task}", msg, cost, item)
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"_invalid": text[:2000]}
    row = {"item": item, "model": MODEL, "served_model": msg.model, "stop_reason": msg.stop_reason, "usage": usage, "cost_usd": round(cost, 6), "created": P.now(),
           "provenance": "Claude (Sonnet) extraction or hypothesis; offline research only", "result": data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1, ensure_ascii=False))
    return row


def cmd_datasheets(a) -> None:
    P.require_paid(a)
    pdfs = sorted(p for p in SHEETS.glob("*.pdf") if "ProfessionalFilmDataGuide" not in p.name)
    total = 0.0
    for pdf in pdfs:
        item = pdf.stem
        data = base64.standard_b64encode(pdf.read_bytes()).decode()
        content = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
                   {"type": "text", "text": "Extract this data sheet into the schema. Document file name (for your reference only): " + pdf.name}]
        row = _call("datasheet", item, DATASHEET_SYSTEM, content, DATASHEET_SCHEMA, est=0.15)
        total += row.get("cost_usd", 0)
        r = row["result"]
        print(f"{item:36s} {r.get('stock_name', '?')[:28]:28s} documented={len(r.get('documented', []))} cost={row['cost_usd']:.4f}", flush=True)
    print("datasheets total cost this listing (cached rows included):", round(total, 4))


def cmd_lutsig(a) -> None:
    P.require_paid(a)
    payload = json.loads(Path(a.input).read_text())
    content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    row = _call("lutsig", a.item, LUTSIG_SYSTEM, content, LUTSIG_SCHEMA, est=0.5, max_tokens=8000)
    print(json.dumps({k: len(v) for k, v in row["result"].items() if isinstance(v, list)}), "cost", row["cost_usd"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("datasheets", "lutsig"):
        s = sub.add_parser(name)
        s.add_argument("--no-dry-run", action="store_true"); s.add_argument("--confirm-paid", action="store_true")
        if name == "lutsig":
            s.add_argument("--input", default="data/c08/lutsig_input.json"); s.add_argument("--item", default="v1")
    a = ap.parse_args()
    {"datasheets": cmd_datasheets, "lutsig": cmd_lutsig}[a.cmd](a)


if __name__ == "__main__":
    main()
