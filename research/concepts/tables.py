"""Markdown tables for docs/research/badge-concepts-local.md from the result JSONs (printed, not written).

  python -m research.concepts.tables
"""
from __future__ import annotations

import json

from research.concepts.common import REPORTS
from research.concepts.concept_map import UNSUPPORTED


def f(x, d=3):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def main() -> None:
    H = json.loads((REPORTS / "heads_results.json").read_text())["heads"]
    S = json.loads((REPORTS / "siglip_results.json").read_text()) if (REPORTS / "siglip_results.json").exists() else {}
    print("| concept | `_WORDS` key | test pos | test P | Wilson 95 % LB | test R | FPR generic | P at generic prevalence | threshold | decision | SigLIP test P / R | SigLIP decision |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(H, key=lambda h: (H[h]["decision"] != "shipped", h))
    for h in order:
        r = H[h]; t = r.get("test", {}); s = S.get(h, {}); st = s.get("test", {})
        print(f"| {h} | {r['key']} | {t.get('n_pos', '–')} | {f(t.get('precision'))} | {f((t.get('precision_wilson95') or [None])[0])} | "
              f"{f(t.get('recall'))} | {f(t.get('fpr_generic'), 4)} | {f(t.get('precision_at_generic_prevalence'))} | {f(r.get('threshold'))} | "
              f"**{r['decision']}**{' (' + r['reason'] + ')' if r.get('reason') else ''} | {f(st.get('precision'))} / {f(st.get('recall'))} | {s.get('decision', '–')} |")
    for k, why in UNSUPPORTED.items():
        print(f"| {k} | {'lesson_frame' if k == 'frame' else k} | – | – | – | – | – | – | – | **unsupported** ({why}) | – | – |")


if __name__ == "__main__":
    main()
