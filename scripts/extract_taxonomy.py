#!/usr/bin/env python
"""Extract photographer and film-stock taxonomies from the FilmLook evaluation app, read-only.

  python scripts/extract_taxonomy.py --source ~/misc/film-evaluation

Reads JSON seed and knowledge files, plus the literal FILM_CATALOG entries of filmlook/film_catalog.py
(parsed with `ast`, never imported or executed). Writes only inside this repository:
  taxonomy/photographers.v1.json   139 profiles: stable ids, en/zh names, the source's own descriptive fields
  taxonomy/film_stocks.v1.json     deduplicated stock entities, with source records kept per entity
  taxonomy/film_aesthetics.v1.json the app's aesthetic directions ("X-like"), each linked to a reference stock
  taxonomy/SOURCE.json             source path, git commit, and per-file sha256

Nothing is invented. Fields missing in the source stay absent: no Japanese names, aliases or reference images
exist there. Name variants are merged only on an exact normalised key or on an alias the source itself
declares. Near-matches are listed for review, never merged automatically.
"""
import argparse
import ast
import difflib
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "taxonomy"
VERSION = "taxonomy-db-v1"

PHOTOGRAPHER_FIELDS = ["name_en", "name_zh", "nationality", "nationality_zh", "born", "died", "active_period",
                       "active_period_zh", "magnum_status", "magnum_note", "category", "genres", "subjects", "palette",
                       "intro_en", "intro_zh", "traits_en", "traits_zh", "signature_en", "signature_zh", "series",
                       "keywords_en", "keywords_zh", "explore_en", "explore_zh", "method_en", "method_zh", "sources",
                       "style_vector", "emphasis"]
BRANDS = {"fuji": "fujifilm", "fujifilm": "fujifilm", "agfaphoto": "agfa", "agfa": "agfa", "cinestill": "cinestill",
          "harman": "ilford"}
NON_STOCK_KINDS = {"simulation": "digital_simulation", "print_medium": "print_medium", "instant": "instant_film"}
PROCESS_MAP = {"Color Negative": "color_negative", "color_negative": "color_negative", "Color Slide": "color_reversal",
               "color_slide": "color_reversal", "Black & White": "black_and_white", "black_and_white": "black_and_white",
               "Black & White Slide": "bw_reversal"}
GENERIC = {"柯达", "黑白胶卷", "黑白盘片", "X光底片", "Generic", "Unknown"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm_key(name: str) -> str:
    """Exact-merge key: case, accents, punctuation, brand spelling, 'plus', '-like' and spaces ignored."""
    s = unicodedata.normalize("NFKC", name).lower().strip()
    s = re.sub(r"[-\s]like$", "", s)
    toks = re.findall(r"[a-z0-9]+|[^\x00-\x7f]+", s)
    if toks and toks[0] in BRANDS:
        toks[0] = BRANDS[toks[0]]
    toks = [t for t in toks if t not in ("plus", "film", "professional")]
    return "".join(toks)


def photographers(src: Path) -> list[dict]:
    out = []
    for f in sorted((src / "filmlook/photographer_dna/seed_data").glob("*.json")):
        if f.name == "axes.json":
            continue
        j = json.loads(f.read_text())
        for p in j["photographers"]:
            e = {"id": p["id"], "names": {"en": p["name_en"], "zh": p["name_zh"]}, "aliases": []}
            e.update({k: p[k] for k in PHOTOGRAPHER_FIELDS if k in p and k not in ("name_en", "name_zh")})
            e["provenance"] = {"file": f"filmlook/photographer_dna/seed_data/{f.name}", "group": j["group"],
                               "style_vector_note": "editorial estimates of a body of work, not measurements"}
            out.append(e)
    ids = [e["id"] for e in out]
    assert len(ids) == len(set(ids)), "duplicate photographer ids in source"
    return out


def catalog_entries(src: Path) -> tuple[list[dict], set[str]]:
    """Literal FilmAesthetic(...) calls and the declared alias set, via ast (no execution)."""
    tree = ast.parse((src / "filmlook/film_catalog.py").read_text())
    entries, aliases = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "FilmAesthetic":
            try:
                entries.append({k.arg: ast.literal_eval(k.value) for k in node.keywords})
            except ValueError:
                pass                            # generated entries (from reference_knowledge) are rebuilt below
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "aliases" for t in node.targets):
            aliases |= set(ast.literal_eval(node.value))
    return entries, aliases


def film(src: Path):
    data = src / "filmlook/data"
    inv = json.loads((data / "film_stocks.json").read_text())
    know = json.loads((data / "reference_knowledge.json").read_text())
    look = json.loads((data / "film_look_notes.json").read_text())
    color = json.loads((data / "film_color_notes.json").read_text())
    cat, declared_aliases = catalog_entries(src)

    groups: dict[str, dict] = {}

    def entity(name: str) -> dict:
        k = norm_key(name)
        if k not in groups:
            groups[k] = {"key": k, "names": set(), "records": []}
        groups[k]["names"].add(name)
        return groups[k]

    for s in inv:
        entity(s["name"])["records"].append({"source": "film_stocks.json", **s})
    for s in know["stocks"]:
        entity(s["name"])["records"].append({"source_file": "reference_knowledge.json", **s})
    # catalog labels ("Gold-like") resolve to the stock they refer to, as the app itself does
    label_to_ref = {c["name"]: c["reference_name"] for c in cat}
    label_to_ref.update({s["name"] + "-like": s["name"] for s in know["stocks"] if s["status"] == "subjective_profile"})
    for s in look["films"]:
        label = s.get("catalog_name") or s["name"]
        ref = label_to_ref.get(label) or label_to_ref.get(s["name"]) or s["name"]
        entity(ref)["records"].append({"source_file": "film_look_notes.json", "catalog_name": label,
                                       "note_name": s["name"], "source": s["source"], "look": s["look"]})
    for s in color["films"]:
        entity(s["name"])["records"].append({"source_file": "film_color_notes.json", **s})
    for c in cat:
        entity(c["reference_name"])["records"].append({"source_file": "film_catalog.py", **c})

    # merges the SOURCE declares: catalog aliases point at the catalog reference stock
    merged_by_alias = []
    for alias in sorted(declared_aliases):
        ak = norm_key(alias)
        target = next((c["reference_name"] for c in cat if norm_key(c["reference_name"]).startswith(ak[:8])
                       and difflib.SequenceMatcher(None, ak, norm_key(c["reference_name"])).ratio() > 0.75), None)
        if target and ak in groups and norm_key(target) in groups and ak != norm_key(target):
            g, t = groups.pop(ak), groups[norm_key(target)]
            t["names"] |= g["names"]
            t["records"] += g["records"]
            merged_by_alias.append([alias, target])

    stocks = []
    for k, g in sorted(groups.items()):
        recs = g["records"]
        kinds = {r.get("kind") for r in recs if r.get("kind") and r.get("kind") != "unknown"}
        types = {r.get("type") for r in recs if r.get("type")}
        cats = {r.get("category") for r in recs if r.get("category")}
        names = sorted(g["names"])
        if any(n in GENERIC or "未分类" in n for n in names):
            etype = "generic_or_brand_only"
        elif kinds & set(NON_STOCK_KINDS):
            etype = NON_STOCK_KINDS[sorted(kinds & set(NON_STOCK_KINDS))[0]]
        else:
            etype = "film_stock"
        raw_process = kinds | types | cats
        process = sorted({PROCESS_MAP[x] for x in raw_process if x in PROCESS_MAP})
        isos = sorted({str(r["iso"]) for r in recs if r.get("iso") not in (None, "")})
        brands = sorted({r["brand"] for r in recs if r.get("brand")} | {r["manufacturer"] for r in recs if r.get("manufacturer")})
        eid = re.sub(r"[^a-z0-9]+", "_", names[0].lower()).strip("_") or "s_" + hashlib.md5(k.encode()).hexdigest()[:8]
        stocks.append({"id": eid, "canonical_name": max(names, key=lambda n: (sum(r.get("name") == n for r in recs), len(n))),
                       "names": names, "entity_type": etype, "brand": brands, "iso": isos, "process": process,
                       "process_conflict": len(process) > 1,
                       "has_description": any(r.get("look") or r.get("notes") or r.get("character") or r.get("description")
                                              for r in recs),
                       "has_ratings": any("ratings" in r for r in recs), "records": recs})
    ids = defaultdict(int)
    for s in stocks:
        ids[s["id"]] += 1
        if ids[s["id"]] > 1:
            s["id"] = f"{s['id']}_{ids[s['id']]}"
    # near-duplicates for human review (never auto-merged)
    keys = [(s["id"], norm_key(s["canonical_name"])) for s in stocks if s["entity_type"] == "film_stock"]
    review = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = difflib.SequenceMatcher(None, keys[i][1], keys[j][1]).ratio()
            if r >= 0.88:
                review.append([keys[i][0], keys[j][0], round(r, 3)])
    aesthetics = [{"id": c["id"], "label": c["name"], "display_name": c["display_name"],
                   "reference_stock": c["reference_name"],
                   "reference_stock_id": next((s["id"] for s in stocks if c["reference_name"] in s["names"]), None),
                   "category": c["category"], "monochrome": c.get("monochrome", False), "character": c["character"],
                   "suits": c["suits"], "poor_fit": c["poor_fit"], "source": "film_catalog.py (literal entries)"}
                  for c in cat]
    gen = [s for s in know["stocks"] if s["status"] == "subjective_profile" and s["name"] not in declared_aliases
           and s["name"].casefold() not in {c["reference_name"].casefold() for c in cat}]
    for s in gen:
        aesthetics.append({"id": re.sub(r"[^a-z0-9]+", "_", s["name"].lower()).strip("_") + "_like",
                           "label": s["name"] + "-like", "display_name": s["name"], "reference_stock": s["name"],
                           "reference_stock_id": next((x["id"] for x in stocks if s["name"] in x["names"]), None),
                           "category": s["kind"], "monochrome": s["kind"] == "black_and_white",
                           "ratings_1_to_5": s["ratings"], "ratings_source": s["source"],
                           "source": "reference_knowledge.json subjective_profile (as film_catalog._reference_aesthetics)"})
    return stocks, aesthetics, {"merged_by_declared_alias": merged_by_alias, "near_duplicates_for_review": review,
                                "sources": {"reference_knowledge": know["sources"], "film_look_notes": look["sources"],
                                            "film_color_notes": color["source"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path.home() / "misc" / "film-evaluation"))
    a = ap.parse_args()
    src = Path(a.source).expanduser()
    OUT.mkdir(exist_ok=True)
    ph = photographers(src)
    stocks, aesthetics, notes = film(src)
    axes = json.loads((src / "filmlook/photographer_dna/seed_data/axes.json").read_text())
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    files = sorted([*(src / "filmlook/photographer_dna/seed_data").glob("*.json"),
                    *[src / "filmlook/data" / n for n in ("film_stocks.json", "reference_knowledge.json",
                                                         "film_look_notes.json", "film_color_notes.json")],
                    src / "filmlook/film_catalog.py"])
    home = Path.home()
    shown = "~/" + str(src.relative_to(home)) if src.is_relative_to(home) else str(src)
    meta = {"taxonomy_version": VERSION, "source_path": shown, "source_git_commit": commit, "source_dirty": dirty,
            "files": {str(f.relative_to(src)): sha(f) for f in files},
            "note": "Read-only extraction. Descriptions are the FilmLook app's own words; style vectors are editorial "
                    "estimates; film ratings are third-party subjective opinions. No images were copied."}
    (OUT / "photographers.v1.json").write_text(json.dumps({"taxonomy_version": VERSION, "axes": axes,
                                                           "photographers": ph}, ensure_ascii=False, indent=1))
    (OUT / "film_stocks.v1.json").write_text(json.dumps({"taxonomy_version": VERSION, "notes": notes,
                                                         "stocks": stocks}, ensure_ascii=False, indent=1))
    (OUT / "film_aesthetics.v1.json").write_text(json.dumps({"taxonomy_version": VERSION,
                                                             "aesthetics": aesthetics}, ensure_ascii=False, indent=1))
    (OUT / "SOURCE.json").write_text(json.dumps(meta, indent=1))
    by = defaultdict(int)
    for s in stocks:
        by[s["entity_type"]] += 1
    print(json.dumps({"photographers": len(ph), "stock_entities": len(stocks), "by_type": dict(by),
                      "aesthetic_directions": len(aesthetics), "merged_by_declared_alias": notes["merged_by_declared_alias"],
                      "near_duplicates_for_review": len(notes["near_duplicates_for_review"]),
                      "source_commit": commit, "source_dirty": dirty}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
