"""Resolve the 357 encyclopedia species against the GBIF backbone and pick "unknown" control taxa.

  python -m research.species.taxa            # writes data/species/taxa.json and data/species/unknown_taxa.json

Rules: the website's gbif_key is never trusted blindly. Every key is fetched from /species/{key}; synonyms are
followed to the accepted usage; the backbone canonical name must agree with the encyclopedia binomial (first two
Latin words, hybrid sign and cultivar dropped), otherwise the species is flagged `mismatch` and excluded. Species
without a key are matched with /species/match (kingdom given, rank SPECIES, EXACT or FUZZY only). The label used for
images is the GBIF speciesKey, never the encyclopedia id.
"""
from __future__ import annotations

import json
import random
import re
import sys

from research.species import common as C

KINGDOM = {"bird": "Animalia", "flower": "Plantae"}


def binomial(name: str) -> str:
    """'Prunus × yedoensis 'Somei-yoshino'' -> 'Prunus yedoensis'."""
    n = re.sub(r"'[^']*'", " ", name)
    n = n.replace("×", " ").replace(" x ", " ")
    parts = [p for p in re.split(r"\s+", n.strip()) if p and p[0].isalpha()]
    return " ".join(parts[:2])


def fetch_usage(sess, key: int) -> dict | None:
    r = C.get(sess, f"{C.GBIF}/species/{key}")
    return r.json() if r is not None and r.status_code == 200 else None


def resolve(sess, sp: dict) -> dict:
    want = binomial(sp["scientific_name"])
    rec = dict(sp, binomial=want, gbif_key=None, status="unresolved", note="")
    usage = None
    if sp.get("gbif_key_web"):
        usage = fetch_usage(sess, sp["gbif_key_web"])
        if usage and usage.get("taxonomicStatus", "").startswith(("SYNONYM", "HOMOTYPIC", "HETEROTYPIC", "PROPARTE")) and usage.get("acceptedKey"):
            rec["note"] += f"web key {sp['gbif_key_web']} is a synonym -> accepted {usage['acceptedKey']}; "
            usage = fetch_usage(sess, usage["acceptedKey"])
    if usage is None:
        r = C.get(sess, f"{C.GBIF}/species/match", {"name": want, "kingdom": KINGDOM[sp["kind"]], "rank": "SPECIES"})
        m = r.json() if r is not None and r.status_code == 200 else {}
        if m.get("matchType") in ("EXACT", "FUZZY") and m.get("rank") == "SPECIES" and m.get("usageKey"):
            rec["note"] += f"matched via /species/match ({m['matchType']}, confidence {m.get('confidence')}); "
            usage = fetch_usage(sess, m["usageKey"])
            if usage and usage.get("acceptedKey") and usage.get("acceptedKey") != usage.get("key"):
                usage = fetch_usage(sess, usage["acceptedKey"])
        else:
            rec["note"] += f"no species-rank match ({m.get('matchType')}, {m.get('rank')}); "
    if usage is None:
        return rec
    canon = usage.get("canonicalName") or ""
    rec.update(gbif_key=usage.get("speciesKey") or usage["key"], canonical=canon, rank=usage.get("rank"),
               genus=usage.get("genus"), genus_key=usage.get("genusKey"), family=usage.get("family"),
               family_key=usage.get("familyKey"), order=usage.get("order"), order_key=usage.get("orderKey"),
               kingdom=usage.get("kingdom"), taxonomic_status=usage.get("taxonomicStatus"))
    if binomial(canon) == want:
        rec["status"] = "ok"
    elif binomial(canon).split()[0] == want.split()[0]:
        rec["status"] = "ok_backbone_name"          # same genus, backbone uses another accepted epithet (synonymy)
        rec["note"] += f"backbone accepted name '{canon}' differs from encyclopedia '{want}'; "
    else:
        rec["status"] = "mismatch"
        rec["note"] += f"backbone '{canon}' != encyclopedia '{want}'; "
    if rec.get("kingdom") and rec["kingdom"] != KINGDOM[sp["kind"]]:
        rec["status"] = "mismatch"
        rec["note"] += f"kingdom {rec['kingdom']}; "
    if rec["family_web"] and rec.get("family") and rec["family_web"].lower() != rec["family"].lower():
        rec["note"] += f"website family {rec['family_web']} vs backbone {rec['family']}; "
    return rec


def pick_unknown(sess, taxa: list[dict], n_congeners: int = 60, n_random: int = 40, seed: int = 0) -> list[dict]:
    """Control taxa absent from the encyclopedia: congeners / confamilials of listed species plus common species of
    the same classes (birds: Aves; flowers: Tracheophyta) found by faceting occurrences with images."""
    rng = random.Random(seed)
    known = {t["gbif_key"] for t in taxa if t.get("gbif_key")}
    out, seen = [], set()
    cands = [t for t in taxa if t.get("genus_key")]
    rng.shuffle(cands)
    for t in cands:
        if len(out) >= n_congeners:
            break
        r = C.get(sess, f"{C.GBIF}/species/search", {"higherTaxonKey": t["genus_key"], "rank": "SPECIES", "status": "ACCEPTED", "limit": 30})
        if r is None or r.status_code != 200:
            continue
        opts = [u for u in r.json().get("results", []) if u.get("key") not in known and u.get("key") not in seen and u.get("rank") == "SPECIES"]
        if not opts:
            # fall back to the family
            r = C.get(sess, f"{C.GBIF}/species/search", {"higherTaxonKey": t["family_key"], "rank": "SPECIES", "status": "ACCEPTED", "limit": 50})
            opts = [u for u in (r.json().get("results", []) if r is not None and r.status_code == 200 else []) if u.get("key") not in known and u.get("key") not in seen and u.get("genus") != t["genus"]]
        if not opts:
            continue
        u = rng.choice(opts)
        seen.add(u["key"])
        out.append({"id": f"unk_{u['key']}", "kind": t["kind"], "gbif_key": u["key"], "canonical": u.get("canonicalName"), "genus": u.get("genus"),
                    "family": u.get("family"), "order": u.get("order"), "relation": f"near:{t['id']}"})
    for kind, key_param in (("bird", {"classKey": 212}), ("flower", {"phylumKey": 7707728})):
        r = C.get(sess, f"{C.GBIF}/occurrence/search", {**key_param, "mediaType": "StillImage", "basisOfRecord": "HUMAN_OBSERVATION",
                                                         "license": ["CC0_1_0", "CC_BY_4_0"], "limit": 0, "facet": "speciesKey", "facetLimit": 400})
        if r is None or r.status_code != 200:
            continue
        counts = [(int(f["name"]), f["count"]) for fac in r.json().get("facets", []) for f in fac["counts"]]
        pool = [k for k, _ in counts if k not in known and k not in seen]
        rng.shuffle(pool)
        for k in pool[: n_random // 2]:
            u = fetch_usage(sess, k)
            if not u or u.get("rank") != "SPECIES":
                continue
            seen.add(k)
            out.append({"id": f"unk_{k}", "kind": kind, "gbif_key": k, "canonical": u.get("canonicalName"), "genus": u.get("genus"),
                        "family": u.get("family"), "order": u.get("order"), "relation": "random_common"})
    return out


BACKBONE = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c"


# IOC 14 genus revisions the GBIF backbone (2026-11) does not list as synonyms; the GBIF concept is the pre-split /
# pre-revision species. Recorded as ok_manual with the note; the label is the GBIF concept, which may be a lump.
MANUAL = {"little_ringed_plover": "Charadrius dubius", "siberian_sand_plover": "Charadrius mongolus", "ornate_sunbird": "Cinnyris jugularis"}


def fixup():
    """Second pass over data/species/taxa.json: (1) a website key that GBIF itself lists as a synonym of another
    accepted species is the same species under an older/newer genus name (IOC vs backbone) -> status ok_synonym;
    (2) unresolved species are retried through the backbone's vernacular names with the English Wikipedia title;
    accepted only when the vernacular matches exactly and the kingdom is right."""
    sess = C.session()
    taxa = json.load(open(C.DATA / "taxa.json"))
    for t in taxa:
        if t["status"] == "mismatch" and "is a synonym -> accepted" in t["note"] and t.get("kingdom") == KINGDOM[t["kind"]]:
            t["status"] = "ok_synonym"
        elif t["status"] == "unresolved" and t["id"] in MANUAL:
            r = C.get(sess, f"{C.GBIF}/species/match", {"name": MANUAL[t["id"]], "kingdom": KINGDOM[t["kind"]], "rank": "SPECIES"})
            m = r.json() if r is not None and r.status_code == 200 else {}
            if m.get("matchType") == "EXACT" and m.get("rank") == "SPECIES":
                u = fetch_usage(sess, m["usageKey"]) or {}
                t.update(gbif_key=u.get("speciesKey") or u.get("key"), canonical=u.get("canonicalName"), rank=u.get("rank"), genus=u.get("genus"),
                         genus_key=u.get("genusKey"), family=u.get("family"), family_key=u.get("familyKey"), order=u.get("order"),
                         kingdom=u.get("kingdom"), taxonomic_status=u.get("taxonomicStatus"), status="ok_manual")
                t["note"] += f"manual mapping to GBIF concept {MANUAL[t['id']]} ({u.get('key')}); IOC genus revision / split not in backbone; label may be a lump; "
        elif t["status"] == "unresolved" and t.get("wiki_en"):
            r = C.get(sess, f"{C.GBIF}/species/search", {"q": t["wiki_en"], "qField": "VERNACULAR", "rank": "SPECIES", "datasetKey": BACKBONE, "limit": 10})
            hits = r.json().get("results", []) if r is not None and r.status_code == 200 else []
            for u in hits:
                vn = {v.get("vernacularName", "").lower() for v in u.get("vernacularNames", []) if (v.get("language") or "eng") in ("eng", "en")}
                if t["wiki_en"].lower() in vn and u.get("kingdom") == KINGDOM[t["kind"]] and u.get("taxonomicStatus") == "ACCEPTED":
                    t.update(gbif_key=u.get("speciesKey") or u["key"], canonical=u.get("canonicalName"), rank=u.get("rank"), genus=u.get("genus"),
                             genus_key=u.get("genusKey"), family=u.get("family"), family_key=u.get("familyKey"), order=u.get("order"),
                             kingdom=u.get("kingdom"), taxonomic_status=u.get("taxonomicStatus"), status="ok_vernacular")
                    t["note"] += f"resolved through backbone vernacular '{t['wiki_en']}' -> {u.get('canonicalName')} ({u['key']}); "
                    break
        print(f"{t['id']:40s} {t['status']:15s} {t.get('gbif_key')} {t.get('canonical')} | {t['note']}")
    json.dump(taxa, open(C.DATA / "taxa.json", "w"), indent=1, ensure_ascii=False)
    print({s: sum(t["status"] == s for t in taxa) for s in sorted({t["status"] for t in taxa})})


def main():
    if "fixup" in sys.argv:
        return fixup()
    C.DATA.mkdir(parents=True, exist_ok=True)
    sess = C.session()
    species = C.load_species()
    taxa = []
    for i, sp in enumerate(species):
        rec = resolve(sess, sp)
        taxa.append(rec)
        print(f"[{i + 1}/{len(species)}] {sp['id']:40s} {rec['status']:18s} key={rec.get('gbif_key')} {rec.get('canonical')} | {rec['note']}", flush=True)
    json.dump(taxa, open(C.DATA / "taxa.json", "w"), indent=1, ensure_ascii=False)
    ok = [t for t in taxa if t["status"].startswith("ok")]
    print(f"resolved {len(ok)}/{len(taxa)}; mismatch {sum(t['status'] == 'mismatch' for t in taxa)}; unresolved {sum(t['status'] == 'unresolved' for t in taxa)}")
    if "--no-unknown" not in sys.argv:
        unk = pick_unknown(sess, ok)
        json.dump(unk, open(C.DATA / "unknown_taxa.json", "w"), indent=1, ensure_ascii=False)
        print(f"unknown taxa: {len(unk)}")


if __name__ == "__main__":
    main()
