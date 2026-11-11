"""Structured style annotation of legally usable images with the Anthropic Message Batches API.

  python -m research.claude_annotations.pipeline status
  python -m research.claude_annotations.pipeline plan   --task bank --phase pilot            # free: counts tokens, estimates cost
  python -m research.claude_annotations.pipeline smoke  --task bank --no-dry-run --confirm-paid   # one synchronous request
  python -m research.claude_annotations.pipeline submit --task bank --phase pilot --no-dry-run --confirm-paid
  python -m research.claude_annotations.pipeline collect [--wait]
  python -m research.claude_annotations.pipeline review-pilot
  python -m research.claude_annotations.pipeline submit --task bank --phase full --no-dry-run --confirm-paid
  python -m research.claude_annotations.pipeline submit --task bank --tier check --phase full ... # Sonnet: hard cases + sample

Cost controls (config.yaml): a hard total cap and a smaller pilot cap that applies until `review-pilot` has written
its report; every submission is refused when recorded spend + the estimate for the new requests would pass the cap;
spend is what the API's usage fields report (input, cache write, cache read, output) priced from the table in the
config, with the batch discount. Images are deduplicated by content hash before a request is built, resized once,
and an item that already has a valid annotation from the same model and prompt is never sent again. Nothing runs
without --no-dry-run --confirm-paid. The key is read from the environment by the SDK and never written anywhere.

Prompt caching: the instructions and vocabulary are one system block marked `cache_control`, identical in every
request, so requests after the first read it from the cache (usage.cache_read_input_tokens in the ledger).

Tasks: `bank` = the public-domain / CC0 photographer reference bank (refdb manifest, eligible, downloaded,
non-duplicate). User uploads and in-copyright photographs are never read by this module.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import schema as S  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "annotations" / "claude_research"
LEDGER, BATCHES = OUT / "usage.jsonl", OUT / "batches.jsonl"
REPORTS = ROOT / "reports" / "claude"


def cfg() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- cost

def price(usage: dict, model: str, batch: bool, c: dict | None = None) -> float:
    c = c or cfg()
    p, m = c["prices_per_mtok"][model], c["multipliers"]
    usd = (usage.get("input_tokens", 0) * p["input"] + usage.get("cache_creation_input_tokens", 0) * p["input"] * m["cache_write_5m"]
           + usage.get("cache_read_input_tokens", 0) * p["input"] * m["cache_read"] + usage.get("output_tokens", 0) * p["output"]) / 1e6
    return usd * (m["batch"] if batch else 1.0)


def spent() -> float:
    return round(sum(r["cost_usd"] for r in read_jsonl(LEDGER)), 6)


def reserved() -> float:
    """Estimates of batches that were submitted and not collected yet: they count against the cap until they land."""
    return round(sum(b["estimate_usd"] for b in read_jsonl(BATCHES) if not b.get("collected")), 6)


def cap(c: dict) -> tuple[float, str]:
    b = c["budget"]
    if (REPORTS / "pilot_review.json").exists():
        return float(b["total_usd"]), "total"
    return float(b["pilot_usd"]), "pilot"


# ---------------------------------------------------------------- tasks

def bank_items() -> list[dict]:
    from filmlook_ml.refdb import manifest
    with manifest.connect() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT work_id, photographer_id, sha256 FROM works WHERE rights_status = 'eligible' AND rights_class IN ('public_domain', 'cc0') "
            "AND sha256 IS NOT NULL AND duplicate_of IS NULL ORDER BY work_id")]
    seen, items = set(), []
    for r in rows:
        if r["sha256"] in seen:  # identical bytes under two work ids: send once
            continue
        seen.add(r["sha256"])
        items.append({"item_id": r["work_id"], "group": r["photographer_id"], "path": str(manifest.image_path(r["work_id"]))})
    return items


def _synth_test() -> dict[str, dict[str, str]]:
    """scene id -> preset -> path, for the scene-disjoint test split of the synthetic preset set (CC-BY scenes)."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for r in read_jsonl(ROOT / "data" / "lut" / "synth" / "manifest.jsonl"):
        if r["split"] == "test" and r["variant"] == 0:
            out[r["scene_id"]][r["preset"]] = r["path"]
    return out


def lutdesc_items(n_scenes: int = 24, seed: int = 0, source: str = "catalog", tag: str = "lutdesc") -> list[dict]:
    """Forced choice: untouched render A and preset render B of one scene, four of the website's look descriptions
    (the applied preset's and three others), which one describes the change? Ground truth is the preset applied."""
    import random
    from research.claude_annotations import lut_tasks as L
    desc = L.descriptions(source)
    scenes = _synth_test(); items = []
    for scene in sorted(scenes, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())[:n_scenes]:
        for preset in sorted(desc):
            if preset not in scenes[scene] or "none" not in scenes[scene]:
                continue
            rng = random.Random(f"{seed}:{scene}:{preset}")
            options = [preset] + rng.sample([q for q in sorted(desc) if q != preset and desc[q] != desc[preset]], 3)
            rng.shuffle(options)
            items.append({"item_id": f"{tag}:{scene}:{preset}", "group": preset, "paths": [scenes[scene]["none"], scenes[scene][preset]],
                          "user": L.user_text([desc[o] for o in options]), "meta": {"options": options, "answer": str(options.index(preset) + 1)}})
    return items


def lutscene_items() -> list[dict]:
    """The untouched render of every synthetic test scene, annotated with the style schema, so that the film-look
    engine's errors can be broken down by what kind of scene it was."""
    return [{"item_id": f"lutscene:{scene}", "group": "scene", "path": by["none"]} for scene, by in sorted(_synth_test().items()) if "none" in by]


def _lut_task(source: str = "catalog", seed: int = 0, tag: str = "lutdesc") -> dict:
    from research.claude_annotations import lut_tasks as L
    return {"items": lambda: lutdesc_items(seed=seed, source=source, tag=tag), "system": L.SYSTEM_PROMPT, "user": "", "schema": L.json_schema, "validate": L.validate,
            "prompt_version": L.VERSION + ("" if source == "catalog" else f"+{source}")}


def filmset_items() -> list[dict]:
    """The film-versus-digital set of research/lut/film_likeness_train.py: CC-BY / CC0 photographs whose capture medium is
    recorded independently (scanner or camera EXIF, owner records). The label is never sent and never comes from the model."""
    seen, items = set(), []
    for r in read_jsonl(ROOT / "data" / "lut" / "film_likeness" / "manifest.jsonl"):
        key = Path(r["path"]).name
        if key in seen or not Path(r["path"]).exists():
            continue
        seen.add(key)
        items.append({"item_id": f"filmset:{r['dataset']}:{key}", "group": f"{r['split']}:{'film' if r['label'] else 'digital'}", "path": r["path"]})
    return items


TASKS = {"bank": {"items": bank_items, "system": S.SYSTEM_PROMPT, "user": S.USER_TEXT, "schema": S.json_schema, "validate": S.validate,
                  "prompt_version": S.VERSION},
         "lutscene": {"items": lutscene_items, "system": S.SYSTEM_PROMPT, "user": S.USER_TEXT, "schema": S.json_schema, "validate": S.validate,
                      "prompt_version": S.VERSION},
         "filmset": {"items": filmset_items, "system": S.SYSTEM_PROMPT, "user": S.USER_TEXT, "schema": S.json_schema, "validate": S.validate,
                     "prompt_version": S.VERSION},
         "lutdesc": _lut_task(),
         "lutdesc2": _lut_task(source="rewrite", seed=1, tag="lutdesc2")}  # rewritten descriptions, different scenes


def store_path(task: str, model: str) -> Path:
    return OUT / task / f"{model}.jsonl"


def annotated(task: str, model: str) -> dict[str, dict]:
    version = TASKS[task]["prompt_version"]
    return {r["item_id"]: r for r in read_jsonl(store_path(task, model)) if r.get("valid") and r.get("prompt_version") == version}


def pick(task: str, phase: str, tier: str, n: int | None, seed: int = 0) -> list[dict]:
    c = cfg(); items = TASKS[task]["items"]()
    cheap, check = c["models"]["cheap"], c["models"]["check"]
    if tier == "check_all":  # the check model on every item of the task (used for filmset after Sonnet measured 0.966 vs Haiku 0.878)
        chosen, model = list(items), check
    elif tier == "check":
        # Sonnet sees: every item Haiku could not annotate validly or marked low confidence, plus a fixed random
        # sample (for agreement). It never sees an item it already annotated.
        done = annotated(task, cheap)
        hard = {i for i, r in done.items() if r["annotation"].get("confidence") == "low" or task == "lutdesc"}  # lutdesc: the check model answers every item
        hard |= {r["item_id"] for r in read_jsonl(store_path(task, cheap)) if not r.get("valid")} - set(done)
        key = lambda it: hashlib.sha256(f"check:{seed}:{it['item_id']}".encode()).hexdigest()  # noqa: E731
        sample = {it["item_id"] for it in sorted(items, key=key)[: (n or 150)]}
        chosen = [it for it in items if it["item_id"] in hard | sample]
        model = check
    else:
        key = lambda it: hashlib.sha256(f"{seed}:{it['item_id']}".encode()).hexdigest()  # noqa: E731
        if phase == "pilot":  # stratified: the same number from every photographer
            per = defaultdict(list)
            for it in sorted(items, key=key):
                per[it["group"]].append(it)
            k = max(1, (n or 240) // max(1, len(per)))
            chosen = [it for g in sorted(per) for it in per[g][:k]]
        else:
            chosen = sorted(items, key=key)[: n] if n else items
        model = cheap
    have = annotated(task, model)
    pending = {cid for b in read_jsonl(BATCHES) if not b.get("collected") and b["model"] == model and b["task"] == task for cid in b["items"].values()}
    return [dict(it, model=model) for it in chosen if it["item_id"] not in have and it["item_id"] not in pending]


def encode_image(path: str, c: dict) -> tuple[str, str]:
    img = Image.open(path).convert("RGB")
    img.thumbnail((c["image"]["long_edge"],) * 2, Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=c["image"]["jpeg_quality"])
    return base64.standard_b64encode(buf.getvalue()).decode(), hashlib.sha256(buf.getvalue()).hexdigest()


def build(task: str, item: dict, c: dict) -> tuple[dict, str]:
    """Request parameters for one item and the hash of the image bytes that are sent."""
    encoded = [encode_image(path, c) for path in item.get("paths") or [item["path"]]]
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}} for b64, _ in encoded]
    content.append({"type": "text", "text": item.get("user") or TASKS[task]["user"]})
    p = params(task, item["model"], "", c)
    p["messages"] = [{"role": "user", "content": content}]
    return p, hashlib.sha256("".join(sha for _, sha in encoded).encode()).hexdigest()


def params(task: str, model: str, b64: str, c: dict) -> dict:
    t = TASKS[task]
    return {
        "model": model, "max_tokens": c["max_tokens"],
        "system": [{"type": "text", "text": t["system"], "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": c["effort"], "format": {"type": "json_schema", "schema": t["schema"]()}},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": t["user"]}]}],
    }


def estimate(task: str, items: list[dict], c: dict, client=None) -> dict:
    """Per-request token estimate: counted by the API when a client is available (free), a formula otherwise."""
    if not items:
        return {"n": 0, "usd": 0.0}
    model = items[0]["model"]
    first = (items[0].get("paths") or [items[0]["path"]])
    if client is not None:
        p, _ = build(task, items[0], c)
        tokens = client.messages.count_tokens(model=model, system=p["system"], messages=p["messages"]).input_tokens
        how = "messages.count_tokens on one request"
    else:
        img = Image.open(first[0]); img.thumbnail((c["image"]["long_edge"],) * 2)
        tokens = int(len(TASKS[task]["system"]) / 3.2 + len(first) * img.width * img.height / 500 + len(items[0].get("user", "")) / 3.2 + 600)  # schema + image + text, deliberately high
        how = "formula (no client)"
    sys_tokens = int(len(TASKS[task]["system"]) / 3.2)
    # Planning assumption: no cache hits at all (an upper bound; batch requests may be processed concurrently).
    per = price({"input_tokens": tokens, "output_tokens": c["est_output_tokens"]}, model, batch=True, c=c)
    return {"n": len(items), "model": model, "input_tokens_per_request": tokens, "system_tokens_approx": sys_tokens,
            "output_tokens_assumed": c["est_output_tokens"], "usd_per_request": round(per, 6), "usd": round(per * len(items), 4), "method": how}


BUCKET_OF_TASK = {"bank": "photographer", "filmset": "film", "lutdesc": "film", "lutdesc2": "film", "lutscene": "film", "lutsynth": "film"}


def bucket_of(task: str, tier: str) -> str:
    """Which allocation a request is charged to: the task's own for bulk annotation, `quality` for second-model
    checks, `advanced` for the strong model."""
    return {"check": "quality", "strong": "advanced"}.get(tier, BUCKET_OF_TASK.get(task, "advanced"))  # check_all is charged to the task's own bucket


def bucket_spent(bucket: str) -> float:
    done = sum(r["cost_usd"] for r in read_jsonl(LEDGER) if r.get("bucket") == bucket)
    open_ = sum(b["estimate_usd"] for b in read_jsonl(BATCHES) if not b.get("collected") and b.get("bucket") == bucket)
    return round(done + open_, 6)


def guard(extra: float, c: dict, bucket: str | None = None) -> None:
    """Refuse a submission that would pass the active cap, its bucket's allocation, or the credit's expiry date."""
    import datetime
    expires = datetime.date.fromisoformat(str(c["budget"].get("expires", "2999-12-31")))
    if datetime.datetime.now(datetime.timezone.utc).date() >= expires - datetime.timedelta(days=1):
        raise SystemExit(f"DEADLINE STOP: the credit expires on {expires}; nothing new is submitted from the day before")
    limit, name = cap(c)
    total = spent() + reserved() + extra
    if total > limit:
        raise SystemExit(f"BUDGET STOP: spent {spent():.4f} + reserved {reserved():.4f} + this run {extra:.4f} = {total:.4f} USD exceeds the {name} cap {limit:.2f}")
    if bucket and bucket in c["budget"].get("buckets", {}):
        used = bucket_spent(bucket)
        if used + extra > float(c["budget"]["buckets"][bucket]):
            raise SystemExit(f"BUDGET STOP: bucket {bucket} holds {used:.4f} + this run {extra:.4f} USD, over its allocation {c['budget']['buckets'][bucket]:.2f}")


def require_paid(a) -> None:
    if not (a.no_dry_run and a.confirm_paid):
        raise SystemExit("dry run: add --no-dry-run --confirm-paid to make paid API calls")


def key_source() -> str:
    """'env' | '.env' | 'missing'. Where the key comes from; the key itself is never returned, logged or stored.
    A key placed in a Claude Code settings.local.json "env" block reaches this process as 'env' once the Claude Code
    session that runs it has been restarted."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip() or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip():
        return "env"
    from filmlook_ml.labeling.teachers import api_key_status
    return api_key_status(ROOT)


def client_or_none():
    """An SDK client when a key is reachable: the environment first, then the repository's git-ignored .env."""
    import anthropic
    source = key_source()
    if source == "env":
        return anthropic.Anthropic()
    if source == ".env":
        from filmlook_ml.labeling.teachers import _key_from_dotenv
        return anthropic.Anthropic(api_key=_key_from_dotenv(ROOT))
    return None


def record(task: str, model: str, item_id: str, message, sent_sha: str, batch_id: str | None) -> dict:
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        ann = json.loads(text); problems = TASKS[task]["validate"](ann)
    except json.JSONDecodeError as exc:
        ann, problems = None, [f"not JSON: {exc}"]
    if message.stop_reason != "end_turn":
        problems.append(f"stop_reason {message.stop_reason}")
    u = message.usage
    usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    row = {"item_id": item_id, "task": task, "model": model, "served_model": message.model, "prompt_version": TASKS[task]["prompt_version"],
           "annotation": ann, "valid": not problems, "problems": problems, "sent_image_sha256": sent_sha, "batch_id": batch_id, "usage": usage,
           "provenance": {"label_source": "claude", "method": "third_party_api", "interface": "anthropic_batch_api" if batch_id else "anthropic_api",
                          "training_permission": "permitted (user_confirmed by the project owner; no written reference on file)", "created": now()}}
    append_jsonl(store_path(task, model), row)
    return row


# ---------------------------------------------------------------- commands

def cmd_status(a) -> None:
    c = cfg(); limit, name = cap(c)
    rows = read_jsonl(LEDGER)
    by = defaultdict(lambda: Counter())
    for r in rows:
        for k in ("requests", "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
            by[r["model"]][k] += r.get(k, 0)
        by[r["model"]]["cost_usd"] += r["cost_usd"]
    out = {"spent_usd": spent(), "reserved_usd": reserved(), "active_cap": {"name": name, "usd": limit}, "total_cap_usd": c["budget"]["total_usd"],
           "expires": str(c["budget"].get("expires")), "buckets": {k: {"allocated": v, "used": bucket_spent(k)} for k, v in c["budget"].get("buckets", {}).items()},
           "by_model": {m: dict(v) for m, v in by.items()},
           "annotations": {f"{t}/{m}": len(annotated(t, m)) for t in TASKS for m in c["models"].values()},
           "open_batches": [{k: b[k] for k in ("batch_id", "task", "model", "n", "estimate_usd", "submitted")} for b in read_jsonl(BATCHES) if not b.get("collected")],
           "key_source": key_source()}
    print(json.dumps(out, indent=1))


def cmd_plan(a) -> None:
    c = cfg(); items = pick(a.task, a.phase, a.tier, a.n)
    est = estimate(a.task, items, c, client_or_none())
    limit, name = cap(c)
    print(json.dumps({"task": a.task, "phase": a.phase, "tier": a.tier, "estimate": est, "spent_usd": spent(), "reserved_usd": reserved(),
                      "cap": {"name": name, "usd": limit}, "would_pass_cap": spent() + reserved() + est.get("usd", 0) > limit,
                      "per_group": dict(Counter(it["group"] for it in items))}, indent=1))


def cmd_smoke(a) -> None:
    require_paid(a); c = cfg(); client = client_or_none()
    if client is None:
        raise SystemExit("no ANTHROPIC_API_KEY in the environment or in the repository's .env")
    items = pick(a.task, "pilot", a.tier, 1)[:1]
    if not items:
        raise SystemExit("nothing to annotate")
    guard(0.05, c, bucket_of(a.task, a.tier))
    p, sha = build(a.task, items[0], c)
    msg = client.messages.create(**p)
    row = record(a.task, items[0]["model"], items[0]["item_id"], msg, sha, None)
    cost = price(row["usage"], items[0]["model"], batch=False, c=c)
    append_jsonl(LEDGER, {"ts": now(), "task": a.task, "bucket": bucket_of(a.task, a.tier), "model": items[0]["model"], "mode": "sync", "requests": 1, **row["usage"], "cost_usd": round(cost, 6)})
    print(json.dumps({"valid": row["valid"], "problems": row["problems"], "usage": row["usage"], "cost_usd": cost, "annotation": row["annotation"]}, indent=1))


def cmd_submit(a) -> None:
    require_paid(a); c = cfg(); client = client_or_none()
    if client is None:
        raise SystemExit("no ANTHROPIC_API_KEY in the environment or in the repository's .env")
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    items = pick(a.task, a.phase, a.tier, a.n)
    if not items:
        raise SystemExit("nothing to annotate: every selected item already has a valid annotation or is in an open batch")
    est = estimate(a.task, items, c, client)
    guard(est["usd"], c, bucket_of(a.task, a.tier))
    for start in range(0, len(items), a.chunk):  # chunks keep each upload well under the 256 MB batch limit
        part = items[start:start + a.chunk]
        requests, ids, shas = [], {}, {}
        for it in part:
            prm, sha = build(a.task, it, c)
            cid = "i" + hashlib.sha256(it["item_id"].encode()).hexdigest()[:24]
            ids[cid], shas[cid] = it["item_id"], sha
            requests.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**prm)))
        batch = client.messages.batches.create(requests=requests)
        append_jsonl(BATCHES, {"batch_id": batch.id, "task": a.task, "phase": a.phase, "tier": a.tier, "bucket": bucket_of(a.task, a.tier), "model": part[0]["model"], "n": len(part),
                               "estimate_usd": round(est["usd_per_request"] * len(part), 4), "submitted": now(), "items": ids, "shas": shas})
        print("submitted", batch.id, len(part), "requests, estimate", round(est["usd_per_request"] * len(part), 4), "USD", flush=True)


def cmd_collect(a) -> None:
    c = cfg(); client = client_or_none()
    if client is None:
        raise SystemExit("no ANTHROPIC_API_KEY in the environment or in the repository's .env")
    while True:
        rows = read_jsonl(BATCHES); waiting = 0
        for b in rows:
            if b.get("collected"):
                continue
            info = client.messages.batches.retrieve(b["batch_id"])
            if info.processing_status != "ended":
                waiting += 1
                print(b["batch_id"], info.processing_status, dict(info.request_counts), flush=True)
                continue
            totals, outcome = Counter(), Counter()
            for res in client.messages.batches.results(b["batch_id"]):
                outcome[res.result.type] += 1
                if res.result.type != "succeeded":
                    continue
                row = record(b["task"], b["model"], b["items"][res.custom_id], res.result.message, b["shas"][res.custom_id], b["batch_id"])
                totals.update(row["usage"]); outcome["valid" if row["valid"] else "invalid"] += 1
            cost = price(totals, b["model"], batch=True, c=c)
            append_jsonl(LEDGER, {"ts": now(), "task": b["task"], "phase": b["phase"], "tier": b["tier"], "bucket": b.get("bucket"), "model": b["model"], "mode": "batch",
                                  "batch_id": b["batch_id"], "requests": b["n"], **dict(totals), "cost_usd": round(cost, 6), "outcome": dict(outcome),
                                  "estimate_usd": b["estimate_usd"]})
            b["collected"], b["outcome"], b["cost_usd"] = now(), dict(outcome), round(cost, 6)
            print("collected", b["batch_id"], dict(outcome), "cost", round(cost, 4), "estimate", b["estimate_usd"], flush=True)
        BATCHES.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        if not waiting or not a.wait:
            break
        time.sleep(60)


def cmd_review_pilot(a) -> None:
    """Pilot gate: cost per image as measured, validity, and field distributions. Writing the report lifts the pilot cap."""
    c = cfg(); cheap = c["models"]["cheap"]
    tasks = [t for t in ("bank", "filmset") if store_path(t, cheap).exists()] or [a.task]  # the pilot covers both annotation sets
    rows = [r for t in tasks for r in read_jsonl(store_path(t, cheap))]; led = [r for r in read_jsonl(LEDGER) if r["model"] == cheap]
    if len(rows) < 50:
        raise SystemExit(f"only {len(rows)} annotations: the pilot has not run")
    valid = [r for r in rows if r["valid"]]
    n_req = sum(r["requests"] for r in led); cost = sum(r["cost_usd"] for r in led)
    dist = {k: dict(Counter(r["annotation"][k] for r in valid).most_common()) for k in ("subject_primary", "tonal_treatment", "confidence", "moment")}
    report = {"created": now(), "model": cheap, "requests": n_req, "valid": len(valid), "invalid": len(rows) - len(valid),
              "cost_usd": round(cost, 4), "usd_per_image": round(cost / max(1, n_req), 6),
              "tokens": {k: sum(r.get(k, 0) for r in led) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")},
              "widely_reproduced_share": sum(r["annotation"]["widely_reproduced"] for r in valid) / max(1, len(valid)), "distributions": dist}
    total_items = sum(len(TASKS[t]["items"]()) for t in tasks)
    report["tasks"] = tasks
    report["projected_full_run_usd"] = round(report["usd_per_image"] * total_items, 2)
    ok = report["valid"] / max(1, len(rows)) >= 0.95 and report["projected_full_run_usd"] < c["budget"]["total_usd"] * 0.5
    report["gate"] = {"valid_share_min": 0.95, "projected_cost_max_usd": c["budget"]["total_usd"] * 0.5, "passed": bool(ok)}
    print(json.dumps(report, indent=1))
    if ok:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "pilot_review.json").write_text(json.dumps(report, indent=1))
        print("pilot gate passed: the total cap now applies")
    else:
        raise SystemExit("pilot gate failed: the pilot cap stays in force")


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "plan", "smoke", "submit", "collect", "review-pilot"):
        p = sub.add_parser(name)
        p.add_argument("--task", default="bank", choices=sorted(TASKS)); p.add_argument("--phase", default="pilot", choices=("pilot", "full"))
        p.add_argument("--tier", default="cheap", choices=("cheap", "check", "check_all")); p.add_argument("--n", type=int, default=None)
        p.add_argument("--chunk", type=int, default=1000); p.add_argument("--wait", action="store_true")
        p.add_argument("--no-dry-run", action="store_true"); p.add_argument("--confirm-paid", action="store_true")
    a = ap.parse_args()
    {"status": cmd_status, "plan": cmd_plan, "smoke": cmd_smoke, "submit": cmd_submit, "collect": cmd_collect, "review-pilot": cmd_review_pilot}[a.cmd](a)


if __name__ == "__main__":
    main()
