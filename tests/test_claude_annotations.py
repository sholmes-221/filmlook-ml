"""The Claude annotation pipeline without the network: vocabulary and vectors, pricing from usage fields, the pilot and
total caps, the dry-run refusal, no item sent twice, and what happens to a malformed or refused answer."""
import json
import types

import numpy as np
import pytest

from research.claude_annotations import lut_tasks, pipeline as P, schema as S


def good_annotation() -> dict:
    a = {k: v[0] for k, v in S.SINGLE.items()}
    a.update({k: v[:2] for k, v in S.MULTI.items()})
    a.update({k: "3" for k in S.ORDINAL})
    a["widely_reproduced"] = False
    return a


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUT", tmp_path / "ann"); monkeypatch.setattr(P, "LEDGER", tmp_path / "ann" / "usage.jsonl")
    monkeypatch.setattr(P, "BATCHES", tmp_path / "ann" / "batches.jsonl"); monkeypatch.setattr(P, "REPORTS", tmp_path / "reports")
    return tmp_path


def message(text: str, stop: str = "end_turn", **usage):
    u = types.SimpleNamespace(input_tokens=usage.get("input", 1000), cache_creation_input_tokens=usage.get("write", 0),
                              cache_read_input_tokens=usage.get("read", 700), output_tokens=usage.get("output", 300))
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], stop_reason=stop, usage=u, model="claude-haiku-5-5")


def test_schema_is_closed_and_vectors_have_a_fixed_length():
    schema = S.json_schema()
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    assert not any("name" in k or "title" in k or "photographer" in k for k in schema["properties"])  # nothing can hold an attribution
    a = good_annotation()
    assert S.validate(a) == []
    v = S.vectorise(a)
    assert v.shape == (len(S.feature_names()),) and v.dtype == np.float32 and 0 <= v.min() and v.max() <= 1
    assert len(S.vectorise(a, include_artefacts=True)) > len(v)  # scan-condition fields are left out of the style vector by default
    bad = dict(a, framing="telephoto", mood=["happy"], humour="7")
    assert len(S.validate(bad)) == 3
    assert "do not name" in S.SYSTEM_PROMPT.lower() and "train" in S.SYSTEM_PROMPT.lower()  # no attribution guessing; intent disclosed


def test_cost_is_priced_from_usage_with_cache_and_batch_multipliers():
    c = P.cfg()
    usage = {"input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000, "output_tokens": 1_000_000}
    p = c["prices_per_mtok"]["claude-haiku-5-5"]; m = c["multipliers"]
    full = p["input"] * (1 + m["cache_write_5m"] + m["cache_read"]) + p["output"]
    assert P.price(usage, "claude-haiku-5-5", batch=False, c=c) == pytest.approx(full)
    assert P.price(usage, "claude-haiku-5-5", batch=True, c=c) == pytest.approx(full * m["batch"])
    assert c["budget"]["total_usd"] <= 100 and c["budget"]["pilot_usd"] <= 5  # the owner's caps


def test_the_pilot_cap_applies_until_the_pilot_review_exists_and_the_total_cap_after(sandbox):
    c = P.cfg()
    assert P.cap(c) == (c["budget"]["pilot_usd"], "pilot")
    P.append_jsonl(P.LEDGER, {"cost_usd": 4.0, "model": "claude-haiku-5-5", "requests": 1})
    P.append_jsonl(P.BATCHES, {"batch_id": "b1", "estimate_usd": 0.9, "model": "claude-haiku-5-5", "task": "bank", "items": {}})
    P.guard(0.05, c)  # 4.0 spent + 0.9 reserved + 0.05 = 4.95: allowed
    with pytest.raises(SystemExit, match="BUDGET STOP"):
        P.guard(0.2, c)  # an open batch's estimate counts until it is collected
    P.REPORTS.mkdir(parents=True); (P.REPORTS / "pilot_review.json").write_text("{}")
    assert P.cap(c) == (c["budget"]["total_usd"], "total")
    P.guard(50, c)
    with pytest.raises(SystemExit, match="BUDGET STOP"):
        P.guard(96, c)


def test_nothing_is_sent_without_both_paid_flags(sandbox):
    for flags in ({"no_dry_run": False, "confirm_paid": False}, {"no_dry_run": True, "confirm_paid": False}, {"no_dry_run": False, "confirm_paid": True}):
        with pytest.raises(SystemExit, match="dry run"):
            P.require_paid(types.SimpleNamespace(**flags))
    P.require_paid(types.SimpleNamespace(no_dry_run=True, confirm_paid=True))


def test_answers_are_validated_and_an_annotated_item_is_never_picked_again(sandbox, monkeypatch):
    items = [{"item_id": f"x:{i}", "group": "g" + str(i % 2), "path": "unused"} for i in range(6)]
    monkeypatch.setitem(P.TASKS["bank"], "items", lambda: list(items))
    ok = P.record("bank", "claude-haiku-5-5", "x:0", message(json.dumps(good_annotation())), "sha", "batch1")
    assert ok["valid"] and ok["usage"]["cache_read_input_tokens"] == 700 and ok["provenance"]["label_source"] == "claude"
    broken = P.record("bank", "claude-haiku-5-5", "x:1", message("not json"), "sha", "batch1")
    refused = P.record("bank", "claude-haiku-5-5", "x:2", message(json.dumps(good_annotation()), stop="refusal"), "sha", "batch1")
    off_vocab = P.record("bank", "claude-haiku-5-5", "x:3", message(json.dumps(dict(good_annotation(), key="medium"))), "sha", "batch1")
    assert not broken["valid"] and not refused["valid"] and not off_vocab["valid"]
    assert set(P.annotated("bank", "claude-haiku-5-5")) == {"x:0"}
    P.append_jsonl(P.BATCHES, {"batch_id": "b2", "estimate_usd": 0.01, "model": "claude-haiku-5-5", "task": "bank", "items": {"c": "x:4"}})
    picked = {it["item_id"] for it in P.pick("bank", "full", "cheap", None)}
    assert picked == {"x:1", "x:2", "x:3", "x:5"}  # not the valid one, not the one waiting in an open batch
    hard = {it["item_id"] for it in P.pick("bank", "full", "check", 1)}
    assert {"x:1", "x:2", "x:3"} <= hard and all(it["model"] == "claude-sonnet-5-5" for it in P.pick("bank", "full", "check", 1))


def test_requests_put_the_shared_instructions_in_one_cached_system_block():
    c = P.cfg()
    p = P.params("bank", "claude-haiku-5-5", "", c)
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"} and p["system"][0]["text"] == S.SYSTEM_PROMPT
    assert p["output_config"]["format"]["type"] == "json_schema" and p["output_config"]["effort"] == "low"
    assert "temperature" not in p and "thinking" not in p and "tool_choice" not in p  # not accepted / not needed on these models
    assert json.dumps(p["system"]) == json.dumps(P.params("bank", "claude-haiku-5-5", "", c)["system"])  # byte-identical prefix


def test_the_description_check_hides_the_answer_and_scores_against_the_applied_preset():
    desc = lut_tasks.descriptions()
    assert len(desc) == 13
    items = P.lutdesc_items(n_scenes=2)
    assert len(items) == 26
    for it in items:
        preset = it["item_id"].split(":")[2]
        assert it["meta"]["options"][int(it["meta"]["answer"]) - 1] == preset and len(set(it["meta"]["options"])) == 4
        assert preset not in it["user"] and "Kodak" not in it["user"] and len(it["paths"]) == 2  # only descriptions are shown, never a name
    answers = [it["meta"]["answer"] for it in P.lutdesc_items(n_scenes=6)]
    assert len(set(answers)) == 4  # the position of the right answer varies
    assert lut_tasks.validate({"choice": "2", "visible_change": "clear", "confidence": "high"}) == []
    assert lut_tasks.validate({"choice": "5", "visible_change": "clear", "confidence": "high"}) != []
