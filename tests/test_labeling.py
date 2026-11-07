import json

import numpy as np
import pytest
from pydantic import ValidationError

from filmlook_ml.constants import ATTRIBUTES
from filmlook_ml.labeling.schema import (ClaudeAssessment, ClaudePairAssessment, ImageLabel, PairLabel, TeacherOutput,
                                         to_teacher_output)
from filmlook_ml.labeling.store import build_targets, pair_arrays
from filmlook_ml.labeling.teachers import (AUTH_ENV, BudgetExceeded, ClaudeTeacher, HeuristicTeacher, LabelCache,
                                           LabelingNotAuthorized, MockTeacher, api_key_status, check_label,
                                           training_authorization)

ATTRS = {a: 0.5 for a in ATTRIBUTES}


def _out(**kw):
    d = dict(film_score=50, attributes=dict(ATTRS), label_confidence=0.8, uncertainty="low")
    d.update(kw)
    return TeacherOutput(**d)


def _assessment(mono=False, score=55.0):
    return ClaudeAssessment.model_validate({
        "monochrome": mono, "film_score": score,
        "attributes": {**ATTRS, "color_response": None if mono else 0.5},
        "style": {"temperature": "not_applicable" if mono else "warm", "saturation": "not_applicable" if mono else "muted",
                  "sharpness": "natural", "key": "balanced", "grain": "fine_grain", "mood": []},
        "uncertainty": "low", "ambiguity_reasons": [], "strengths": ["smooth highlight roll-off"],
        "weaknesses": [], "label_confidence": 0.8})


@pytest.mark.parametrize("bad", [
    dict(film_score=120),
    dict(attributes={**ATTRS, "highlight_rolloff": 1.5}),
    dict(attributes={k: v for k, v in ATTRS.items() if k != "microcontrast"}),
    dict(attributes={**ATTRS, "made_up": 0.1}),
    dict(attributes={**ATTRS, "microcontrast": None}),   # only colour may be null
])
def test_schema_rejects(bad):
    with pytest.raises(ValidationError):
        _out(**bad)


def test_schema_mono_color_null_ok():
    assert _out(attributes={**ATTRS, "color_response": None}).attributes["color_response"] is None
    o = to_teacher_output(_assessment(mono=True))
    assert o.attributes["color_response"] is None and o.monochrome


def test_claude_api_schemas_are_strict():
    from anthropic import transform_schema
    for m in (ClaudeAssessment, ClaudePairAssessment):
        s = json.dumps(transform_schema(m))
        assert '"additionalProperties": true' not in s


def test_heuristic_and_mock_teachers_validate(fixture_dir):
    p = fixture_dir / "scene_0.png"
    for t in (HeuristicTeacher(), MockTeacher()):
        out = t.label(p)
        assert 0 <= out.film_score <= 100 and set(out.attributes) == set(ATTRIBUTES)
    assert MockTeacher().label(p) == MockTeacher().label(p)


class FakeUsage:
    input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens = 1500, 700, 0, 0


class FakeResp:
    def __init__(self, parsed):
        self.usage, self.stop_reason, self.parsed_output = FakeUsage(), "end_turn", parsed


class FakeMessages:
    def __init__(self):
        self.calls, self.kinds = 0, []

    def parse(self, **kw):
        self.calls += 1
        content = kw["messages"][0]["content"]
        assert sum(c["type"] == "image" for c in content) in (2, 4)   # full view + crop (per image)
        assert kw["system"][0]["text"].startswith("You label photographs")
        if kw["output_format"] is ClaudePairAssessment:
            self.kinds.append("pair")
            return FakeResp(ClaudePairAssessment(film_score="A", highlight_rolloff="A", color_response="tie",
                                                 microcontrast="B", grain_naturalness="A", tonal_depth="tie",
                                                 film_coherence="A", confidence="high", reason="x"))
        self.kinds.append("image")
        return FakeResp(_assessment())


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


CFG = {"model": "claude-opus-5-5", "max_budget_usd": 1.0, "price_input_per_mtok": 4.0, "price_output_per_mtok": 20.0,
       "max_tokens": 2000, "max_image_side": 512, "texture_crop": 128}


def test_claude_dry_run_makes_no_call(tmp_path, fixture_dir):
    c = FakeClient()
    t = ClaudeTeacher(CFG, tmp_path / "c.jsonl", dry_run=True, client=c)
    assert t.label(fixture_dir / "scene_0.png", consent_ok=True) is None
    assert c.messages.calls == 0


def test_claude_requires_consent_and_confirmation(tmp_path, fixture_dir):
    t = ClaudeTeacher(CFG, tmp_path / "c.jsonl", dry_run=False, confirm=False, client=FakeClient())
    with pytest.raises(LabelingNotAuthorized):
        t.label(fixture_dir / "scene_0.png", consent_ok=False)
    with pytest.raises(LabelingNotAuthorized):
        t.label(fixture_dir / "scene_0.png", consent_ok=True)


def test_claude_cache_resume_and_usage_log(tmp_path, fixture_dir):
    c = FakeClient()
    log = tmp_path / "usage.jsonl"
    t = ClaudeTeacher(CFG, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=c, usage_log=log)
    a = t.label(fixture_dir / "scene_0.png", consent_ok=True)
    b = t.label(fixture_dir / "scene_0.png", consent_ok=True)
    assert a == b and c.messages.calls == 1
    row = json.loads(log.read_text().splitlines()[0])
    assert row["input_tokens"] == 1500 and row["model"] == "claude-opus-5-5" and row["prompt_version"] == "v2"
    assert "ts" in row and row["usd"] > 0
    c2 = FakeClient()   # new run resumes from cache: no paid call
    ClaudeTeacher(CFG, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=c2).label(
        fixture_dir / "scene_0.png", consent_ok=True)
    assert c2.messages.calls == 0
    ClaudeTeacher(CFG, tmp_path / "c.jsonl", prompt_version="v3", dry_run=False, confirm=True, client=c2).label(
        fixture_dir / "scene_0.png", consent_ok=True)
    assert c2.messages.calls == 1


def test_claude_budget_cap_is_cumulative_across_runs(tmp_path, fixture_dir):
    log = tmp_path / "usage.jsonl"
    cfg = {**CFG, "max_budget_usd": 0.05}
    c = FakeClient()
    t = ClaudeTeacher(cfg, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=c, usage_log=log)
    t.label(fixture_dir / "scene_0.png", consent_ok=True)      # ~0.02 spent
    t2 = ClaudeTeacher(cfg, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=c, usage_log=log)
    assert t2.cost.spent_before == pytest.approx(t.cost.spent)
    with pytest.raises(BudgetExceeded):
        for i in range(1, 6):
            t2.label(fixture_dir / f"scene_{i}.png", consent_ok=True)
    assert t2.cost.spent <= 0.05


def test_claude_pair_compare(tmp_path, fixture_dir):
    c = FakeClient()
    t = ClaudeTeacher(CFG, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=c)
    r = t.compare(fixture_dir / "scene_0.png", fixture_dir / "scene_1.png", "a", "b", consent_ok=True)
    assert r.film_score == "A" and c.messages.kinds == ["pair"]
    t.compare(fixture_dir / "scene_0.png", fixture_dir / "scene_1.png", "a", "b", consent_ok=True)
    assert c.messages.calls == 1   # cached


def test_training_authorization_gate(monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)
    assert training_authorization({"allow_training_use": True}) is None              # no record -> blocked
    rec = {"status": "confirmed_by_project_owner", "recorded": "2026-11-06", "reference": None}
    auth = training_authorization({"allow_training_use": True, "training_authorization": rec})
    assert auth and "confirmed_by_project_owner" in auth and "none supplied" in auth
    assert training_authorization({"allow_training_use": False, "training_authorization": rec}) is None
    monkeypatch.setenv(AUTH_ENV, "REF-1")
    assert "REF-1" in training_authorization({"allow_training_use": True})
    assert ClaudeTeacher.training_use_allowed({"allow_training_use": True})


def test_api_key_status_never_leaks(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert api_key_status(tmp_path) == "missing"
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n")
    s = api_key_status(tmp_path)
    assert s == ".env" and "sk-" not in s


def test_check_label_flags():
    f = {"sat_p95": 0.01, "noise_std": 0.002, "warmth": 0.0}
    out = to_teacher_output(_assessment(mono=False, score=95))   # colour on a mono image, score >> attrs
    flags = check_label(out, f)
    assert "mono_mismatch" in flags and "score_attr_mismatch" in flags
    assert check_label(to_teacher_output(_assessment()), {"sat_p95": 0.5, "noise_std": 0.01, "warmth": 0.05}) == []


def _lab(iid, src, score, usable=True, sup=None, status="unreviewed", flags=()):
    return ImageLabel(image_id=iid, sha256="x", source=src, teacher_model="t", prompt_version="v",
                      label=_out(film_score=score), usable_for_training=usable, supervised_attributes=sup,
                      review_status=status, flags=list(flags))


def test_label_priority_and_flags():
    labels = {"heuristic": {k: _lab(k, "heuristic", 30) for k in "abcd"},
              "human": {"a": _lab("a", "human", 90, status="corrected")},
              "claude": {"b": _lab("b", "claude", 70), "c": _lab("c", "claude", 99, usable=False),
                         "d": _lab("d", "claude", 80, flags=["mono_mismatch"])}}
    t = build_targets(list("abcd"), labels, {"human": 1.0, "heuristic": 0.3, "claude": 0.8})
    assert t.score[0] == pytest.approx(0.9) and t.source[0] == "human"
    assert t.score[1] == pytest.approx(0.7) and t.score_w[1] == 0.8                  # authorised claude label
    assert t.source[2] == "heuristic" and t.source[3] == "heuristic"                  # unusable / flagged -> skipped


def test_null_color_is_unsupervised():
    lab = ImageLabel(image_id="a", sha256="x", source="claude", teacher_model="t", prompt_version="v",
                     label=_out(attributes={**ATTRS, "color_response": None}))
    t = build_targets(["a"], {"claude": {"a": lab}}, {"claude": 0.8})
    j = ATTRIBUTES.index("color_response")
    assert t.attrs_w[0, j] == 0 and t.attrs_w[0].sum() == pytest.approx(0.8 * 5)


def test_medium_targets_originals_only():
    man = [{"image_id": "a", "edit_type": "original", "medium": "film_scan"},
           {"image_id": "b", "edit_type": "film_emulation", "medium": "film_scan"},
           {"image_id": "c", "edit_type": "original", "medium": "digital_filtered"},
           {"image_id": "d", "edit_type": "original"}]
    t = build_targets(list("abcd"), {}, {}, man)
    assert list(t.medium) == [1, 0, 0, 0] and list(t.medium_w) == [1, 0, 1, 0]


def test_pair_arrays():
    pairs = [PairLabel(image_a="a", image_b="b", attribute="film_score", winner="a", source="human"),
             PairLabel(image_a="a", image_b="b", attribute="microcontrast", winner="b", source="synthetic_ordering"),
             PairLabel(image_a="a", image_b="z", attribute="film_score", winner="a", source="human")]
    ia, ib, at, sign, w = pair_arrays(pairs, {"a": 0, "b": 1}, {"human": 1.0, "synthetic_ordering": 0.7})
    assert list(at) == [-1, ATTRIBUTES.index("microcontrast")] and list(sign) == [1.0, -1.0]


def test_label_cache_roundtrip(tmp_path):
    c = LabelCache(tmp_path / "x.jsonl")
    k = LabelCache.key("abc", "v1", "m")
    c.put(k, {"a": 1})
    assert LabelCache(tmp_path / "x.jsonl").get(k) == {"a": 1}


# ------------------------------------------------------------------ CommonCatalog acquisition
def test_commoncatalog_classify():
    from filmlook_ml.datasets.commoncatalog import classify
    base = {"licenseurl": "http://creativecommons.org/licenses/by/2.0/", "status": "success",
            "blip2_caption": "a woman walking down a street at night", "usertags": "", "capturedevice": ""}
    film = classify({**base, "exif": json.dumps({"Image Make": "NORITSU KOKI", "Image Model": "QSS-32_33"})})
    assert film["license_ok"] and film["medium"] == "film_scan" and {"street", "night", "portrait"} <= set(film["subjects"])
    phone = classify({**base, "exif": json.dumps({"Image Make": "Apple", "Image Model": "iPhone 4",
                                                  "EXIF DateTimeOriginal": "x"}), "usertags": "instagram,nyc"})
    assert phone["medium"] == "smartphone_filtered"
    assert not classify({**base, "licenseurl": "http://creativecommons.org/licenses/by-sa/2.0/", "exif": "{}"})["license_ok"]
    assert classify({**base, "blip2_caption": "a screenshot of a website", "exif": "{}"})["exclude"]


def test_parquet_page_reader_matches_pyarrow(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from filmlook_ml.datasets.commoncatalog import fetch_rows

    rng = np.random.default_rng(0)
    blobs = [b"\xff\xd8\xff" + rng.bytes(int(n)) for n in rng.integers(1000, 5000, 20)]
    blobs[3] = None
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"jpg": pa.array(blobs, pa.binary()), "id": list(range(20))}), p,
                   data_page_size=2048, row_group_size=10, compression="snappy", write_page_index=True,
                   use_dictionary=False)

    class FS:
        def open(self, path, **kw):
            return open(path, "rb")

    got = fetch_rows(FS(), str(p), 1, {0, 4, 9})
    assert got == {k: blobs[10 + k] for k in (0, 4, 9)}
    assert 3 not in fetch_rows(FS(), str(p), 0, {3})
