"""Exploratory categorization workflow: taxonomy/schema validation, discovery, resume, atomic never-overwrite
storage, stop conditions, export gate, and the related preflight checks. No network: the API client is faked."""
import json
import shutil
from pathlib import Path

import anthropic
import httpx2 as httpx
import numpy as np
import pytest
from pydantic import ValidationError

from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.export import ExportBlocked, export_training
from filmlook_ml.annotation.schema import Annotation, CategorizationResult
from filmlook_ml.annotation import heads as H
from filmlook_ml.annotation import qa
from filmlook_ml.annotation.workflow import (AnnotationStore, Categorizer, atomic_write_text, discover, film_stock_tags,
                                             run, system_prompt, validate_store)
from filmlook_ml.compliance.policy import AUTH_ENV, assert_request_compliant, audit_prompt_text, policy_docs_status
from filmlook_ml.compliance.preflight import run_labeling_preflight, run_training_preflight
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.labeling.teachers import ClaudeTeacher

ROOT = Path(__file__).resolve().parents[1]
UNCERTAIN = {"model": "claude-opus-5-5", "allow_training_use": True, "max_budget_usd": 5.0, "price_input_per_mtok": 4.0,
             "price_output_per_mtok": 20.0, "effort": "low", "max_tokens": 4000, "max_image_side": 256,
             "texture_crop": 64, "consented_datasets": ["pd12m_pilot", "synthetic_fixture"],
             "training_authorization": {"status": "confirmed_by_project_owner", "recorded": "2026-11-06", "reference": None}}
WRITTEN = {**UNCERTAIN, "training_authorization": {"status": "written_reference", "recorded": "2026-11-07",
                                                   "reference": "ANTHROPIC-AUTH-TEST-1"}}
APPROVED = {**UNCERTAIN, "exploratory_use": {"approved": True, "approved_by": "owner", "recorded": "2026-11-07"}}
USERCONF = {**UNCERTAIN, "training_authorization": {"status": "user_confirmed", "recorded": "2026-11-07", "reference": None}}
RESULT = {"photographer_style": {"primary": "Saul Leiter", "alternatives": ["Rinko Kawauchi"], "confidence": 0.7,
                                 "evidence": ["muted palette", "layered view through glass"]},
          "film_aesthetic": {"primary": "Kodak Portra 400", "alternatives": [], "confidence": 0.6,
                             "evidence": ["soft contrast"]},
          "attributes": {"color": {"temperature": "warm", "saturation": "low", "dominant_colors": ["red", "beige"],
                                   "color_cast": "none"},
                         "lighting": {"source": "natural", "quality": "soft", "key": "balanced", "direction": "diffuse"},
                         "tone": {"contrast": "low", "highlight_rolloff": "soft", "shadow_depth": "lifted"},
                         "texture": {"visible_grain": "subtle", "grain_appearance": "fine", "sharpness": "moderate"},
                         "genre": ["street"], "overall_aesthetic": ["muted"], "image_flags": []}}


@pytest.fixture(autouse=True)
def _no_auth_env(monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)


@pytest.fixture
def records(fixture_dir):
    rows = [{"image_id": f"img{i}_original", "path": str(p), "sha256": f"sha{i}", "edit_type": "original",
             "dataset_id": "pd12m_pilot", "license": "CC0-1.0", "source_url": "u", "attribution": "", "split": "train"}
            for i, p in enumerate(sorted(fixture_dir.glob("scene_*.png")))]
    return rows


class U:
    input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens = 1200, 150, 0, 0


class Resp:
    def __init__(self, parsed, stop="end_turn"):
        self.usage, self.stop_reason, self.parsed_output = U(), stop, parsed


class FakeMessages:
    def __init__(self, behaviour=None):
        self.calls, self.kw, self.behaviour = 0, [], behaviour or (lambda n: Resp(CategorizationResult(**RESULT)))

    def parse(self, **kw):
        self.calls += 1
        self.kw.append(kw)
        assert kw["output_format"] is CategorizationResult
        assert sum(c["type"] == "image" for c in kw["messages"][0]["content"]) == 2
        return self.behaviour(self.calls)


class FakeClient:
    def __init__(self, behaviour=None):
        self.messages = FakeMessages(behaviour)


def _teacher(tmp_path, client, cfg=UNCERTAIN, dry=False):
    return ClaudeTeacher(cfg, tmp_path / "cache.jsonl", "v2", dry_run=dry, confirm=True, client=client,
                         usage_log=tmp_path / "usage.jsonl")


def _run(tmp_path, records, client, cfg=UNCERTAIN, dry=False, n=10):
    store = AnnotationStore(tmp_path / "annotations")
    res = run(eligible=records, store=store, categorizer=Categorizer(_teacher(tmp_path, client, cfg, dry)), ccfg=cfg,
              max_images=n, min_interval_s=0, sleep=lambda s: None)
    return store, res


def _api_error(cls, code):
    return cls("err", response=httpx.Response(code, request=httpx.Request("POST", "https://api.anthropic.com")), body=None)


# ------------------------------------------------------------------ taxonomy + schema
def test_taxonomy_v2_is_independent_of_mini_tool_profiles():
    from filmlook_ml.style.profiles import PROFILES
    assert len(T.PHOTOGRAPHERS) == 10 and len(T.FILM_STOCKS) == 17
    assert T.PHOTOGRAPHER_LABELS[-2:] == ["none_of_the_above", "unknown"] and T.FILM_LABELS[-1] == "unknown"
    assert len(PROFILES) == 9                                                  # Mini Tool profiles untouched


def _bad(path, value):
    d = json.loads(json.dumps(RESULT))
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return d


@pytest.mark.parametrize("path,value", [
    (("photographer_style", "primary"), "Ansel Adams"),                        # off-taxonomy
    (("photographer_style", "alternatives"), ["Fan Ho", "Martin Parr", "Alex Webb"]),   # > 2 alternatives
    (("photographer_style", "alternatives"), ["Saul Leiter"]),                 # repeats primary
    (("photographer_style", "alternatives"), ["unknown"]),                     # special label as alternative
    (("photographer_style", "confidence"), 1.3),
    (("photographer_style", "evidence"), []),
    (("photographer_style", "evidence"), ["x" * 81]),
    (("film_aesthetic", "primary"), "Kodak Portra 100"),
    (("attributes", "texture", "visible_grain"), "lots"),
    (("attributes", "texture", "grain_appearance"), "fine_and_coarse"),
    (("attributes", "color", "temperature"), "not_applicable"),                # only for monochrome
    (("attributes", "genre"), []),
    (("attributes", "color", "dominant_colors"), ["red", "blue", "green", "pink"]),
    (("explanation",), "because"),                                             # no free text fields
])
def test_schema_rejects_invalid(path, value):
    CategorizationResult(**RESULT)
    with pytest.raises(ValidationError):
        CategorizationResult(**_bad(path, value))


def test_monochrome_rules():
    mono = _bad(("attributes", "color"), {"temperature": "not_applicable", "saturation": "monochrome",
                                          "dominant_colors": ["gray"], "color_cast": "not_applicable"})
    CategorizationResult(**mono)
    with pytest.raises(ValidationError):
        CategorizationResult(**_bad(("attributes", "color"), {"temperature": "warm", "saturation": "monochrome",
                                                              "dominant_colors": ["gray"], "color_cast": "none"}))


def test_prompt_requests_no_reasoning_and_states_purpose():
    p = system_prompt()
    assert audit_prompt_text(p) == []
    assert "authorship" in p and "DINOv2" in p and "none_of_the_above" in p


def test_film_stock_tags_are_uploader_asserted_not_verified():
    m = film_stock_tags("film,portra+400,35mm,trix")
    assert [x.value for x in m] == ["Kodak Portra 400", "Kodak Tri-X 400"] and all(not x.verified for x in m)
    assert film_stock_tags("") == []


# ------------------------------------------------------------------ discovery
def test_discover_filters_and_counts(records, tmp_path):
    extra = [{**records[0], "image_id": "e", "edit_type": "film_emulation"},
             {**records[0], "image_id": "s", "dataset_id": "synthetic_fixture", "license": "synthetic"},
             {**records[0], "image_id": "l", "license": "CC-BY-SA-4.0"},
             {**records[0], "image_id": "m", "path": str(tmp_path / "missing.jpg")},
             {**records[0], "image_id": "c", "dataset_id": "commoncatalog_ccby", "license": "CC-BY-2.0"}]
    el, skipped = discover(records + extra, Registry.load(), {"pd12m_pilot"})
    assert [r["image_id"] for r in el] == sorted(r["image_id"] for r in records)
    assert skipped == {"derived_edit": 1, "not_a_photograph": 1, "license_not_approved": 1, "file_missing": 1,
                       "no_upload_consent": 1}


# ------------------------------------------------------------------ run / resume / storage
def test_dry_run_sends_nothing_and_writes_no_annotations(tmp_path, records):
    c = FakeClient()
    store, res = _run(tmp_path, records, c, dry=True)
    assert c.messages.calls == 0 and res["dry_run_not_sent"] == len(records) and store.load_all() == []


def test_paid_run_writes_validated_exploratory_annotations_and_resumes(tmp_path, records):
    c = FakeClient()
    store, res = _run(tmp_path, records, c, n=4)
    assert res["categorized"] == 4 and res["remaining"] == len(records) - 4 and "per-run limit" in res["stop_reason"]
    for kw in c.messages.kw:
        assert_request_compliant(kw) is None and "thinking" not in kw
    anns = store.load_all()
    assert all(a.annotation_interface == "anthropic_api" and a.annotation_type == "inferred_aesthetic_labels"
               for a in anns)
    assert all(a.provenance.training_permission == "uncertain" and not a.training_eligible for a in anns)
    assert anns[0].photographer_style.primary == "Saul Leiter" and anns[0].image_source["license"] == "CC0-1.0"
    store2, res2 = _run(tmp_path, records, c, n=10)                     # resume: only the rest is sent
    assert res2["categorized"] == len(records) - 4 and res2["already_done"] == 4
    assert c.messages.calls == len(records)
    store3, res3 = _run(tmp_path, records, c, n=10)
    assert res3["categorized"] == 0 and c.messages.calls == len(records)
    assert validate_store(store3) == {"valid": len(records), "invalid": []}
    assert qa.distributions(store3.load_all())["film_primary"] == {"Kodak Portra 400": len(records)}
    prog = [json.loads(x) for x in (store.dir / "progress.jsonl").read_text().splitlines()]
    assert [p["event"] for p in prog if "event" in p].count("run_start") == 3


def test_cached_result_is_not_paid_twice_after_crash(tmp_path, records):
    c = FakeClient()
    store, _ = _run(tmp_path, records[:1], c, n=1)
    store.path(records[0]["image_id"]).unlink()                          # simulate crash before the write
    _run(tmp_path, records[:1], c, n=1)
    assert c.messages.calls == 1 and store.has(records[0]["image_id"])


def test_annotations_are_never_overwritten(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_text(p, "1")
    with pytest.raises(FileExistsError):
        atomic_write_text(p, "2")
    assert p.read_text() == "1" and list(tmp_path.glob(".*tmp")) == []


@pytest.mark.parametrize("err", [lambda: _api_error(anthropic.AuthenticationError, 401),
                                 lambda: _api_error(anthropic.PermissionDeniedError, 403)])
def test_auth_and_permission_errors_stop_the_run(tmp_path, records, err):
    def boom(n):
        raise err()
    c = FakeClient(boom)
    store, res = _run(tmp_path, records, c)
    assert c.messages.calls == 1 and res["stop_reason"].startswith("stopped") and store.load_all() == []
    assert json.loads((store.dir / "errors.jsonl").read_text().splitlines()[0])["fatal"] is True


def test_refusal_stops_the_run(tmp_path, records):
    c = FakeClient(lambda n: Resp(None, stop="refusal"))
    _, res = _run(tmp_path, records, c)
    assert c.messages.calls == 1 and "PolicyStop" in res["stop_reason"]


def test_budget_cap_stops_the_run(tmp_path, records):
    c = FakeClient()
    _, res = _run(tmp_path, records, c, cfg={**UNCERTAIN, "max_budget_usd": 0.02})
    assert "BudgetExceeded" in res["stop_reason"] and c.messages.calls < len(records)


def test_consecutive_failures_stop(tmp_path, records):
    class Broken:
        t = _teacher(tmp_path, FakeClient())

        def categorize(self, *a):
            raise RuntimeError("Claude labeling failed after 5 attempts")
    store = AnnotationStore(tmp_path / "annotations")
    res = run(eligible=records, store=store, categorizer=Broken(), ccfg=UNCERTAIN, max_images=10, min_interval_s=0)
    assert res["failed"] == 3 and "consecutive" in res["stop_reason"]


# ------------------------------------------------------------------ export gate + preflight
def test_export_blocked_without_written_reference(tmp_path, records):
    store, _ = _run(tmp_path, records[:2], FakeClient())
    with pytest.raises(ExportBlocked):
        export_training(store, UNCERTAIN)
    assert not (store.root / "training_approved").exists()


def test_export_never_revives_uncertain_annotations(tmp_path, records):
    store, _ = _run(tmp_path, records[:2], FakeClient())                 # stamped uncertain
    res = export_training(store, WRITTEN)
    assert res["exported"] == 0 and res["refused_uncertain_provenance"] == 2


def test_export_with_written_reference(tmp_path, records):
    store, _ = _run(tmp_path, records[:2], FakeClient(), cfg=WRITTEN)
    res = export_training(store, WRITTEN)
    assert res["exported"] == 2 and "ANTHROPIC-AUTH-TEST-1" in res["basis"]
    assert export_training(store, WRITTEN)["already_exported"] == 2


def _project(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    shutil.copy(ROOT / "CLAUDE.md", root / "CLAUDE.md")
    shutil.copy(ROOT / "docs" / "AI_DATA_POLICY.md", root / "docs" / "AI_DATA_POLICY.md")
    (root / "data" / "labels").mkdir(parents=True)
    man = root / "data" / "manifests" / "manifest.jsonl"
    man.parent.mkdir(parents=True)
    man.write_text("")
    return root, man


def test_labeling_preflight_exploratory_needs_owner_approval(tmp_path):
    root, _ = _project(tmp_path)
    kw = dict(root=root, intended_use="exploratory", n_images=10, prompts={"p": system_prompt()})
    rep = run_labeling_preflight(ccfg=UNCERTAIN, **kw)
    assert not rep.ok and [f.name for f in rep.failures] == ["exploratory_use_approval"]
    assert run_labeling_preflight(ccfg=APPROVED, **kw).ok
    assert not run_labeling_preflight(ccfg={**UNCERTAIN, "exploratory_use": {"approved": True}}, **kw).ok


def test_training_preflight_rejects_training_approved_annotations_without_permission(tmp_path):
    root, man = _project(tmp_path)
    cfg = {"claude": UNCERTAIN, "source_weights": {"human": 1, "claude": 0.8, "synthetic_ordering": 0.7,
                                                   "heuristic": 0.3, "mock": 0.0}}
    kw = dict(root=root, label_dir=root / "data" / "labels", manifest_path=man, labeling_cfg=cfg)
    expl = root / "data" / "annotations" / "claude_exploratory" / "annotations"
    expl.mkdir(parents=True)
    (expl / "x.json").write_text("{}")
    assert "categorization_annotations" not in [f.name for f in run_training_preflight(**kw).failures]
    appr = root / "data" / "annotations" / "training_approved"
    appr.mkdir()
    (appr / "x.json").write_text("{}")
    assert "categorization_annotations" in [f.name for f in run_training_preflight(**kw).failures]


def test_claude_md_has_both_mandatory_sections():
    text = (ROOT / "CLAUDE.md").read_text()
    assert policy_docs_status(ROOT) == []
    assert "Photography Classification Compliance" in text


# ------------------------------------------------------------------ user-confirmed authorization, QA, heads
def test_user_confirmed_status_is_permitted_but_recorded_as_unverified(tmp_path, records):
    from filmlook_ml.compliance.policy import claude_training_permission
    status, basis = claude_training_permission(USERCONF)
    assert status == "permitted" and "not independently verified" in basis and "no written reference" in basis
    store, _ = _run(tmp_path, records[:2], FakeClient(), cfg=USERCONF)
    anns = store.load_all()
    assert all(a.provenance.training_permission == "permitted" for a in anns)
    assert all(a.training_eligible == (not a.qa.needs_review) for a in anns)
    assert export_training(store, USERCONF)["exported"] == sum(1 for _ in anns)


def test_training_eligible_cannot_be_set_with_pending_review(tmp_path, records):
    store, _ = _run(tmp_path, records[:1], FakeClient(), cfg=USERCONF)
    a = store.load_all()[0]
    d = json.loads(a.model_dump_json())
    d["qa"]["needs_review"], d["training_eligible"] = True, True
    with pytest.raises(ValidationError):
        Annotation.model_validate(d)


def test_qa_flags_hallucinated_attributes():
    r = CategorizationResult(**_bad(("film_aesthetic", "primary"), "Kodak Tri-X 400"))
    measured = {"tags": ["warm", "high_contrast", "no_visible_grain", "low_key"], "is_mono": False,
                "noise_std": 0.001, "y_std": 0.35, "y_mean": 0.2, "sat_mean": 0.2, "warmth": 0.1}
    r2 = CategorizationResult(**_bad(("attributes", "texture"), {"visible_grain": "strong", "grain_appearance": "coarse",
                                                                 "sharpness": "soft"}))
    assert "bw_stock_on_colour_image" in qa.check(r, measured)
    flags = qa.check(r2, measured)
    assert {"grain_claim_unsupported", "contrast_disagrees_with_measurement"} <= set(flags)
    assert qa.needs_review(flags) and not qa.needs_review(["style_none_of_the_above"])


def test_near_duplicates_by_phash():
    recs = [{"image_id": "a", "phash": "ffff0000ffff0000", "split": "train"},
            {"image_id": "b", "phash": "ffff0000ffff0001", "split": "test"},
            {"image_id": "c", "phash": "0000ffff0000ffff", "split": "train"}]
    assert qa.near_duplicates(recs) == [("a", "b", 1, True)]


def _synthetic_heads_data(n_per=12, classes=("Saul Leiter", "Fan Ho", "none_of_the_above"), seed=0):
    """In-memory synthetic data for the training code path only (never written to data/)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), 768)) * 3
    y, split = [], []
    for c in range(len(classes)):
        for i in range(n_per):
            y.append(c)
            split.append("train" if i < n_per - 4 else ("val" if i < n_per - 2 else "test"))
    y = np.array(y)
    emb = centers[y] + rng.normal(size=(len(y), 768))
    return H.HeadData(head="photographer", classes=list(classes),
                      split={s: np.array([i for i, x in enumerate(split) if x == s]) for s in ("train", "val", "test")},
                      y=y, alts=[[] for _ in y], conf=np.full(len(y), 0.7), emb=emb,
                      feats=rng.normal(size=(len(y), 28)), ids=[str(i) for i in range(len(y))])


def test_heads_pipeline_runs_and_reports_heldout_metrics_only():
    import numpy as np  # noqa: F811
    d = _synthetic_heads_data()
    res = H.run_head(d, H.METHODS())
    assert set(res) == {"colour_texture", "knn_dinov2", "linear_dinov2"}
    for m in res.values():
        assert set(m) >= {"val", "test", "model_size_bytes"} and "train" not in m
        assert {"macro_f1_present_classes", "top3", "per_class", "confusion", "none_of_the_above"} <= set(m["test"])
    assert res["linear_dinov2"]["test"]["top1"] == 1.0          # separable synthetic clusters
    assert res["colour_texture"]["test"]["top1"] < 1.0          # random features carry no signal


def test_readiness_refuses_sparse_labels(tmp_path, records):
    store, _ = _run(tmp_path, records, FakeClient(), cfg=USERCONF)
    r = H.readiness(store.load_all(), "photographer", {x["image_id"]: "train" for x in records})
    assert r["ready"] is False and r["trainable_classes"] == []
