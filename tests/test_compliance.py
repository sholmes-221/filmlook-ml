"""Compliance gates: permission classification, provenance, preflight checks, request audits, job lock."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from filmlook_ml.compliance.policy import (AUTH_ENV, ConcurrentLabelingJob, LabelingLock, NonCompliantRequest,
                                           PolicyStop, assert_request_compliant, audit_prompt_text,
                                           claude_training_permission, policy_docs_status)
from filmlook_ml.compliance.preflight import run_labeling_preflight, run_training_preflight
from filmlook_ml.compliance.provenance import effective_provenance, make_provenance
from filmlook_ml.constants import ATTRIBUTES
from filmlook_ml.labeling.schema import ImageLabel, PairLabel, Provenance, StyleJudgement, TeacherOutput
from filmlook_ml.labeling.store import LABEL_FILES, PAIR_FILES
from filmlook_ml.labeling.teachers import PAIR_V2, SYSTEM_V2, ClaudeTeacher, style_system_prompt

ROOT = Path(__file__).resolve().parents[1]
ATTRS = {a: 0.5 for a in ATTRIBUTES}
UNCERTAIN = {"allow_training_use": True, "max_budget_usd": 5.0, "price_input_per_mtok": 4.0, "price_output_per_mtok": 20.0,
             "effort": "low", "max_tokens": 4000, "consented_datasets": ["synthetic_fixture"],
             "training_authorization": {"status": "confirmed_by_project_owner", "recorded": "2026-11-06", "reference": None}}
WRITTEN = {**UNCERTAIN, "training_authorization": {"status": "written_reference", "recorded": "2026-11-07",
                                                   "reference": "ANTHROPIC-AUTH-TEST-1"}}
WEIGHTS = {"human": 1.0, "claude": 0.8, "synthetic_ordering": 0.7, "heuristic": 0.3, "mock": 0.0}
IMG = {"image_id": "a", "dataset_id": "synthetic_fixture", "license": "synthetic", "sha256": "s"}


def _out(score=50.0):
    return TeacherOutput(film_score=score, attributes=dict(ATTRS), label_confidence=0.8, uncertainty="low")


def _lab(iid, src, usable=True, prov=None):
    return ImageLabel(image_id=iid, sha256="s", source=src, teacher_model="t", prompt_version="v", label=_out(),
                      usable_for_training=usable, provenance=prov)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Minimal repo root: real policy docs + configs, 3 manifest records, empty label dir."""
    monkeypatch.delenv(AUTH_ENV, raising=False)
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    shutil.copy(ROOT / "CLAUDE.md", root / "CLAUDE.md")
    shutil.copy(ROOT / "docs" / "AI_DATA_POLICY.md", root / "docs" / "AI_DATA_POLICY.md")
    shutil.copytree(ROOT / "configs", root / "configs")
    man = root / "data" / "manifests" / "manifest.jsonl"
    man.parent.mkdir(parents=True)
    rows = [{**IMG, "image_id": i, "sha256": f"s{i}", "path": f"/x/{i}.jpg"} for i in "abc"]
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    labels = root / "data" / "labels"
    labels.mkdir()
    return {"root": root, "labels": labels, "manifest": man, "rows": rows}


def _write(labels_dir, src, labs, pairs=False):
    fn = (PAIR_FILES if pairs else LABEL_FILES)[src]
    with open(labels_dir / fn, "a") as fh:
        for x in labs:
            fh.write(x.model_dump_json() + "\n")


def _training(project, cfg=UNCERTAIN, weights=WEIGHTS):
    return run_training_preflight(root=project["root"], label_dir=project["labels"], manifest_path=project["manifest"],
                                  labeling_cfg={"claude": cfg, "source_weights": weights},
                                  report_path=project["root"] / "reports" / "p.json")


# ------------------------------------------------------------------ permission classification
def test_permission_matrix(monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)
    assert claude_training_permission(UNCERTAIN)[0] == "uncertain"
    assert claude_training_permission({"allow_training_use": True})[0] == "uncertain"
    assert claude_training_permission(WRITTEN)[0] == "permitted"
    assert claude_training_permission({**UNCERTAIN, "allow_training_use": False})[0] == "not_permitted"
    assert claude_training_permission({**UNCERTAIN, "training_authorization": {"status": "denied"}})[0] == "not_permitted"
    # written_reference status without an actual reference is still uncertain
    assert claude_training_permission({**UNCERTAIN, "training_authorization": {"status": "written_reference",
                                                                               "reference": None}})[0] == "uncertain"
    monkeypatch.setenv(AUTH_ENV, "REF-9")
    status, basis = claude_training_permission(UNCERTAIN)
    assert status == "permitted" and "REF-9" in basis
    assert ClaudeTeacher.training_use_allowed(UNCERTAIN)
    monkeypatch.delenv(AUTH_ENV)
    assert not ClaudeTeacher.training_use_allowed(UNCERTAIN)


def test_make_provenance_per_source(monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)
    h = make_provenance("human", IMG, reviewer="shan")
    assert h.labeling_method == "human_review" and h.training_permission == "permitted" and "shan" in h.permission_basis
    assert make_provenance("heuristic", IMG, model_id="heuristic-v2").training_permission == "permitted"
    assert make_provenance("synthetic_ordering", IMG).labeling_method == "synthetic_edit_ordering"
    c = make_provenance("claude", IMG, model_id="claude-opus-5-5", claude_cfg=UNCERTAIN)
    assert c.provider == "anthropic" and c.training_permission == "uncertain" and c.image_license == "synthetic"
    assert make_provenance("claude", IMG, claude_cfg=WRITTEN).training_permission == "permitted"
    assert make_provenance("mock", IMG).training_permission == "not_permitted"
    with pytest.raises(ValueError):
        make_provenance("gemini", IMG)


def test_effective_provenance_uses_stricter_of_stored_and_current(monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)
    stale = Provenance(image_id="a", label_source="claude", labeling_method="third_party_api", provider="anthropic",
                       training_permission="permitted", permission_basis="written reference: OLD")
    lab = _lab("a", "claude", prov=stale)
    assert effective_provenance(lab, IMG, UNCERTAIN).training_permission == "uncertain"   # authorization withdrawn
    assert effective_provenance(lab, IMG, WRITTEN).training_permission == "permitted"
    legacy = _lab("a", "claude")                                                           # record without provenance
    p = effective_provenance(legacy, IMG, UNCERTAIN)
    assert p.labeling_method == "third_party_api" and p.training_permission == "uncertain"
    assert effective_provenance(_lab("a", "human"), IMG, UNCERTAIN).training_permission == "permitted"


# ------------------------------------------------------------------ training preflight
def test_training_preflight_passes_with_local_labels(project):
    _write(project["labels"], "heuristic", [_lab(i, "heuristic", prov=make_provenance("heuristic", r))
                                            for i, r in zip("abc", project["rows"])])
    _write(project["labels"], "human", [_lab("a", "human")])                                  # legacy, no provenance
    _write(project["labels"], "synthetic_ordering",
           [PairLabel(image_a="a", image_b="b", attribute="film_score", winner="a", source="synthetic_ordering")], pairs=True)
    rep = _training(project)
    assert rep.ok, rep.summary()
    names = {c.name: c.status for c in rep.checks}
    assert names["label_provenance"] == "pass" and names["third_party_ai_permission"] == "info"
    assert (project["root"] / "reports" / "p.json").exists()


def test_training_preflight_blocks_uncertain_claude_labels(project, monkeypatch):
    _write(project["labels"], "claude", [_lab("a", "claude", prov=make_provenance("claude", IMG, claude_cfg=UNCERTAIN))])
    rep = _training(project)
    assert not rep.ok and any(c.name == "label_provenance" and "claude x1 (uncertain)" in c.detail for c in rep.failures)
    # remedies: zero weight or the unusable flag. A later written reference does NOT revive a label whose stored
    # provenance says uncertain (stricter wins): re-run the labeler (cache hits, no cost) to re-stamp it.
    assert _training(project, weights={**WEIGHTS, "claude": 0.0}).ok
    assert not _training(project, cfg=WRITTEN).ok
    project["labels"].joinpath(LABEL_FILES["claude"]).unlink()
    _write(project["labels"], "claude", [_lab("a", "claude")])                      # legacy record, no provenance
    assert not _training(project).ok and _training(project, cfg=WRITTEN).ok
    project["labels"].joinpath(LABEL_FILES["claude"]).unlink()
    _write(project["labels"], "claude", [_lab("a", "claude", usable=False)])
    assert _training(project).ok
    # a label that CLAIMS permitted while the config is uncertain is still blocked
    project["labels"].joinpath(LABEL_FILES["claude"]).unlink()
    _write(project["labels"], "claude", [_lab("a", "claude", prov=make_provenance("claude", IMG, claude_cfg=WRITTEN))])
    assert not _training(project).ok
    monkeypatch.setenv(AUTH_ENV, "REF-1")
    assert _training(project).ok


def test_training_preflight_blocks_claude_pairs_and_warns_on_style_judgements(project):
    _write(project["labels"], "claude",
           [PairLabel(image_a="a", image_b="b", attribute="film_score", winner="a", source="claude")], pairs=True)
    rep = _training(project)
    assert not rep.ok and "pair:claude" in rep.failures[0].detail
    project["labels"].joinpath(PAIR_FILES["claude"]).unlink()
    with open(project["labels"] / "style_judgements.jsonl", "w") as fh:
        fh.write(StyleJudgement(image_id="a", sha256="s", source="claude", matches=["saul_leiter"]).model_dump_json() + "\n")
    rep = _training(project)
    assert rep.ok and {c.name: c.status for c in rep.checks}["style_judgements"] == "warn"


def test_training_preflight_fails_closed(project):
    assert _training(project, weights={**WEIGHTS, "mock": 0.5}).failures[0].name == "mock_excluded"
    # image whose per-record license is not allowlisted
    rows = project["rows"] + [{**IMG, "image_id": "d", "sha256": "sd", "license": "CC-BY-NC-4.0"}]
    project["manifest"].write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rep = _training(project)
    assert any(c.name == "dataset_registry" and "1 records rejected" in c.detail for c in rep.failures)
    # unknown dataset
    rows[-1] = {**IMG, "image_id": "d", "sha256": "sd", "dataset_id": "scraped_portfolio"}
    project["manifest"].write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert not _training(project).ok
    project["manifest"].write_text("")
    assert any(c.name == "dataset_registry" and "empty" in c.detail for c in _training(project).failures)
    # missing policy documents
    (project["root"] / "docs" / "AI_DATA_POLICY.md").unlink()
    assert any(c.name == "policy_documents" for c in _training(project).failures)
    (project["root"] / "CLAUDE.md").write_text("# no mandatory section\n")
    assert len(policy_docs_status(project["root"])) == 2


def test_training_preflight_fails_on_corrupt_label_file(project):
    (project["labels"] / LABEL_FILES["human"]).write_text('{"image_id": "a", "source": "human"}\n')
    rep = _training(project)
    assert any(c.name == "label_files_parse" for c in rep.failures)


# ------------------------------------------------------------------ labeling preflight
def _labeling(project, cfg, use, images=40, pairs=10, **kw):
    return run_labeling_preflight(root=project["root"], ccfg=cfg, intended_use=use, n_images=images, n_pairs=pairs,
                                  prompts={"system": SYSTEM_V2, "pair": PAIR_V2, "style": style_system_prompt()}, **kw)


def test_labeling_preflight_training_intent_requires_written_reference(project):
    rep = _labeling(project, UNCERTAIN, "training")
    assert not rep.ok and rep.failures[0].name == "training_permission" and "STOP" in rep.failures[0].detail
    assert _labeling(project, WRITTEN, "training").ok
    assert any(c.name == "intended_use" for c in _labeling(project, WRITTEN, "undeclared").failures)


def test_labeling_preflight_evaluation_intent_needs_owner_approval(project):
    rep = _labeling(project, UNCERTAIN, "evaluation")
    assert not rep.ok and rep.failures[0].name == "evaluation_use_approval"
    approved = {**UNCERTAIN, "evaluation_use": {"approved": True, "approved_by": "owner", "recorded": "2026-11-07"}}
    assert _labeling(project, approved, "evaluation").ok
    half = {**UNCERTAIN, "evaluation_use": {"approved": True}}            # name/date missing -> still blocked
    assert not _labeling(project, half, "evaluation").ok
    assert _labeling(project, WRITTEN, "evaluation").ok


def test_labeling_preflight_bounds_and_request_config(project):
    assert any(c.name == "batch_bounds" for c in _labeling(project, WRITTEN, "training", images=101).failures)
    assert any(c.name == "batch_bounds" for c in _labeling(project, {**WRITTEN, "max_pairs_per_run": 5}, "training").failures)
    assert any(c.name == "concurrency" for c in _labeling(project, {**WRITTEN, "max_concurrency": 4}, "training").failures)
    assert any(c.name == "budget_cap" for c in _labeling(project, {**WRITTEN, "max_budget_usd": 0}, "training").failures)
    for bad in ({"thinking": {"type": "adaptive"}}, {"effort": "max"}, {"max_tokens": 50000}):
        rep = _labeling(project, {**WRITTEN, **bad}, "training")
        assert any(c.name == "request_config" for c in rep.failures), bad
    assert any(c.name == "upload_consent" for c in
               _labeling(project, {**WRITTEN, "consented_datasets": ["ava"]}, "training").failures)
    rep = run_labeling_preflight(root=project["root"], ccfg=WRITTEN, intended_use="training", n_images=1,
                                 prompts={"bad": "Think step by step and show your reasoning before answering."})
    assert any(c.name == "request_config" and "reasoning" in c.detail for c in rep.failures)


# ------------------------------------------------------------------ request / prompt audits
def test_repository_prompts_request_no_reasoning():
    for text in (SYSTEM_V2, PAIR_V2, style_system_prompt()):
        assert audit_prompt_text(text) == []
    for text in (SYSTEM_V2, style_system_prompt()):   # system prompts (sent with every call) disclose training intent
        assert "training" in text.lower()
    assert audit_prompt_text("explain your chain of thought") and audit_prompt_text("use a scratchpad")
    assert audit_prompt_text("list ambiguity reasons") == []   # 'reasons' is not 'reasoning'


def test_assert_request_compliant():
    good = {"model": "m", "max_tokens": 4000, "output_config": {"effort": "low"}, "output_format": object(),
            "system": [{"type": "text", "text": SYSTEM_V2}], "messages": [{"role": "user", "content": [{"type": "text", "text": "Assess."}]}]}
    assert_request_compliant(good)
    for bad in ({"thinking": {"type": "adaptive"}}, {"output_config": {"effort": "high"}}, {"max_tokens": 9000},
                {"output_format": None}, {"system": "think step by step"}):
        with pytest.raises(NonCompliantRequest):
            assert_request_compliant({**good, **bad})


class _Resp:
    def __init__(self, parsed, stop="end_turn"):
        class U:
            input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens = 1500, 700, 0, 0
        self.usage, self.stop_reason, self.parsed_output = U(), stop, parsed


class _Client:
    def __init__(self, stop="end_turn"):
        self.kwargs, self.stop = [], stop
        self.messages = self

    def parse(self, **kw):
        self.kwargs.append(kw)
        from filmlook_ml.labeling.schema import ClaudeAssessment
        if self.stop == "refusal":
            return _Resp(None, "refusal")
        return _Resp(ClaudeAssessment.model_validate({
            "monochrome": False, "film_score": 55, "attributes": dict(ATTRS),
            "style": {"temperature": "warm", "saturation": "muted", "sharpness": "natural", "key": "balanced",
                      "grain": "fine_grain", "mood": []},
            "uncertainty": "low", "ambiguity_reasons": [], "strengths": ["x"], "weaknesses": [], "label_confidence": 0.8}))


CFG = {**WRITTEN, "model": "claude-opus-5-5", "max_image_side": 256, "texture_crop": 64}


def test_claude_teacher_requests_are_structured_low_effort_no_thinking(tmp_path, fixture_dir):
    c = _Client()
    t = ClaudeTeacher(CFG, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=c)
    assert t.label(fixture_dir / "scene_0.png", consent_ok=True) is not None
    kw = c.kwargs[0]
    assert "thinking" not in kw and kw["output_config"] == {"effort": "low"} and kw["max_tokens"] <= 4000
    assert kw["output_format"] is not None
    t2 = ClaudeTeacher({**CFG, "effort": "max"}, tmp_path / "d.jsonl", dry_run=False, confirm=True, client=c)
    with pytest.raises(NonCompliantRequest):
        t2.label(fixture_dir / "scene_1.png", consent_ok=True)
    assert len(c.kwargs) == 1                          # the non-compliant request was never sent


def test_refusal_stops_immediately_without_retry(tmp_path, fixture_dir, monkeypatch):
    import filmlook_ml.labeling.teachers as T
    monkeypatch.setattr(T.time, "sleep", lambda *_: None)
    c = _Client(stop="refusal")
    t = ClaudeTeacher(CFG, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=c)
    with pytest.raises(PolicyStop):
        t.label(fixture_dir / "scene_0.png", consent_ok=True)
    assert len(c.kwargs) == 1


def test_claude_labels_are_unusable_without_written_reference(tmp_path, fixture_dir, monkeypatch):
    from filmlook_ml.labeling.pipeline import claude_labels
    monkeypatch.delenv(AUTH_ENV, raising=False)
    sel = [{"image_id": "a", "path": str(fixture_dir / "scene_0.png"), "sha256": "sa", "dataset_id": "synthetic_fixture",
            "license": "synthetic"}]
    cfg = {**UNCERTAIN, "model": "claude-opus-5-5", "max_image_side": 256, "texture_crop": 64}
    t = ClaudeTeacher(cfg, tmp_path / "c.jsonl", dry_run=False, confirm=True, client=_Client())
    labs, _, rep = claude_labels(sel, [], t, {"synthetic_fixture"}, {})
    assert len(labs) == 1 and labs[0].usable_for_training is False
    assert labs[0].provenance.training_permission == "uncertain" and labs[0].provenance.provider == "anthropic"
    assert rep["training_permission"] == "uncertain" and rep["usable_for_training"] is False
    t = ClaudeTeacher(CFG, tmp_path / "d.jsonl", dry_run=False, confirm=True, client=_Client())
    labs, _, rep = claude_labels(sel, [], t, {"synthetic_fixture"}, {})
    assert labs[0].usable_for_training is True and labs[0].provenance.training_permission == "permitted"


# ------------------------------------------------------------------ job lock
def test_labeling_lock_is_exclusive_and_clears_stale(tmp_path):
    p = tmp_path / "x.lock"
    with LabelingLock(p):
        assert p.read_text() == str(os.getpid())
        with pytest.raises(ConcurrentLabelingJob):
            with LabelingLock(p):
                pass
    assert not p.exists()
    p.write_text("999999999")          # stale pid
    with LabelingLock(p):
        assert p.exists()


# ------------------------------------------------------------------ CLI
def test_preflight_cli_exit_codes(project):
    env = {**os.environ, "FILMLOOK_ROOT": str(project["root"]), "PYTHONPATH": str(ROOT / "src")}
    env.pop(AUTH_ENV, None)
    # pin the "uncertain" state (owner statement without reference) regardless of the live config's status
    cfgp = project["root"] / "configs" / "labeling.yaml"
    cfgp.write_text(cfgp.read_text().replace("status: user_confirmed", "status: confirmed_by_project_owner"))
    _write(project["labels"], "heuristic", [_lab("a", "heuristic")])
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "preflight_check.py")], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr
    _write(project["labels"], "claude", [_lab("b", "claude")])
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "preflight_check.py"), "--json"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 1 and json.loads(r.stdout)["ok"] is False
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "preflight_check.py"), "--mode", "labeling",
                        "--intended-use", "training"], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "training_permission" in r.stdout
