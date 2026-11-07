"""Preflight validation for training / evaluation and for third-party-AI labeling runs.

Training must not proceed when dataset provenance or permissions are missing (policy section 7). The
checks here are deliberately cheap (metadata only) so they can run at the top of every entry point.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filmlook_ml.compliance.policy import (
    ALLOWED_EFFORT,
    MAX_TOKENS_CAP,
    THIRD_PARTY_SOURCES,
    audit_prompt_text,
    claude_evaluation_use_approved,
    claude_exploratory_use_approved,
    claude_training_permission,
    policy_docs_status,
)
from filmlook_ml.compliance.provenance import effective_provenance
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.labeling.schema import ImageLabel, PairLabel, StyleJudgement
from filmlook_ml.labeling.store import LABEL_FILES, PAIR_FILES

STYLE_JUDGEMENTS = "style_judgements.jsonl"


@dataclass
class Check:
    name: str
    status: str            # pass | fail | warn | info
    detail: str


@dataclass
class PreflightReport:
    mode: str
    checks: list[Check] = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def ok(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    def summary(self) -> str:
        head = f"preflight[{self.mode}] {'PASS' if self.ok else 'FAIL'}"
        body = "\n".join(f"  [{c.status:4}] {c.name}: {c.detail}" for c in self.checks)
        tail = "" if self.ok else "\nSee docs/AI_DATA_POLICY.md (pre-training compliance checklist)."
        return f"{head}\n{body}{tail}"

    def to_dict(self) -> dict:
        return {"mode": self.mode, "ok": self.ok, "created": self.created, "checks": [asdict(c) for c in self.checks]}

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))


# ----------------------------------------------------------------------------- training / evaluation
def run_training_preflight(*, root: Path, label_dir: Path, manifest_path: Path, labeling_cfg: dict,
                           registry: Registry | None = None, report_path: Path | None = None) -> PreflightReport:
    rep = PreflightReport("training")
    registry = registry or Registry.load()
    ccfg = labeling_cfg.get("claude") or {}
    weights = labeling_cfg.get("source_weights") or {}

    # 1. policy documents
    probs = policy_docs_status(root)
    rep.add("policy_documents", "fail" if probs else "pass", "; ".join(probs) or "CLAUDE.md and docs/AI_DATA_POLICY.md present")

    # 2. image provenance: every manifest record must pass the registry (dataset approved AND record license)
    manifest = read_jsonl(manifest_path)
    by_id = {r["image_id"]: r for r in manifest}
    if not manifest:
        rep.add("dataset_registry", "fail", f"manifest {manifest_path} is empty or missing; run scripts/prepare_dataset.py")
    else:
        bad = Counter()
        for r in manifest:
            ok, why = registry.check_record(r.get("dataset_id", ""), r.get("license"))
            if not ok:
                bad[why] += 1
        missing = [k for k in ("image_id", "dataset_id", "license", "sha256") if any(k not in r for r in manifest)]
        if bad or missing:
            rep.add("dataset_registry", "fail", f"{sum(bad.values())} records rejected {dict(bad)}; missing fields {missing}")
        else:
            ds = Counter(r["dataset_id"] for r in manifest)
            rep.add("dataset_registry", "pass", f"{len(manifest)} records, all registry-approved: {dict(ds)}")

    # 3. label files parse
    labels: dict[str, list[ImageLabel]] = {}
    pairs: dict[str, list[PairLabel]] = {}
    parse_errors = []
    for src, fn in LABEL_FILES.items():
        try:
            labels[src] = [ImageLabel.model_validate(r) for r in read_jsonl(Path(label_dir) / fn)]
        except Exception as e:  # noqa: BLE001
            parse_errors.append(f"{fn}: {type(e).__name__}: {str(e)[:120]}")
    for src, fn in PAIR_FILES.items():
        try:
            pairs[src] = [PairLabel.model_validate(r) for r in read_jsonl(Path(label_dir) / fn)]
        except Exception as e:  # noqa: BLE001
            parse_errors.append(f"{fn}: {type(e).__name__}: {str(e)[:120]}")
    rep.add("label_files_parse", "fail" if parse_errors else "pass",
            "; ".join(parse_errors) or f"labels {dict((s, len(v)) for s, v in labels.items() if v)}, "
                                       f"pairs {dict((s, len(v)) for s, v in pairs.items() if v)}")

    # 4. mock labels can never carry weight
    if float(weights.get("mock", 0.0) or 0.0) > 0:
        rep.add("mock_excluded", "fail", f"source_weights.mock must be 0, got {weights.get('mock')}")
    else:
        rep.add("mock_excluded", "pass", "mock labels carry zero weight")

    # 5. third-party-AI permission status (informational; enforced per label below)
    status, basis = claude_training_permission(ccfg)
    rep.add("third_party_ai_permission", "pass" if status == "permitted" else "info", f"claude: {status} - {basis}")

    # 6. per-label provenance for everything that could enter training
    def could_train(lab: ImageLabel) -> bool:
        return lab.usable_for_training and lab.review_status != "rejected" and float(weights.get(lab.source, 0.0) or 0) > 0

    blocked, unknown_img, counts = Counter(), Counter(), Counter()
    for src, labs in labels.items():
        for lab in labs:
            if not could_train(lab):
                continue
            img = by_id.get(lab.image_id)
            if img is None:
                unknown_img[src] += 1
                continue
            prov = effective_provenance(lab, img, ccfg)
            counts[src] += 1
            if prov.training_permission != "permitted":
                blocked[(src, prov.training_permission)] += 1
    for src, ps in pairs.items():
        for p in ps:
            if float(weights.get(p.source, 0.0) or 0) <= 0:
                continue
            a, b = by_id.get(p.image_a), by_id.get(p.image_b)
            if a is None or b is None:
                unknown_img[f"pair:{p.source}"] += 1
                continue
            prov = effective_provenance(p, a, ccfg)
            counts[f"pair:{p.source}"] += 1
            if prov.training_permission != "permitted":
                blocked[(f"pair:{p.source}", prov.training_permission)] += 1
    if blocked:
        rep.add("label_provenance", "fail",
                "labels that would enter training lack permission: "
                + ", ".join(f"{s} x{n} ({st})" for (s, st), n in sorted(blocked.items()))
                + ". Fix: obtain a written authorization reference, or set their source weight to 0 / "
                  "usable_for_training=false. Do not combine restricted labels with approved data.")
    else:
        rep.add("label_provenance", "pass", f"training-eligible labels all permitted: {dict(counts)}")
    if unknown_img:
        rep.add("labels_without_image", "warn", f"labels whose image is not in the manifest (ignored): {dict(unknown_img)}")

    # 7. style judgements from third-party AI are evaluation-only unless permitted
    try:
        js = [StyleJudgement.model_validate(r) for r in read_jsonl(Path(label_dir) / STYLE_JUDGEMENTS)]
        n_tp = sum(j.source in THIRD_PARTY_SOURCES for j in js)
        if n_tp and status != "permitted":
            rep.add("style_judgements", "warn", f"{n_tp} claude style judgements present; permission {status}: "
                                                f"they must not be used as training targets (evaluation only)")
        else:
            rep.add("style_judgements", "pass", f"{len(js)} judgements ({n_tp} third-party)")
    except Exception as e:  # noqa: BLE001
        rep.add("style_judgements", "fail", f"{STYLE_JUDGEMENTS}: {type(e).__name__}: {str(e)[:120]}")

    # 8. categorization annotations: exploratory ones are never read by training; training_approved ones must
    #    still be permitted under the CURRENT configuration (a withdrawn authorization blocks them)
    ann_root = Path(label_dir).parent / "annotations"
    n_expl = len(list((ann_root / "claude_exploratory" / "annotations").glob("*.json")))
    n_appr = len(list((ann_root / "training_approved").glob("*.json")))
    if n_appr and status != "permitted":
        rep.add("categorization_annotations", "fail",
                f"{n_appr} files in {ann_root / 'training_approved'} but Claude training permission is {status}; "
                f"remove them or record a written authorization")
    else:
        rep.add("categorization_annotations", "pass",
                f"{n_expl} exploratory (never used for training), {n_appr} training-approved")

    if report_path:
        rep.write(report_path)
    return rep


def require_training_preflight(labeling_cfg: dict | None = None) -> PreflightReport:
    """Gate used by every training/evaluation entry point (via training.data.load_train_data)."""
    from filmlook_ml import paths as P
    cfg = labeling_cfg or P.load_yaml("labeling.yaml")
    rep = run_training_preflight(root=P.ROOT, label_dir=P.LABELS, manifest_path=P.MANIFESTS / "manifest.jsonl",
                                 labeling_cfg=cfg, report_path=P.REPORTS / "preflight_training.json")
    print(rep.summary())
    if not rep.ok:
        raise SystemExit("training preflight failed; training must not proceed (docs/AI_DATA_POLICY.md)")
    return rep


# ----------------------------------------------------------------------------- third-party-AI labeling
def run_labeling_preflight(*, root: Path, ccfg: dict, intended_use: str, n_images: int, n_pairs: int = 0,
                           dry_run: bool = True, prompts: dict[str, str] | None = None,
                           registry: Registry | None = None, report_path: Path | None = None) -> PreflightReport:
    rep = PreflightReport("labeling")
    registry = registry or Registry.load()

    probs = policy_docs_status(root)
    rep.add("policy_documents", "fail" if probs else "pass", "; ".join(probs) or "present")

    if intended_use not in ("training", "evaluation", "exploratory"):
        rep.add("intended_use", "fail",
                f"declare --intended-use training|evaluation|exploratory (got {intended_use!r})")
    else:
        rep.add("intended_use", "pass", intended_use)

    status, basis = claude_training_permission(ccfg)
    if intended_use == "training":
        rep.add("training_permission", "pass" if status == "permitted" else "fail",
                f"{status}: {basis}" + ("" if status == "permitted" else
                                        " -> STOP: outputs may not be used as training targets; request clarification"))
    elif intended_use == "evaluation":
        if status == "permitted":
            rep.add("training_permission", "pass", f"{status}: {basis}")
        else:
            ok, why = claude_evaluation_use_approved(ccfg)
            rep.add("evaluation_use_approval", "pass" if ok else "fail",
                    f"training permission is {status}; {why}. Labels will be stored usable_for_training=false.")
    elif intended_use == "exploratory":
        # exploratory categorization (data/annotations/claude_exploratory): never a training target, but a paid
        # run still needs a recorded owner decision, exactly like evaluation-only use
        ok, why = claude_exploratory_use_approved(ccfg)
        rep.add("exploratory_use_approval", "pass" if ok else "fail",
                f"training permission is {status}; {why}. Annotations stay in data/annotations/claude_exploratory "
                f"(training_eligible only if permitted and not flagged) and leave it only through the export gate.")

    # bounded batches, single job
    max_i, max_p = int(ccfg.get("max_images_per_run", 100)), int(ccfg.get("max_pairs_per_run", 50))
    if n_images > max_i or n_pairs > max_p:
        rep.add("batch_bounds", "fail", f"requested {n_images} images / {n_pairs} pairs exceeds "
                                        f"max_images_per_run={max_i} / max_pairs_per_run={max_p}")
    else:
        rep.add("batch_bounds", "pass", f"{n_images} images / {n_pairs} pairs within {max_i}/{max_p}")
    conc = int(ccfg.get("max_concurrency", 1))
    rep.add("concurrency", "pass" if conc == 1 else "fail", f"max_concurrency={conc} (must be 1)")

    cap = float(ccfg.get("max_budget_usd", 0) or 0)
    if cap <= 0 or not ccfg.get("price_input_per_mtok") or not ccfg.get("price_output_per_mtok"):
        rep.add("budget_cap", "fail", "max_budget_usd and both prices must be set")
    else:
        rep.add("budget_cap", "pass", f"cumulative cap US${cap:.2f}")

    # request configuration (section 3)
    rp = []
    if "thinking" in ccfg:
        rp.append("claude.thinking must not be configured")
    if ccfg.get("effort", "low") not in ALLOWED_EFFORT:
        rp.append(f"effort must be one of {sorted(ALLOWED_EFFORT)}")
    if int(ccfg.get("max_tokens", 0) or 0) > MAX_TOKENS_CAP:
        rp.append(f"max_tokens exceeds {MAX_TOKENS_CAP}")
    for name, text in (prompts or {}).items():
        hits = audit_prompt_text(text)
        if hits:
            rp.append(f"prompt {name} requests reasoning {hits}")
    rep.add("request_config", "fail" if rp else "pass", "; ".join(rp) or "structured output, low effort, no reasoning requested")

    bad = [d for d in ccfg.get("consented_datasets", []) if not registry.dataset_allowed(d)]
    rep.add("upload_consent", "fail" if bad else "pass",
            f"consented_datasets not registry-approved: {bad}" if bad else "only registry-approved datasets may be uploaded")

    rep.add("dry_run", "info", "dry run: no paid calls" if dry_run else "PAID run requested (--no-dry-run --confirm-paid)")
    if report_path:
        rep.write(report_path)
    return rep
