# FilmLook-ML AI Data Policy

Mandatory for every dataset labeling, pseudo-label generation, model training, fine-tuning or evaluation
workflow in this repository. The checklist in `CLAUDE.md` points here; `scripts/preflight_check.py` and
`filmlook_ml.compliance` enforce the mechanical parts. This is an engineering record, not legal advice;
items marked ⚠ need the project owner (and, where noted, counsel) to decide.

Last review of provider terms: **2026-11-07**.

## 1. Objective and scope

FilmLook-ML trains a lightweight photography-style classifier: image → DINOv2-small (frozen) → embedding →
small classification head → photographer-style resemblance (Saul Leiter, William Eggleston, Alex Webb, Daido
Moriyama, Shomei Tomatsu and the other profiles in `src/filmlook_ml/style/profiles.py`), plus the film-look
scorer already shipped in v1. The goal is to classify photographic aesthetics. It is **not** to reproduce
Claude's reasoning, to build a general-purpose language model, or to build anything that competes with
Anthropic's services. Any Claude API use is offline, label-only, budget-capped, and never part of inference.

## 2. Provider terms reviewed (Anthropic)

Checked on 2026-11-07. Re-check the live pages before each labeling or training run; quote the wording you
rely on in `configs/labeling.yaml` or here, with the date.

| Source | Date | Wording relied on |
|---|---|---|
| Commercial Terms of Service, section D.4 Use Restrictions (https://www.anthropic.com/legal/commercial-terms) | effective 2025-06-17 | "Customer may not and must not attempt to (a) access the Services to build a competing product or service, including to train competing AI models or resell the Services except as expressly approved by Anthropic". Section B: "Customer ... owns its Outputs." |
| Usage Policy, "Do Not Abuse Our Platform" (https://www.anthropic.com/legal/aup) | effective 2026-11-12 (the prior version carried the same bullet) | Prohibited: "Use outputs to train an AI model (e.g., 'model scraping' or 'model distillation') without prior authorization from Anthropic". |
| Help Center, "Can I use my Outputs to train an AI model?" (https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model) | 2026-03-16 | "We prohibit customers from using our services to train or develop AI models without our written permission." Permitted: "models that don't compete with Anthropic's own models ... specialized classifiers" such as content categorization. Prohibited list includes "using Outputs as training targets for models". |

**Interpretation applied by this repository**

- A photography-style classifier does not compete with Anthropic's models, which the Help Center lists as the
  kind of classifier that *can* be trained on Outputs. However, pseudo-labels are literally "Outputs used as
  training targets", which the same article lists as prohibited, and the Usage Policy requires **prior
  authorization** for any training on outputs. The two documents are not reconcilable from the project side.
- Therefore training on Claude outputs is **permitted only with a written authorization reference** from
  Anthropic that names this use. Anything less is **uncertain**, and uncertain means stop (section 8, step 5).
- Returning "only classification labels" does not change this. Labels are outputs.
- Training intent is disclosed in every prompt (the system prompts say the labels train a small model). Never
  remove that disclosure and never describe the task to the provider as something else.

**Status history.** The paragraph below describes 2026-11-06 to 2026-11-07. Since 2026-11-07 (later) the status
is **PERMITTED (user-confirmed, unverified)**; see "Owner decision 2026-11-07" at the end of this section.

**Former status: UNCERTAIN.** `configs/labeling.yaml` records that the project owner stated on 2026-11-06
that they confirmed with Anthropic that outputs may be used; no written reference was supplied and none has
been invented. Until `FILMLOOK_ANTHROPIC_TRAINING_AUTHORIZATION_REF` (or
`training_authorization.status: written_reference` + `reference`) points at a written authorization:

- the preflight refuses Claude labels as training targets;
- `scripts/generate_pseudo_labels.py --teacher claude` and `scripts/style_labels.py` refuse paid runs declared
  `--intended-use training`;
- evaluation-only runs (outputs used to score a model, never to train one) additionally require the owner to
  set `claude.evaluation_use: {approved: true, approved_by: <name>, recorded: <date>}`. ⚠ Owner decision:
  the Usage Policy bullet concerns *training*, but the Help Center wording ("train or develop") is broader.

**Owner decision 2026-11-07.** The project owner stated that they confirmed with Anthropic that the intended use
is permitted: content categorization, structured information extraction and photography-aesthetics
classification for the DINOv2-small model. They instructed that this confirmation be used. It is recorded as
`training_authorization.status: user_confirmed` with `reference: null`. `claude_training_permission` maps it to
`permitted`, with the basis "user-confirmed by the project owner; no written reference on file; not independently
verified". This repository did not see or verify any document. Keep the written confirmation with the project
records and add its reference when available. Every safeguard below still applies: structured output only, no
reasoning requests, budget cap, single job, stop on policy errors, and no subscription automation.

## 3. Annotation sources: permitted, uncertain, requires an alternative

| Source | `label_source` / method | Status | Basis and conditions |
|---|---|---|---|
| Human review (`scripts/review_labels.py`) | `human` / `human_review` | **Permitted** | Reviewer named in provenance. Highest priority and weight. Test-split labels form the isolated benchmark. |
| Licensed dataset metadata (EXIF scanner / camera, uploader tags) → capture medium | manifest `medium` | **Permitted** | Image must pass `configs/datasets.yaml` (dataset `training_allowed` **and** per-record license on the allowlist). CommonCatalog CC-BY, PD12M CC0/PDM, synthetic. |
| Synthetic edit orderings (known edits applied in-repo) | `synthetic_ordering` / `synthetic_edit_ordering` | **Permitted** | Deterministic, local, no provider. |
| Heuristic formulas over measured statistics | `heuristic` / `heuristic_formula` | **Permitted** | Local, license-clean; weak supervision only (weight 0.3), not validated against humans. |
| Locally hosted open-source vision models (DINOv2-small, Apache-2.0, as a frozen feature extractor) | `local_model` | **Permitted** as features | Record the model id and license in provenance if a local model is ever used as a *teacher*; check its license permits derivative models. |
| Claude API outputs as **training targets** (film labels, pair judgements, style judgements) | `claude` / `third_party_api` | **Uncertain → blocked** | Needs a written authorization reference (section 2). Stored labels carry `training_permission: uncertain` and `usable_for_training: false`. |
| Claude API **exploratory categorization** (photographer-inspired style, film aesthetic, visual attributes; `scripts/annotate_categories.py`) | `claude` / `third_party_api` | **Uncertain → blocked until the owner approves exploratory use** | `claude.exploratory_use` (approved, approved_by, recorded). Stored only in `data/annotations/claude_exploratory/` with `usable_for_training: false`; reaches `training_approved/` only through `scripts/export_annotations.py`, which needs a written authorization reference. See section 4a. |
| Claude API outputs as an **evaluation reference** (teacher agreement, never trained on) | `claude` / `third_party_api` | **Uncertain → blocked until the owner approves evaluation-only use** | `claude.evaluation_use.approved` with name and date; outputs stay out of training; `scripts/style_evaluate.py --train-source claude` is refused. |
| Mock teacher | `mock` | **Not permitted** (by design) | Weight 0; `usable_for_training: false`. Tests only. |
| Other third-party AI APIs (any provider) | — | **Not permitted** until added here | Add the provider's terms to section 2, a permission function to `compliance/policy.py`, and tests. |
| Photographs by the named photographers, photographer portfolios, AVA, Unsplash Full, scraped sites | — | **Requires an alternative** | No license grant (see `DATASET_SOURCES.md`). Style profiles remain textual; style supervision must come from human judgements on licensed photos. |

Alternative data path for the style classifier while Claude is blocked: human style judgements collected in
the review app (Style mode) on registry-approved originals, grouped by uploader for CV, with the profile
engine as the zero-training baseline. About 300 judgements is the planned pilot (`REPORT.md`).

## 4. Claude API output restrictions (when a run is permitted)

- Do not request chain-of-thought, hidden reasoning traces, internal deliberation, or imitation of Claude's
  reasoning or decision process. Prompts are audited for such phrases (`compliance.policy.audit_prompt_text`)
  and every request is checked (`assert_request_compliant`) before it is sent.
- Do not set the `thinking` parameter. Note: on `claude-opus-5-5` adaptive thinking cannot be disabled
  server-side; the repository neither requests it, nor asks for its display, and sets
  `output_config.effort: low` so that token spend stays minimal. Any change of effort above `medium`, or any
  extended-thinking request, needs a separate written justification in this file.
- Request concise, structured results only: `messages.parse` with the strict pydantic schemas in
  `labeling/schema.py` (`ClaudeAssessment`, `ClaudePairAssessment`, `ClaudeStyleAssessment`). Free-text fields
  are limited to short evidence phrases. `max_tokens` is capped at 4000.
- The minimal structured format for photographer-style analysis is:

  ```json
  {
    "photographer_style": "Saul Leiter",
    "confidence": 0.82,
    "visual_attributes": {"color_palette": "muted", "lighting": "soft", "composition": "layered", "contrast": "low"}
  }
  ```

  `ClaudeStyleAssessment` is the repository's equivalent (style id, 0-1 resemblance, one evidence phrase).
- Confidence values are subjective self-reports, never calibrated probabilities. They route labels to human
  review (`check_label` flags); they are not trusted as probabilities anywhere.
- Prompt caching is used for the fixed system prompt; results are cached by content hash so no image is ever
  sent twice for the same prompt version and model.

## 4a. Exploratory categorization annotations

Workflow: `scripts/annotate_categories.py` (package `filmlook_ml.annotation`). Scope: content categorization and
information extraction only. Each request returns exactly one photographer-inspired style, one film aesthetic and
eight enumerated visual attributes plus up to three palette colours (`annotation/taxonomy.py`, `taxonomy-v1`).
"Unknown" is allowed for both categories. There is no free text and no explanation. Names describe visual
resemblance, never authorship or the stock actually used.

- **Eligible images:** originals only. The dataset must pass the registry per record and be listed in
  `claude.consented_datasets`. Synthetic test scenes and edited derivatives are skipped.
- **Storage:** one JSON file per image, written atomically and never overwritten, with provenance (image dataset,
  license, attribution, model, prompt and taxonomy version, `training_permission` at annotation time) and request
  metadata (tokens, US$). No raw response text is stored. Progress and error logs sit alongside.
- **Unattended safety:** resume and skip, single job lock, minimum request interval, rate limits honoured. The run
  stops (exit 3) on authentication, permission, bad-request, not-found, refusal, non-compliant-request or budget
  errors, or after three consecutive failures. It is never restarted automatically.
- **Paid runs** need the owner's `claude.exploratory_use` record, an API key, `--no-dry-run --confirm-paid`, and
  the shared cumulative cap. At about US$0.024 per image, the US$5 cap covers about 200 of the 968 eligible images.
- **Export to training:** `scripts/export_annotations.py` refuses unless the status is `permitted` (written
  reference). Annotations stamped `uncertain` are never exported; re-annotate after an authorization is recorded.
  The training preflight fails if `training_approved/` holds files while the permission is not `permitted`.
- **Since 2026-11-07 (later):** schema `categorization-2.0`, taxonomy `taxonomy-v2` (10 photographers, 17 stocks,
  plus `none_of_the_above` and `unknown`), with up to 2 alternatives, a self-reported confidence and 1-3 short
  evidence phrases. Uploader film-stock tags and EXIF capture medium are stored as metadata, apart from inferred
  labels, and are never sent to the model. `training_eligible` = permitted provenance AND no QA flag that needs
  review (`annotation/qa.py`). A 20-image pilot was annotated interactively (`scripts/categorize_pilot.py`) at the
  owner's request. Larger runs use the API path.
- **Film-aesthetic categories** have no licensed reference images. They are exploratory metadata and are not
  shipped in any runtime model (`DATASET_SOURCES.md`).

## 5. Training data provenance

Every label, pair judgement and style judgement carries a `Provenance` record (`labeling/schema.py`):

| Field | Meaning |
|---|---|
| `image_id`, `image_dataset_id`, `image_license` | which image, from which registry dataset, under which per-record license |
| `label_source` | `human`, `synthetic_ordering`, `heuristic`, `claude`, `mock` |
| `labeling_method` | `human_review`, `synthetic_edit_ordering`, `heuristic_formula`, `third_party_api`, `local_model`, `mock` |
| `annotated_at` | UTC timestamp |
| `provider`, `model_id` | e.g. `anthropic` / `claude-opus-5-5`; `None` for human and local methods |
| `training_permission` | `permitted`, `uncertain`, `not_permitted` at annotation time |
| `permission_basis` | human-readable basis (license id, reviewer, written reference). Never invented. |

Rules:

- Records written before 2026-11-07 have no `provenance`; the preflight derives one from `source` (human,
  heuristic and synthetic → permitted; claude → the current config status; mock → not permitted).
- Third-party-API records are **re-evaluated against the current configuration** at preflight time and the
  stricter status wins, so a label recorded as permitted cannot outlive a withdrawn authorization. The reverse
  also holds: a label stamped `uncertain` is not revived by a later authorization. Re-run the labeling script
  after recording the reference; it resumes from the local cache (no paid calls) and re-stamps provenance.
- Restricted labels are never silently combined with approved data: the training preflight fails if any label
  that could carry weight is not `permitted`. The remedy is a written reference, or weight 0 /
  `usable_for_training: false`, never a bypass.
- Human, licensed-dataset, authorized-provider and local-model labels stay distinguishable by `label_source`
  and `labeling_method` in every report (`label_sources` in checkpoint metadata, `teacher_agreement` in
  evaluation).

## 6. DINOv2-small training

Architecture: image → DINOv2-small (frozen, `facebook/dinov2-small`, Apache-2.0) → CLS + mean-patch embedding
→ small head. Stage A trains only the head on cached embeddings; Stage B (partial unfreeze) stays off unless
held-out evaluation justifies it.

Supervision priority: 1. human-labeled photographs; 2. appropriately licensed dataset metadata; 3. authorized
open-source vision models; 4. weak supervision from documented visual attributes (heuristics, synthetic edit
orderings); 5. a small human validation set, always on the test split and never used for selection. Claude
labels enter only when section 2 is satisfied, and then at weight 0.8 behind human labels.

## 7. API cost and account safety

| Safeguard | Implementation |
|---|---|
| No large-scale Claude labeling without explicit approval | dry run by default; paid runs need `--no-dry-run --confirm-paid`, a declared `--intended-use`, a passing labeling preflight, and a request from the owner in the session |
| No unbounded batches | `claude.max_images_per_run` / `max_pairs_per_run` enforced by the preflight |
| No unconfigured concurrent jobs | `claude.max_concurrency: 1`; `data/labels/claude_labeling.lock` (`compliance.policy.LabelingLock`) |
| Cumulative budget cap | `claude.max_budget_usd`, read from `data/labels/claude_usage.jsonl` across runs |
| No rate-limit bypass, no key or account rotation | one key from `ANTHROPIC_API_KEY` / `.env`; retries honour `retry-after` with backoff; never printed |
| No Claude Code subscription use as a bulk-labeling API | labeling only through the API key path in `ClaudeTeacher`; Claude Code sessions never label images by hand |
| No reasoning tokens | section 4 |
| Stop immediately on policy-related errors | `refusal` stop reason, 400/401/403/404 raise at once (`PolicyStop`); no retry |
| Local caching | `data/labels/claude_cache.jsonl` keyed by sha256 + prompt version + model |

## 8. Pre-training compliance checklist

Run before every training, pseudo-labeling, fine-tuning or evaluation workflow. `scripts/preflight_check.py`
performs the mechanical items; the rest are yours.

1. Read `CLAUDE.md` and this file.
2. Re-check the provider pages in section 2; note the date and any wording change here.
3. State the intended use of every AI-assisted label (`--intended-use training|evaluation`).
4. Confirm the permission status: training on Claude outputs is `permitted` only with a written reference.
5. If the terms are unclear or the status is `uncertain`: stop the Claude workflow and ask the project owner
   for clarification. Do not proceed on the assumption that label-only output is exempt.
6. Run `python scripts/preflight_check.py` (training) and, for Claude runs,
   `python scripts/preflight_check.py --mode labeling --intended-use <use> --images N --pairs M`. Both must
   PASS. Reports land in `reports/preflight_*.json`.
7. Verify every image passes the registry (dataset approved and per-record license allowlisted) and every
   label has provenance with `training_permission: permitted`.
8. Verify `source_weights.mock` is 0 and no restricted source carries weight.
9. For paid Claude runs: budget cap, batch bounds, single job, API key present, prompts audited.
10. Record the outcome (what ran, on what, with which permission basis) in `REPORT.md`.

## 9. Recording an authorization

- Written authorization from Anthropic: set `FILMLOOK_ANTHROPIC_TRAINING_AUTHORIZATION_REF=<reference>` in
  `.env` (git-ignored), or set `claude.training_authorization.status: written_reference` and `reference` in
  `configs/labeling.yaml`. Keep the document with the project records. Either makes the status `permitted`.
- Evaluation-only approval by the project owner: `claude.evaluation_use: {approved: true, approved_by: <name>,
  recorded: <YYYY-MM-DD>}`.
- Denial: `claude.training_authorization.status: denied` → `not_permitted`; stored Claude labels become
  unusable automatically at the next preflight.

## 10. Change log

- 2026-11-07: policy created; permission classification, provenance records, preflight checks and tests added.
  Status of Claude-as-teacher moved from "confirmed by owner" to **uncertain / blocked** pending a written
  reference. No Claude labels existed at that time; no training was run.
- 2026-11-07 (later): exploratory categorization workflow added (section 4a): `exploratory` intended use gated on
  `claude.exploratory_use`, `data/annotations/{claude_exploratory,training_approved}`, export gate, training
  preflight check `categorization_annotations`. `CLAUDE.md` gained the "Photography Classification Compliance"
  section. No Claude requests were made: no API key, and no exploratory-use approval recorded.
- 2026-11-07 (later still): owner decision recorded (`user_confirmed` → permitted, unverified, no reference).
  Categorization schema 2.0 / taxonomy v2, QA checks, DINOv2-small heads with baselines and a readiness gate
  (`annotation/heads.py`, `scripts/train_category_heads.py`). 20-image interactive pilot annotated. No heads trained:
  the readiness gate refused.
