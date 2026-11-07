# FilmLook-ML — repository rules for Claude Code sessions

### MANDATORY — AI Training Data Compliance

Before performing any training, pseudo-label generation, or third-party AI-assisted annotation:

1. Read `docs/AI_DATA_POLICY.md`.
2. Verify the intended data usage is permitted by the provider's current terms.
3. Never request hidden chain-of-thought or internal reasoning.
4. Never conceal training intent from an AI provider.
5. Never use restricted model outputs as training targets.
6. Prefer human annotations and appropriately licensed datasets.
7. Stop and ask for approval when compliance is uncertain.

These rules are mandatory for every future Claude Code session working on this repository. They apply to
dataset labeling, pseudo-label generation, model training, fine-tuning and evaluation. They are enforced
mechanically by `scripts/preflight_check.py`, which also runs inside every training / evaluation entry point
(`filmlook_ml.training.data.load_train_data`) and inside the Claude labeling scripts. Do not bypass, weaken or
special-case the preflight; if it blocks, the fix is a recorded authorization or a different data source.

**Current status (2026-11-07, later):** `training_authorization.status: user_confirmed`. The project owner
confirmed that Anthropic permits this use (categorization and information extraction for the DINOv2-small
photography classifier). No written reference is on file and none was verified or invented. The permission
check classifies it as `permitted`, with that basis recorded in every annotation's provenance. If a written
reference arrives, record it; if the confirmation is withdrawn, set `status: denied` and every Claude label
becomes unusable at the next preflight. An actual API or platform policy restriction always stops the
workflow; it is reported and never bypassed.

### MANDATORY — Photography Classification Compliance

Before any dataset annotation or training:

1. Read the Anthropic usage restrictions (Usage Policy, Commercial Terms; quoted with dates in
   `docs/AI_DATA_POLICY.md` section 2).
2. Confirm the workflow is limited to specialized content categorization (photographer-inspired style, film
   aesthetic) and information extraction (observable visual attributes). No chatbot, no open-ended text
   generation, nothing that reproduces Claude's general reasoning, image understanding or text generation.
3. Verify permission before using model Outputs as training targets. Exploratory annotations
   (`data/annotations/claude_exploratory/`) leave that directory only through `scripts/export_annotations.py`,
   which requires the permission status `permitted` (a written reference, or the owner's recorded
   `user_confirmed`).
4. Never request chain-of-thought or hidden reasoning. Prompts are audited, and outputs are controlled labels
   plus at most three short evidence phrases.
5. Never perform prohibited model distillation or reverse engineering.
6. Never bypass Claude Code subscription or API restrictions. Bulk annotation runs only through the API-key path
   (`scripts/annotate_categories.py`). An interactive Claude Code session may annotate only a small pilot that
   the owner asked for in that session (`scripts/categorize_pilot.py`, at most the 20 planned images). It never
   loops `claude -p` or automates the subscription.
7. Keep exploratory annotations separate from approved training data (`data/annotations/training_approved/`).
8. Stop safely if the intended use is not permitted (status `uncertain` / `not_permitted`), or if an API or
   platform policy restriction occurs.

These rules must be checked before every future annotation or training task.

Also:

- Never rotate API keys or accounts, never bypass rate limits, never use Claude Code subscription access as a
  bulk-labeling API, never start a paid Claude run without `--no-dry-run --confirm-paid` and an explicit
  request from the project owner.
- Do not start model training or make bulk API calls unless the user asked for that in the current session.
- When adding a new label source or teacher, give every record a `provenance` (`labeling.schema.Provenance`)
  and extend `compliance/provenance.py`; the preflight fails closed on unknown sources.

## Repository map

| Area | Where |
|---|---|
| Policy / provenance / preflight | `docs/AI_DATA_POLICY.md`, `src/filmlook_ml/compliance/`, `scripts/preflight_check.py` |
| Runtime package (CPU, no network) | `src/filmlook_infer/` |
| Training package | `src/filmlook_ml/` (datasets, labeling, training, evaluation, export, style) |
| Dataset registry + licensing | `configs/datasets.yaml`, `DATASET_SOURCES.md`, `filmlook_ml.datasets.registry` |
| Labeling config (teachers, Claude budget, authorization record) | `configs/labeling.yaml` |
| Label storage | `data/labels/*.jsonl` (git-ignored), merged by `filmlook_ml.labeling.store` |
| Photographer-style module (JS) | `js/filmlook-style/`, Python reference in `src/filmlook_ml/style/` |
| Results and limitations | `REPORT.md` |

## Commands

```bash
source .venv/bin/activate
python scripts/preflight_check.py                                   # training/eval preflight (must PASS)
python scripts/preflight_check.py --mode labeling --intended-use evaluation --images 40 --pairs 10
pytest -q                                                            # tests/test_compliance.py covers the gates
```

Training and labeling commands are listed in `README.md`. The DINOv2-small backbone stays frozen (Stage A);
Stage B is off by default.

## Conventions

- Images, labels, embeddings, checkpoints, reports and `dist/` are git-ignored; never commit them or `.env`.
- `configs/datasets.yaml` `training_allowed` plus the per-record license allowlist is the only data gate;
  a Hugging Face card tag never overrides image copyright.
- Confidence values from any teacher are self-reports, not calibrated probabilities.
- No photographs by the named photographers are downloaded, used or shipped; style profiles are textual.
