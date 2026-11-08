# Reference data for the Gemini-free analysis: what exists, its provenance, and what it may be used for

Written 2026-11-08. Every set below is already registered in `configs/datasets.yaml` / `DATASET_SOURCES.md` with
per-record licences; this page records how the MVP uses each one and what it must not be used for. "Verified"
metadata means a catalogue record or EXIF/owner record; "inferred" means a model or a rule produced it.

| set | source and licence | size used | metadata (verified / inferred) | role in the MVP | leakage status |
|---|---|---|---|---|---|
| CommonCatalog CC-BY originals (main manifest, `train` split) | Flickr Commons-derived, CC-BY-2.0 / CC0 / PDM per record | 968 originals (+ 8,064 derived edits, synthetic) | capture medium from EXIF (verified, weak); edit parameters (known, by construction) | training of `checkpoints/best` (v1) and `checkpoints/student_sup`; PCA fit of the frozen student | training data: never used for any reading |
| CommonCatalog `test` originals | same | 89 (30 film scans, 41 digital camera, 18 smartphone) | EXIF capture medium (verified, weak) | E58 FIT/READ halves for the film-likeness calibration (grouped by `group_id`) | held out from both models |
| `commoncatalog_film_hd` (fresh snapshot) | Flickr CC-BY-2.0, uploaders excluded from every earlier set | 324 scans with a stock claim (474 total) | uploader stock claim (verified as a claim, not as fact), scanner EXIF | E53 and E57 readings; E58 positives (film) | fresh for E53; E57/E58 are post-hoc on the same images |
| The Met Open Access photographs | CC0 | 378 analysis-half images (test half untouched) | catalogue process (verified) | E53/E57 readings; E58 prototype *evaluation* | fresh for E53; test half still unread |
| Art Institute of Chicago photographs | CC0 | 593 (analysis half 306 used) | catalogue process (verified) | E58 prototype *fit* (spent set) | spent |
| ARTeFACT film originals | CC BY 4.0, one photographer | 44 | owner-recorded emulsion (verified) | E53 same-emulsion reading | fresh for E53 |
| BlueNeg | owner-recorded stock, previews | 491 frames | owner record (verified), date stamps | earlier film-vs-digital probes (`student_sup` cross-source) | spent |
| synthetic edits and SDXL scenes | generated here / permissive | 8,064 edits, 40 scenes | parameters (known) | v1 training only | not photographs; never a reference |

Rules that follow:

- No photograph by a named photographer in the website's database is downloaded, indexed or shipped; photographer
  matching stays the measured 17-axis ranking against *textual* profiles (copyright policy in the website's
  `CLAUDE.md`). A licensed per-photographer image bank does not exist and is not planned without the owner's
  explicit sourcing decision.
- Film-stock labels in every set are **claims or records about the roll**, not about the look; across
  photographers no representation recovers them (E53/E57), so they are never surfaced as identification.
- The fresh sets have now been read more than once (E53 pre-registered, E57/E58 post-hoc). The next genuinely
  independent read needs new data: a new CommonCatalog snapshot with excluded uploaders, and a **fresh digital set**
  (the E58 negatives were only the 59 test originals, so its AUROC CI is wide).
- User uploads are never added to any reference set; the website keeps consented contributions encrypted and
  separate (`FILMLOOK_TRAINING_KEY`), and nothing in this repository reads them.

## Storage shape for a reference index (if one is built later)

SQLite table `references(image_id, source, license, attribution, photographer, film_stock_claim, tags_json,
split, embedding_version, quality_flags)` beside a memory-mapped `embeddings.int8` (PCA-32, 32 bytes/row) and a
`norms.f16`; 100k rows = 3.2 MB of vectors and a few MB of metadata. Only embeddings and metadata would ship;
licences and attribution travel with every row (`service/build_bank.py` already enforces the licence gate).
