# filmlook-style: offline photographer-style similarity (pure JavaScript)

Copy `dist/filmlook-style.js` into the FilmLook Mini Tool package. It is **one classic script** (ES2017, no
dependencies, no `eval` / `Function` constructor / WebAssembly / network), **37 KB** (15 KB gzipped).
It exposes the global `FilmLookStyle`.

> **What it is:** a resemblance score between a photo's *measurable* rendering (colour, tone, contrast,
> texture, light distribution, composition) and short descriptions of nine photographers' widely published
> visual approaches. **What it is not:** it does not identify the photographer, authenticate authorship, or
> judge artistic merit. It is allowed to answer "no reliable match", and for most ordinary photos it should.
>
> **Validation status (v1.0.0): UNVALIDATED against human judgement.** No human or authorized-teacher style
> judgements exist yet; see "Evaluation" below and the training repo's `REPORT.md`.

## Use in the Mini Tool

```html
<input type="file" id="pick" accept="image/*">
<script src="./filmlook-style.js"></script>
<script src="./app.js"></script>          <!-- inline scripts are blocked by the sandbox -->
```

```js
// app.js - file input -> blob URL -> <img> -> analyzeImage (same-origin, so getImageData is allowed)
document.getElementById('pick').addEventListener('change', function (e) {
  var url = URL.createObjectURL(e.target.files[0]), img = new Image();
  img.onload = function () {
    var result = FilmLookStyle.analyzeImage(img, 'zh');   // 'en' | 'zh'
    URL.revokeObjectURL(url);
    render(result);
  };
  img.src = url;
});
```

`analyzeImage` draws the photo at 2x the working size (progressive halving, each `drawImage` step <= 2x), then
applies its own exact 2x2 box average (`box2x`) down to <= 384 px on the long side. That final averaging step
makes the result largely independent of how the WebView resamples (measured below). If you already have pixels,
call `FilmLookStyle.analyzeImageData({width, height, data})` with RGBA at <= 384 px (or `box2x` of a 2x image).

### Worker (optional)

The Mini Tool allows one classic dedicated Worker (client >= 9.53). See `examples/worker.js`: draw the 2x image on
the main thread with `FilmLookStyle.targetSize(w, h)` x 2, transfer `imageData.data.buffer` to the worker, and call
`analyzeImageData(box2x(img))` there. Feature-detect `Worker` and fall back to the main thread.

## Output (`schema/style_output_v1.json`, schema `style-1.0`)

```jsonc
{
  "schema_version": "style-1.0", "model_version": "filmlook-style-v1.0.0", "language": "en",
  "status": "match",                 // or "uncertain": show `message`, do NOT show a photographer as a match
  "message": "Closest measurable style resemblance: Stephen Shore.",
  "monochrome": false,
  "matches": [                       // always the top 3, best first; only `reliable: true` entries are matches
    { "id": "stephen_shore", "name": "Stephen Shore",
      "similarity": 0.52,            // share of the style's distinctive measurable signature present (0 = typical photo)
      "percentile": 1.0,             // rarity: fraction of ordinary reference photos scoring lower
      "reliable": true, "strength": "moderate",
      "evidence": ["crisp, precise detail", "gentle, low contrast", "bright, open exposure"],
      "counter_evidence": ["this style typically shows clean, fine texture; this photo does not"],
      "basis": "neutral, even colour, crisp large-format detail, gentle contrast" },
    ...
  ],
  "dims": { "monochrome": 0, "saturation": 0.19, ... },   // measured style dimensions, 0.5 = typical
  "disclaimer": "Style resemblance is computed from measurable characteristics ... does not identify the photographer ..."
}
```

Rules built into the engine:
- **Monochrome gate.** Colour-defined styles (Leiter, Eggleston, Webb, Herzog, Kawauchi, Haas, Shore) are not matched to black-and-white photos, and colour evidence is never quoted for them.
- **Evidence from measurements.** Evidence phrases cite only dimensions the photo actually measures close to the style's targets. `counter_evidence` names the strongest disagreement.
- **Reliable match.** A match needs all three: similarity >= 0.5, rarity percentile >= 0.9 against ordinary photos, and a monochrome-compatible style. Flat, low-information images always return `uncertain`.
- **Deterministic.** The same pixels give the same JSON.

Styles in v1: Saul Leiter, William Eggleston, Daido Moriyama, Alex Webb, Fan Ho, Stephen Shore, Fred Herzog,
Rinko Kawauchi, Ernst Haas.

## Measured (training repo, 2026-11-06)

| | result |
|---|---|
| Python/OpenCV vs JS parity (48 features, 14 fixtures) | max abs diff < 1e-9; engine JSON identical; `box2x` bit-identical |
| ES2017 | bundle parses with acorn `ecmaVersion: 2017`; test suite bans `?.`, `??`, eval, WASM, fetch, ... |
| Runtime, Node 24, 1 core of an i9-14900KF, 768 px intermediate -> result | p50 8.7 ms, p95 28 ms, first call 64 ms, ~23 MB transient heap |
| Runtime on phones | **not measured**. Expect several times slower on low-end Android WebViews; measure in the Mini Tool before launch |
| Top match unchanged under: WebView-like bilinear halving / nearest / JPEG q75 / 4% crop / +8% exposure / flip | 97.7% / 87.6% / 92.2% / 95.9% / 83.9% / 100% (218 held-out photos) |
| Reliable-match rate on ordinary held-out photos | ~26% overall (smartphone ~43%, film scans ~19%) |

Known failure mode: processing artefacts can look like style. For example, an over-processed HDR edit can read as "Stephen Shore-like" (flat contrast plus crisp detail).

## Evaluation and updating

The training repo has the full loop:
- **Collect judgements:** human judgements via the review app's Style mode; Claude-teacher judgements via `scripts/style_labels.py`, which is gated and budget-capped.
- **Evaluate:** `scripts/style_evaluate.py` compares this profile engine with linear, random-forest and MLP models on the same JS features, and with a DINOv2 baseline. It reports top-1/top-3/MRR with uploader-grouped CV and keeps teacher agreement separate from human agreement.
- **Ship a learned model:** a learned model can ship in `model-data.js` (`learned` block). The JS inference for all three model types is already implemented and parity-tested.

Rebuild with `python scripts/style_export_js.py`, then run `node --test test/style.test.js`.

## Licensing

No photographer's photographs were used to build or calibrate this module, and none are distributed:
- **Style profiles** are short textual descriptions of measurable characteristics.
- **The rarity reference pool** is 601 CC-BY 2.0 Flickr photos by ordinary uploaders (CommonCatalog). Only their aggregate score quantiles ship, not the images.
- **Test fixtures** are CC0 (PD12M) and synthetic images.

Photographer names are used descriptively. Do not present results as endorsement by, or affiliation with, the photographers or their estates.
