# filmlook-infer: integration guide

Standalone CPU inference package for the FilmLook film-aesthetics model. It needs **no torch, no GPU,
no network and no LLM**. Runtime dependencies: `numpy`, `pillow`, `scipy`, `onnxruntime` (pinned in
`requirements.txt`).

## Layout

```
filmlook_infer/                     runtime code (copy as-is; do not edit)
filmlook_infer/schema/filmlook_output_v1.json   JSON Schema of the result (versioned)
model/model.onnx + model/meta.json  model artifact (meta.json = preprocessing config + model version)
examples/                           sample input + outputs (en, zh) produced by this package
TRAINING_DATA_ATTRIBUTION.jsonl     CC-BY credits for photos used in training
```

## Use from Python (recommended: load once per worker process)

```python
from filmlook_infer import FilmLookAnalyzer, InvalidImage, UnsupportedLanguage

analyzer = FilmLookAnalyzer("model")          # ~0.1-0.3 s; keep it alive, do NOT load per request
result = analyzer.analyze("photo.jpg", language="zh")   # "en" | "zh"; path or binary file-like object
```

## Use from any language (CLI, one process per call: slower, includes cold start)

```bash
python -m filmlook_infer.cli photo.jpg --lang zh --model model      # prints one JSON object per image
# exit code 0 = ok, 2 = invalid input ({"error": "invalid_image" | "unsupported_language", "message": ...})
```

## Output (schema v1.0)

See `filmlook_infer/schema/filmlook_output_v1.json` and `examples/`. Fields:

| field | meaning |
|---|---|
| `schema_version`, `model_version` | contract versions; the schema version changes only when fields change meaning or are removed |
| `film_score` | int 0-100, how convincingly film-like the rendering is |
| `attributes` | six 0-1 scores, **higher = more film-like/natural**; `color_response` is `null` for monochrome |
| `attribute_names` | localized display names |
| `style_tags`, `style_tag_names` | measured descriptors (warm/cool/neutral, muted/vivid, soft/crisp, high/low-key, contrast, grain); never colour tags on monochrome |
| `monochrome` | measured |
| `confidence`, `confidence_level` | heuristic (seed-ensemble agreement + in-distribution check). **Not a calibrated probability.** Show a softer message when `low` |
| `strengths`, `weaknesses` | 0-3 short localized phrases from deterministic templates |
| `summary`, `disclaimer` | localized text |

Determinism: identical bytes + identical package version give identical JSON (single-threaded ONNX Runtime by
default). The model does **not** identify camera, film stock or photographer, and it is not an artistic-merit score.

## Deployment notes (Xiaohongshu Mini Program backend)

* Run inference **server-side**; the Mini Program uploads the image to your backend. Do not assume the Mini Program
  runtime can execute ONNX.
* Load `FilmLookAnalyzer` once per worker; serialise requests on small instances (1 worker, queue).
* Inputs are bounded: JPEGs decode via DCT scaling and are downscaled to `meta.json:max_side`; images above
  60 MP are rejected (`invalid_image`).
* Re-encode/strip EXIF on your side if you store images; the analyzer only reads pixels and EXIF orientation.
* Measured latency/memory for this artifact: see `model/meta.json` (`benchmark` key, when present) and the
  training repo's `deployment/README.md`. Those numbers come from a development workstation and Docker CPU/memory
  limits, **not** from real deployment hardware.
* Upgrading: compare `schema_version` before parsing; ignore unknown fields (minor versions may add fields).
