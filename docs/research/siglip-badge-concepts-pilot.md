# Pilot: local zero-shot concept detection for badges (SigLIP base), 2026-11-09

Question: can a local open model replace the vision provider's scene words for badge triggers (umbrella, moon, duck,
bird, street, night ...) without false unlocks? Model `google/siglip-base-patch16-224` (Apache-2.0, already in the HF
cache, run offline on CPU), one prompt per concept ("a photo of ..."), sigmoid score, threshold 0.1.
Images: 20 CC0 / public-domain Commons photographs labelled by eye (the 光影收藏家 calibration set and the badge e2e set).

| Photo | Detected (≥ 0.1) | Truth | Gemini (badge e2e, same photo) |
|---|---|---|---|
| man under an umbrella | umbrella 0.75 | umbrella, street | umbrella (object list), street |
| full moon on black | moon 0.45 | moon, night | moon, night |
| moon behind branches | — | moon, night | missed the moon too |
| rain on umbrella fabric | — | none | — (said "fabric with raindrops") |
| stapler | stapler 0.98 | stapler | — |
| candle, two stage photos, high-key bird | candle / stage / bird | same | not run |
| duck in sun glitter (silhouette) | — (duck 0.006) | duck, bird | not run |
| 11 scenes with none of the concepts | — | — | — |

Reading: no false positive in 20 photos at 0.1; misses on hard cases (a silhouetted duck, a moon behind branches, "street"
as a vague concept). High precision, moderate recall, n = 20: promising, not validated. Before it could replace the
provider for any badge it needs: a per-concept CC0 test set with negatives (≥ 30 positives per concept), thresholds
chosen on one split and checked on another, a side-by-side with the provider, and an instance memory check (SigLIP base
is about 400 MB in RAM; the production app container has 640 MB).
