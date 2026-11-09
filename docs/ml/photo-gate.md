# Photo gate: photograph vs non-photograph, before any provider call (2026-11-09)

## What it replaces

The website asked the vision provider (Gemini) for a `content_check` verdict inside the full analysis call, so a
screenshot cost a provider call and the visitor's daily allowance before it was refused, and local mode accepted
everything. The gate runs locally, first. The provider's verdict stays as the second layer (it also screens unsafe
content, which the gate does not attempt).

## Model

DINOv2-small (frozen, `facebook/dinov2-small` rev ed25f3a31f01), 224 px with the shared `dinov2_pixels`
preprocessing; features = [CLS, mean of patch tokens], each L2-normalised (768); logistic regression (C=3, balanced),
chosen on validation among logistic regression and Platt-calibrated linear SVMs (C grids) and CLS-only vs CLS+patch
features. Artifact: `photo-gate-20261109-6b8b875b.json` (768 weights, bias, thresholds) + `dinov2_small_cls_patch.onnx`
(fp32, 88 MB; int8 failed the 0.99-cosine gate). Thresholds from validation: REJECT below t_lo = 0.362 (at most 0.2 %
of validation photographs), ACCEPT at or above t_hi = 0.768 (at most 2 % of validation non-photographs); UNCERTAIN between continues to the provider.

## Data (all legally usable; labels by construction or catalogue record, never a model's opinion)

| part | photographs | non-photographs |
|---|---|---|
| train + val | CommonCatalog CC-BY originals, CommonCatalog film scans, Commons film scans, refdb public-domain FSA/OWI scans, AIC and Met public-domain photographs, ARTeFACT (2,674) | locally rendered documents, charts, posters, UI screenshots (family a); AIC paintings and prints (CC0, by catalogue class); Z-Image-Turbo illustrations (prompt family a); composites from colour photos: posters over a photo, UIs containing a photo; photos with captions as positives (2,454) |
| test (held out) | commoncatalog_film_hd (uploaders excluded elsewhere), PD12M iNaturalist field photos (694) | unseen generator families b (documents, charts, posters, UI), AIC drawings and posters (classes never trained), illustration family b, FilmAnalysis's own page screenshots (876) |
| hard | test photos blurred, motion-blurred, grain +28, darkened to 18 %, monochrome; photos with captions (photos) | UIs containing a photo, posters over a photo (black-and-white FSA photos; training composites used colour photos) |
| ambiguous | photorealistic generated images (reported only) | |

## Results (test read once)

| measure | value |
|---|---|
| photographs rejected (false rejection) | 0.14 % (1 of 694) |
| non-photographs accepted | 0.23 % |
| photographs uncertain | 3.6 % |
| non-photographs uncertain | 3.1 % |
| accuracy of decided cases | 99.8 % |

Per kind: documents, charts, posters, UI (unseen families) and real site screenshots 100 % rejected; AIC drawings
90 % rejected (9 % uncertain); AIC posters 95 %; illustrations family b 90 % (10 % uncertain). Hard photos: rejected
0 % (blur, grain, motion, monochrome, captions), 0.8 % (very dark). UIs containing a photo 97.5 % rejected; posters
over a photo 67 % rejected, 17 % uncertain, 17 % accepted (the weakest case). Photorealistic generated images are
mostly accepted or uncertain: the gate does not and cannot claim to detect AI generation.

CLS-only features were clearly worse (photographs rejected 1.4 %, uncertain 17 %): the patch mean carries the texture
that separates paintings and drawings from photographs.

## Serving

Oracle E2.1.Micro, throwaway container from the app image (--cpus=1, 640 MB): p50 385 ms, p95 707 ms, cold load 2.2 s,
whole-process peak RSS 237 MB; vectors identical to the workstation. The website module (`filmlook/photo_gate.py`)
reproduces these outcomes on held-out images through the API's own `prepare_image` path.

## Limits

Synthetic negatives dominate the non-photo classes; real-world screenshots are represented only by FilmAnalysis's own
pages; anime illustrations are generated, not collected; photos of paintings are treated as non-photographs (museum
reproductions); posters built on photographs remain the weakest case.
