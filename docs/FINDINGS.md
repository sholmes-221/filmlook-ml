# Validated findings

What the evidence supports about FilmLook-ML's representations, with the strength of each claim. Each line names
the experiment(s), the report file and the uncertainty. "Synthetic" means the filmsynth / SDXL / Z-Image
derivatives of 100 CC-BY sources; "real" means BlueNeg (491 scans of 1990s consumer stocks by one photographer,
owner-recorded stock labels, 27 rolls) or CommonCatalog originals with EXIF capture type. No human judged any
aesthetic at any point; nothing here is a measurement of beauty, and no model identifies a film stock or a
photographer.

Confidence scale: **high** = replicated across splits/generators with non-overlapping CIs; **medium** = one test
split, CI clear of chance; **low** = suggestive, CI touches chance or n is small.

## About the representations

1. **Frozen DINOv2-small global features are a content embedding, not a style embedding.** (high)
   Benchmark v1 test: A2 style-gap ratio 0.06, own-source rank-1 0.99, subject probe at the content ceiling,
   same-treatment retrieval 0.13 vs chance 0.11. `reports/style_benchmark_test.json`.

2. **Style information lives in intermediate-layer patch statistics plus explicit measured statistics; a small
   SupCon projection on them is style-dominated on synthetic data.** (high on synthetic)
   hybrid + family SupCon: A 0.93, A2 1.51, B filmsynth p@5 0.44 [0.39, 0.48] vs 0.11, content-neighbour Jaccard
   0.09; replicated on val (0.47). E6/E7/E9/E9b: L4+L8 statistics outperform any single layer and the last layer.

3. **The projection still encodes some scene and generator signal.** (high)
   Own-source rank-1 0.33, subject probe 0.34 vs 0.22 majority (0.48 ceiling), same-generator top-5 share 0.59 vs
   0.40 prior. Attempts to remove it (E16 texture neutralisation, E17 rich descriptors, E19 hard negatives +
   content decorrelation) either re-encoded content or raised the generator fingerprint.

4. **"Treated vs untreated" is not an aesthetics target.** (high)
   Within a generator type it is a generator signature (E2-E4); across generator types it reduces to a single
   statistic, saturation clipping (E15, E18, `reports/leakage_audit*.json`). It is retired.

5. **Film authenticity is a texture property; colour/tone cannot separate film scans from digital captures.** (high)
   E15: explicit colour/tone statistics give chance-level film-vs-digital AUROC; DINOv2 mid-layer texture
   statistics and the EXIF student carry the signal. Authenticity and aesthetics stay separate outputs.

6. **The hybrid family-SupCon embedding transfers to real scans, weakly.** (medium)
   BlueNeg same-stock p@5 0.25 [0.16, 0.31] vs chance 0.12 (roll bootstrap), test-only run B1; earlier sessions
   0.25-0.26 with the same construction. The 4.4 MB student (0.18 [0.13, 0.23]) and the 16 colour/tone statistics
   (0.18 [0.09, 0.28]) are not separable from the raw DINOv2 reference (0.17) on 27 rolls.

7. **A label-free confidence (top-1 cosine similarity) orders the embedding's errors on real scans.** (medium)
   Same-stock rank-1 precision 0.22 → 0.37 at 50 % coverage → 0.50 at 25 % (48 queries, ±0.14); AURC 0.59 vs 0.78
   unordered. The student's confidence is informative on synthetic data only. `reports/abstention_test.json`.

8. **Diffusion img2img treatments at strength 0.35 are not usable style labels.** (high)
   Two treatments of the same source differ by 0.19× (SDXL) / 0.35× (Z-Image) the JPEG+rescale nuisance distance
   under raw features (filmsynth 1.6×); same-treatment retrieval within SDXL/Z-Image is at chance for every
   embedding including those trained on the labels (E19). `reports/diffusion_treatment_distinctness.json`.
   Whether a higher strength fixes this is E10's question (see the experiment log).

## About the data and the pipeline

9. **Grain drawn independently of the label removed the cross-generator leakage** (film-synth-v2; E4/E5), and
   standardisation + VAE controls are hygiene with little effect on their own. (high)
10. **A third generator (Z-Image-Turbo) generalised the generator-invariant signal (0.63 [0.52, 0.75]) but improved
    nothing on real data** (E8, E14); scaling synthesis (E13) is therefore not justified. (medium)
11. **The EXIF student's authenticity head is 0.78 [0.69, 0.88] within source with ECE 0.20**, not the 0.86 quoted
    earlier; its probabilities are not calibrated. (high) `reports/film_authenticity_eval.json`.
12. **The benchmark's real-film evidence rests on one source.** CI widths of ±0.08 on D are the floor with 27 rolls;
    a second rights-cleared film source with stock records is worth more than any further model variant. (high)

13. **The student's failure on real scans is an input-domain shift of the distillation, not a capacity limit.**
    student_fs_v2 reproduces its teacher's neighbourhoods on filmsynth (top-10 Jaccard 0.35) but not on real
    photographs (0.04-0.05); distilling on unlabeled real CommonCatalog train originals (E23b) raises held-out real
    agreement (Spearman 0.53 -> 0.62) and BlueNeg-corrupted agreement (0.38 -> 0.47) with no loss of synthetic
    treatment retrieval (val B 0.557 vs 0.520), while SDXL outputs as distillation inputs do not (E23c). (medium:
    agreement metrics are proxies; no real-film labels were used)
14. **Scanner family is not what style embeddings retrieve on real scans.** On CommonCatalog film scans with EXIF
    scanner fields (train+val only), an uploader-grouped scanner probe sits at the majority rate (0.51) for every style
    embedding; only colour_tone16 recovers it slightly (0.59). Photographer identity leaks more than the scanner
    (same-uploader share 0.02-0.05 vs prior 0.007). (medium: 4 coarse scanner families, EXIF coverage partial)
15. **Linear content residualisation destroys the treatment signal.** Regressing hybrid features on DINOv2-global and
    projecting the residual (E25) drops val B to 0.12-0.26 and raises own-source rank-1 to 0.57-0.69; a
    random-projection control keeps B at 0.40. DINOv2-global is not a content-only vector; subtraction is the wrong
    operation, selection of statistics (E26) is the open alternative. (high)
16. **A second real-film source exists and is rights-cleared for evaluation.** Wikimedia Commons' film-brand category
    tree yields 5,632 CC0/PD/CC-BY(-SA) photographs from 401 uploaders across 112 uploader-asserted stocks, with
    roll-level categories for some and scanner categories for 162. Labels are uploader claims (like CommonCatalog
    tags), not owner records (BlueNeg). Acquisition is rate-limited by Wikimedia and resumable; the dataset is
    evaluation-only (`training_allowed: false`). (high for existence; the evaluation itself is pending)

17. **Which DINOv2 statistics carry treatment: early-layer first moments, not Gram matrices, not late layers.**
    Over 12 layers x {mean++std, 32-d projected covariance} (E26, analysis data only), mean/std of layers 2-4 match or
    beat the L4+L8 pair on cross-scene treatment retrieval (L3: val B 0.572 [0.538, 0.603] vs 0.467 [0.418, 0.514])
    with less content leakage; Gram statistics are worse everywhere; layers >= 9 are content-heavy. The real-film
    check used (BlueNeg blue-corrupted) cannot separate candidates (chance 0.565, CI +-0.13), so the choice of target
    stays open until the Commons evaluation (E27). (medium on synthetic; none on real film yet)

## What is not established
- That any embedding captures aesthetics as a person would judge them (no human evaluation exists).
- That same-stock retrieval generalises beyond one photographer's 1990s consumer stocks.
- Composition: a frozen 2×2 patch-grid probe reaches R² ≈ 0.23 only; the spatial student failed (E12).
- Photographer style: textual profiles only; no retrieval evidence.
