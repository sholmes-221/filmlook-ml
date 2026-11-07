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
    with less content leakage but more generator fingerprint (share 0.594 vs 0.541, E27a); Gram statistics are worse everywhere; layers >= 9 are content-heavy. The real-film
    check used (BlueNeg blue-corrupted) cannot separate candidates (chance 0.565, CI +-0.13), so the choice of target
    stays open until the Commons evaluation (E27). (medium on synthetic; none on real film yet)

18. **The layer-3 gain is bound to the filmsynth-trained projection, not to a scanner or photographer shortcut.** Under a
    generator-swapped family SupCon projection (trained on the 4 sdxl families), both teachers collapse on held-out
    filmsynth val (L3 0.203 [0.183, 0.221] vs L4+L8 0.192 [0.166, 0.223]; chance 0.11) and tie on zimage (0.413), so the
    E26/E27 advantage (0.572 vs 0.467) exists only when the projection is trained on the filmsynth pipeline: the
    projection is generator-specific whichever layer feeds it. On 265 CommonCatalog film-scan originals (138 uploaders)
    L3 recovers the scanner family less than L4+L8 (probe 0.423 vs 0.457, majority 0.506) and its same-uploader share is
    lower (0.014 vs 0.019); explicit28 alone recovers the scanner best (0.581). E28, rejected under its pre-registered
    rule; the teacher target stays L4+L8. (medium; analysis data only)

19. **Training the family projection on two pipelines does not make it generator-invariant.** A SupCon projection on
    the union of filmsynth and sdxl train families (E30) lifts sdxl val B from 0.347 to 0.396 and keeps filmsynth
    (0.457 vs 0.467), but the held-out generator moves only from 0.342 to 0.351 [0.324, 0.378] and the generator share
    rises from 0.541 to 0.635: the projection learns both fingerprints rather than a shared treatment code. Rejected
    under its pre-registered rule. Generator invariance has to come from the features or the training signal, not from
    pooling pipelines. (medium; synthetic val only)

20. **On an independent real-film set, same-stock retrieval from uploader tags is at chance for every embedding, on two snapshots.** CommonCatalog film scans with uploader stock tags (`commoncatalog_film`, class B): first snapshot 487 scans / 175 uploaders, second 905 / 252 (analysis half 445 scans, 124 uploaders, 356 uploader-excluded queries). Same-stock p@5 is 0.04-0.07 for every candidate against chance 0.044 (hybrid teacher 0.051 [0.036, 0.069]; L3 teacher 0.055); with stock-line labels the teacher is 0.078 [0.065, 0.091] vs 0.066 and the best student 0.101 [0.087, 0.118]; film-type retrieval is 0.44-0.49 vs 0.264 but carries the monochrome cue. Raw DINOv2-global and colour statistics leak the photographer (uploader share 0.14 / 0.10 vs prior 0.03); the SupCon hybrid and the students do not (0.04-0.07). E27b inconclusive twice; the L3 target is closed without adoption; no student promoted. (high for the null result on this label source; the labels are uploader claims)

21. **On CC0 museum prints the hybrid teacher retrieves the physical process above chance without artist leakage; the real-distilled students do not.** aic_photographs analysis half (306 prints, 111 artists; E29): process p@5 artist-excluded 0.532 [0.453, 0.596] vs chance 0.389 for the hybrid teacher with artist share 0.10 (prior 0.017); raw DINOv2-global and colour statistics reach 0.58-0.62 but leak the artist (share 0.21-0.22); student_style_fs_v2_real, l3_real and kdsdxl fall to 0.445-0.468 with lower CIs below chance, while the synthetic-only students keep 0.48-0.51. Real-image distillation did not transfer to this third domain. (medium; historic prints, process correlates with era)

22. **Across photographers, the only film-process signal left on Flickr scans is a weak colour-type cue, and the teacher does not carry it cleanly.** Colour scans only (295, 93 uploaders; E31): negative-vs-reversal p@5 with the uploader excluded is 0.670 [0.597, 0.726] for the hybrid teacher against chance 0.642 (rejected; uploader share 0.063 vs prior 0.025), 0.715 for colour statistics alone and 0.72-0.74 for the synthetic-only students (lower CIs above chance, but with uploader share 0.05-0.07). The real-film retrieval evidence for these embeddings therefore rests on within-photographer structure (BlueNeg), not on cross-photographer stock or process identity. (medium; uploader-claimed types, one source)

## What is not established
- That any embedding captures aesthetics as a person would judge them (no human evaluation exists).
- That same-stock retrieval generalises beyond one photographer's 1990s consumer stocks.
- Composition: a frozen 2×2 patch-grid probe reaches R² ≈ 0.23 only; the spatial student failed (E12).
- Photographer style: textual profiles only; no retrieval evidence.
