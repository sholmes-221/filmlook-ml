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

23. **The E29 museum-print signal is rendering, not format, but format inflates it.** With the 76 stereo cards removed and the candidate bank restricted to the query's decade (E32), the hybrid teacher retrieves the print process at 0.483 [0.401, 0.560] vs chance 0.358 (full regime 0.532 vs 0.389; a 2-d aspect/stereo baseline alone reaches 0.479 vs 0.389). Colour-tone statistics (0.520 vs 0.266 with stereo removed) and the DINOv2 L4+L8 statistics (0.489) carry more of it than the family SupCon projection (0.410). (medium; 306 prints, 111 artists, historic processes)

24. **The real-distilled students lost E29 because their distillation target is collapsed.** The family SupCon projection has effective rank 3.9 on Flickr film scans and 3.3 on museum prints; the students' 64-d outputs have 2.1-3.2 with mean pairwise cosine 0.63-0.79 on monochrome prints. Label-free whitening restores every student's museum lower CI above chance (0.50-0.54 fitted on the evaluated half, 0.48-0.56 fitted on independent CC-BY originals; E32, E36), and artist-grouped probes recover the teacher from the students at R^2 0.37-0.43. (high for the diagnosis; the fix is open: E38)

25. **In the frozen DINOv2-Small statistics, film/process information and photographer identity occupy the same directions.** Every head that keeps the real-photograph signal clusters by photographer: whitened PCA heads reach stock-line p@5 0.103 [0.080, 0.128] vs 0.066 and museum 0.565 vs 0.358 with uploader share 0.14-0.17 (projection 0.06, prior 0.03) and artist share 0.21-0.28 (projection 0.10); VICReg rank control moves information and leakage together (E34); removing the between-uploader LDA subspace estimated on 92 CC-BY uploaders does no more than removing random directions (E35); independent whitening of the students raises artist share to 0.11-0.18 (E36). The scanner probe stays at its 0.44 majority throughout. (medium-high; three sources, linear heads only)

26. **Within a photographer, stock lines are separable, mostly by the monochrome cue.** Among 15 uploaders with two or more stock lines (182 scans; E37), the teacher ranks same-line scans from other days above other-line scans with AUC 0.640 [0.563, 0.734] against a within-uploader permutation null of [0.454, 0.547] (p < 0.001; colour-tone alone 0.681; real-distilled student 0.673). On colour scans only (8 uploaders) the teacher is 0.555 [0.473, 0.682], p = 0.06. Stock similarity is therefore defensible within a user's own library and across colour/monochrome, not across photographers. (medium; uploader-claimed lines, 15 uploaders)

27. **Raising the student's rank helps on real photographs only as far as scene content is let back in.** A real-distilled student with a VICReg anti-collapse target (E38/E39, three seeds) goes from effective rank 3.2 to 5.0-5.9, museum p@5 0.419 -> 0.457 (one seed's lower CI 0.004 below chance), within-uploader stock AUC 0.660 -> 0.692, no synthetic cost, but artist share 0.058 -> 0.133 and own-source rank-1 0.17 -> 0.27 (chance 0.067). A label-free whitened target reaches museum 0.540 and stock-line 0.113 [0.088, 0.143] at own-source rank-1 0.76. On the frozen features every head that gains on the real metrics is content-dominated (own-source rank-1 0.71-1.00; projection 0.30; E40). The content-invariant embeddings are the collapsed ones. (high for the diagnosis; medium for the vic3 gain, which is within one seed of chance on the museum set)

28. **The end-to-end student's penultimate layer, not its distilled head, is the usable lightweight representation.** Three seeds of the E23b recipe (E42/E43): the 576-d pooled features have effective rank 10.8-11.2, own-source rank-1 0.31-0.32 and museum p@5 0.52-0.54 with lower CIs above chance where the 64-d head has rank 3.2 and 0.40-0.43; a PCA-64 fitted on CC-BY train originals keeps all of it (rank 8.6-8.9, museum 0.50-0.53, within-uploader AUC 0.63-0.67, synthetic family p@5 within 0.02). On the held-out halves (E44, read once) the PCA-64 candidate reads museum 0.593 [0.504, 0.683] vs 0.448, stock-line 0.103 vs 0.077, BlueNeg same-stock 0.191 [0.126, 0.263] vs 0.117 (teacher 0.248), benchmark B 0.589, own-source 0.32; the within-uploader AUC lower CI (0.452) failed the pre-registered rule, as it did for every comparator on that half. Not promoted; recorded as the research candidate. Trained without the teacher term the pooled layer reads the same (museum 0.492-0.494, own-source 0.24-0.31; E45, missed the rank bound by 0.01), so the signal comes from the end-to-end synthetic supervision, not from distillation. (high for the analysis-half pattern; medium for the held-out gain, one reading)

29. **A rank regulariser on the student's own pooled layer buys real-photograph metrics with scene content, end to end; the low-variance directions of the pooled layer are the content directions.** Three seeds of the no-teacher recipe (`--kd 0`) with a VICReg variance/covariance term (weight 1) on the 576-d pooled layer (E46): PCA-64 effective rank 22.4-23.0 (raw layer 42), museum p@5 0.520 (lowest lower CI 0.419 vs chance 0.358), within-uploader stock AUC 0.657, colour-only AUC 0.620, but own-source rank-1 0.61 (plain no-teacher seeds 0.28, teacher-distilled 0.28; chance 0.067), artist share 0.18 vs 0.15 and synthetic family p@5 0.575 vs 0.627. The plain no-teacher seeds read rank 7.99-8.01, museum 0.489, within-AUC 0.631. The reduction curve of the teacher-distilled students (E47, three seeds, PCA fitted on CC-BY train originals) says the same from the other side: PCA 8/16/32/64/128 give museum 0.447/0.486/0.507/0.515/0.523 and own-source 0.18/0.23/0.26/0.28/0.29 with within-AUC 0.63-0.64 throughout, while whitening the 64 components lifts museum to 0.539 at own-source 0.60; int8 quantisation of PCA-32/64 costs nothing (0.503/0.512). PCA truncation is a content filter, and 32 dimensions keep the signal (the pre-registered rank-8 bound, written for 64 dimensions, falsified PCA-32 at 7.2). (high; three seeds each, analysis halves)
30. **The FilmLook training is necessary for content invariance and within-photographer stock separation, not for museum process retrieval, which every representation reaches.** The ImageNet-pretrained MobileNetV3-Small pooled layer with no FilmLook training, same export and PCA-64 (E48): museum 0.528 [0.446, 0.605] (trained seed-mean 0.515), within-uploader AUC 0.550 [0.464, 0.611] (trained 0.625-0.670), own-source rank-1 0.93 (trained 0.26-0.30), synthetic family p@5 0.217 (trained 0.627). Frozen DINOv2-Small L4+L8 PCA-64 reads museum 0.545, within-AUC 0.594, own-source 1.00; its CLS++mean 0.486 / 0.526 / 1.00; the 28 deterministic statistics alone 0.594 [0.481, 0.691] / 0.631 / 0.705 with a scanner probe of 0.56 (majority 0.44; the student 0.36-0.41). Cost on one CPU thread: MobileNetV3-Small 4.4 MB, 1.3 ms; DINOv2-Small 88 MB, 90 ms. Museum process p@5 therefore does not discriminate the student from untrained features; own-source rank-1 and the within-uploader AUC do. (high; analysis halves, one untrained control)
31. **Robustness profile of the pooled PCA-64 on real photographs it never trained on: invariant to crops and flips, nearly so to JPEG and rescale, and sensitive to the rendering edits it is meant to see.** On the 113 commoncatalog_ccby validation originals and their eight manifest-pipeline edits (a treatment family no model was trained on; E49), own-image rank-1 is 0.96-1.00 under an 80 % centre crop, 0.99-1.00 under a flip, 0.90-0.95 under JPEG q50, 0.89-0.94 under 0.5x rescale and 0.74-0.82 under a 70 % corner crop (three seeds); edit-type p@5 with the image's own family excluded is 0.46-0.48 against chance 0.11 (explicit statistics 0.53; DINOv2 L4+L8 0.24, CLS++mean 0.13, ImageNet-init 0.23), an edited image retrieves its own original 0.41-0.42 of the time (chance 0.009; the content encoders 0.99-1.00, the explicit statistics 0.54), and the film emulation displaces the embedding 7.6-7.8 times as far as the centre crop (monochrome emulation 10.5x, over-sharpening 0.5x, posterising 0.3-0.4x). The pre-registered 0.9 bound was missed by one seed's rescale reading (0.885; two seeds pass); JPEG and rescale sensitivity is the price of grain sensitivity (the explicit statistics fall to 0.45 under rescale). (medium-high; one validation set of 113, five media)
32. **The pooled PCA-64's neighbourhoods are rendering neighbourhoods: neighbours share colour-tone and texture statistics, and the embedding is not predictable from scene content.** On 389 Flickr film scans with the uploader excluded (E50, three seeds), the five nearest neighbours differ from the query in the standardised colour-tone statistics 0.65 times as much as five random candidates, in the texture statistics 0.85-0.87 times, all 28 together 0.74-0.75 (DINOv2 CLS++mean 0.95 / 0.85 / 0.90; ImageNet-init 0.86 / 0.83 / 0.84; DINOv2 L4+L8 0.87 / 0.79 / 0.84; the statistics themselves, the ceiling, 0.45 / 0.58 / 0.51); neighbours agree on the monochrome flag 0.95 of the time against a prior of 0.69 (content embedding 0.71). The uploader-grouped ridge R^2 of the embedding from a 28-d PCA of the content embedding is -0.02 (CLS++mean itself 0.39, L4+L8 0.15, ImageNet-init 0.11) and from the 28 statistics 0.01-0.02, so the embedding is neither a linear function of scene content nor of the deterministic statistics (the rule's R^2 comparison was satisfied trivially and is not evidence on its own); provenance one-hots -0.03. The museum half repeats the pattern (colour 0.68-0.70, texture 0.82-0.84). (medium-high; attribute coherence is a proxy, no human judgement of "look")
33. **Rank control on the student's pooled layer is closed: expanding treatment variance only still expands content.** With the VICReg term computed on features centred on their synthetic source's in-batch mean (batches led by 12 sources x 4 of their treated rows; E51, three seeds) the PCA-64 reads rank 22.6-23.5, museum 0.49-0.55, within-uploader AUC 0.60-0.64 and own-source rank-1 0.72-0.75 (raw layer 0.82-0.83), worse than the batch-wise term (0.61) and far from the plain recipe (0.28). Together with claims 25, 27 and 29, every rank-raising intervention tried on this backbone, on the frozen teacher or end to end, moved content and film information together; the E43 teacher-distilled pooled layer at PCA 32-64 remains the strongest lightweight representation. (high; six seeds across two variants)
34. **Re-encoding half of the training crops makes the pooled PCA-64 robust to re-encoded uploads at a small, bounded cost in edit sensitivity.** With JPEG q40-70 or 0.5x rescale applied to half of every batch's crops (E52, three seeds), own-image rank-1 on the 113 validation originals is 0.98-1.00 under JPEG q50 and 0.99-1.00 under 0.5x rescale (reference 0.90-0.95 and 0.89-0.94), 0.96-0.99 under the centre crop, 1.00 under a flip and 0.84-0.88 under the 70 % corner crop (0.74-0.82); edit-type p@5 0.44-0.45 (reference 0.466), edited own-source rank-1 0.45-0.46 (0.41-0.42), film-emulation displacement 6.0-6.3x the crop (7.6-7.8x). On the analysis halves the PCA-64 keeps museum 0.49-0.51 with lower CIs above chance, within-uploader AUC 0.62-0.64, own-source 0.30-0.31, rank 8.8-9.0, synthetic family 0.60 (0.63). (medium-high; validation split of 113 and analysis halves, no held-out reading yet)

35. **On fresh real photographs the student's only statistically supported advantage over the untrained ImageNet backbone is within-photographer stock separation.** Met analysis half (378 images, 140 artists) and 324 film scans from 91 never-seen uploaders, three seeds, PCA-32 int8 (E53): within-uploader different-day AUC 0.672-0.715 against 0.560 for the same backbone untrained with the same PCA, paired seed-mean difference +0.128 [+0.039, +0.232] over 93 queries; against DINOv2-global +0.187 [+0.029, +0.360]; against the 28 measured statistics a tie (-0.013 [-0.083, +0.044]). Museum process p@5 is equal to the untrained backbone (+0.007) and below the statistics (-0.052 [-0.084, -0.022]); stock-line p@5 across uploaders is 0.14-0.16 for every representation against chance 0.08. (high; one pre-registered read)
36. **The pre-registered holdout rule was falsified by two guards, not by the claim.** The museum lower CI (0.642-0.648) did not clear a chance level of 0.648 set by an albumen-dominated half, and the 80 % centre crop on the 324-scan bank read 0.87-0.88 against the 0.9 bound (the untrained backbone 0.99). JPEG q50 and 0.5x rescale held at 0.95-0.99 on fresh scans where the E23b seeds lose 24-33 %. (high)
37. **Film-stock identification is not supported by any representation on fresh data.** Per stock, only Portra (0.32) and Tri-X (0.29), the two largest classes, retrieve above chance; nine stocks read 0.00; on ARTeFACT's owner-recorded emulsions the student (AUC 0.62) is below the untrained backbone (0.73) and DINOv2 (0.75). Stock and photographer matching stay exploratory similarity estimates. (high)
38. **Abstention raises label precision modestly and unstably.** The 3-of-5 agreement gate with a spent-half threshold gives museum process claims 0.754 [0.686, 0.824] at 39 % coverage (answer-everything 0.747) and stock claims 0.24-0.27 at 31-36 % coverage (0.16); one seed found no threshold at that gate (E55). (medium)
39. **The embedding is a holistic, monotone look signature, not a set of linear attribute axes.** Seven single-parameter edits move it monotonically (Spearman 0.77-0.98) and 1.2-10x a centre crop, but a scene-grouped ridge recovers only contrast (R^2 0.73) and grain (0.57) across scenes; the 28 statistics decode four of seven by construction, DINOv2 none (E54, falsified). (medium-high; 113 validation originals)
40. **The stock signal does not run through the scanner, and no identity is linearly readable.** Stock-line p@5 with the scanner family forced to differ (0.111-0.116) is not below the unrestricted value (0.092-0.097); ridge from stock, scanner or uploader one-hots explains none of the variance under CV for any representation; content PCs explain 4 % of the candidate against 13 % for the untrained backbone and 45 % for DINOv2-global (E56, spent analysis half). (medium)

41. **For rendering-family recognition the 28 measured statistics are the strongest single representation, learned embeddings add nothing to them, and no method identifies film-stock lines across photographers.** On the fresh Met half (4 process families, artist excluded, decade matched) plain 5-NN on the standardised statistics reaches 0.847 balanced accuracy; hybrids (statistics plus the student, the untrained backbone or DINOv2) 0.836-0.846; the student alone 0.677, ImageNet init 0.632, DINOv2 global 0.576; prototypes 0.71-0.75. On the fresh scans (23 stock lines, uploader excluded) the best classifier of any kind reaches 0.124 balanced accuracy against chance 0.043, with lower CIs at or near chance. The student remains the least photographer-dominated representation (uploader share 0.131 vs 0.181 statistics, 0.222 DINOv2) and ties the statistics within one photographer (AUC 0.678 vs 0.691). Vote-share confidences are over-confident for every method (ECE 0.17-0.43). (medium-high; E57, post-hoc on sets read once by E53)

## What is not established
- That any embedding captures aesthetics as a person would judge them (no human evaluation exists).
- That same-stock retrieval generalises beyond one photographer's 1990s consumer stocks.
- Composition: a frozen 2×2 patch-grid probe reaches R² ≈ 0.23 only; the spatial student failed (E12).
- Photographer style: textual profiles only; the content- and aspect-controlled same-uploader retrieval on Flickr film scans (0.10-0.14 vs prior 0.03) is not separable from film look (claim 25), so it is not a validated photographer-style feature.
