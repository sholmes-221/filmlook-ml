# Photography style, aesthetics and retrieval: literature and implementation review (2026-11-08)

Sources were read on 2026-11-08 (arXiv abstracts and HTML, GitHub READMEs, the Google Research tree). Where a
page could not be read it is said so. Each row states which *task* the method addresses, because the briefs
mix four different problems: image **style** recognition, generic **aesthetic** quality, technical **image
quality**, and **film-stock identity**. None of the public methods below addresses film-stock identity.

| method | task | model and size | weights | licence (code / weights) | training data | runtime | relevance to FilmLook |
|---|---|---|---|---|---|---|---|
| Karayev et al., *Recognizing Image Style*, BMVC 2014 (arXiv 1311.3715) | photographic **style** (20 Flickr style labels: HDR, Macro, Noir, Long Exposure, …) and art genre (25 Wikipaintings labels) | one-vs-all linear classifiers on hand-crafted features (colour histograms, GIST, …) and DeCAF (AlexNet fc6/fc7) features | the paper links predictors and data at the author's site; not fetched here | not stated in the paper | Flickr Style 80k (built from Flickr groups; licence not stated), Wikipaintings 85k, AVA | Caffe-era; trivially re-implementable on any embedding | **Closest precedent**: it shows that style labels of this kind are learnable from generic features plus colour statistics with linear heads, which is exactly the hybrid-plus-kNN result below. Its labels are *style communities*, not film stocks. |
| DINOv2 (Oquab et al. 2023; facebookresearch/dinov2) | self-supervised generic visual features | ViT-S/14 21 M params (also B 86 M, L 300 M, g 1.1 B) | yes, dl.fbaipublicfiles.com and torch.hub | Apache-2.0 (code and main weights); XRay/Cell variants non-commercial | 142 M unlabelled images (LVD-142M; licence of the images not stated) | PyTorch; ViT-S/14 is 88 MB and ~90 ms per image on one CPU thread here, ~400 MB RSS: **not deployable on the 1 GB instance** | Already used as the frozen teacher. On FilmLook's fresh data its global embedding is the most *content*-dominated (uploader share 0.22, within-uploader stock AUC 0.49, E57). Fine offline for distillation; wrong for style retrieval on its own. |
| LAION aesthetic predictor v1 (LAION-AI/aesthetic-predictor) | generic **aesthetic** preference score | one linear layer (768→1) on CLIP ViT-L/14 or ViT-B/32 embeddings | yes (.pth in repo) | MIT (code); weights not separately licensed | not stated in the README (LAION/AVA-style preference ratings) | needs CLIP ViT-L/14 (~430 M params) or ViT-B/32 | Measures "how much people like the image", not film-likeness; a Velvia landscape and a clean digital landscape can score alike. Useful only as a *diagnostic* that FilmLook's score is not an aesthetics score. CLIP-L is too heavy for production. |
| Improved aesthetic predictor (christophschuhmann) | generic aesthetic score | MLP on CLIP ViT-L/14 embeddings | yes (sac+logos+ava1, ava+logos) | Apache-2.0 (code); weights not separately licensed | AVA, SAC, "logos" (SAC/logos licences not described) | CLIP ViT-L/14 | Same task caveat as above; same deployment problem. |
| clip-retrieval / LAION model-retrieval | CBIR tooling: CLIP embeddings → faiss index → kNN service | no model of its own (CLIP ViT-B/32 default) | n/a | MIT (clip-retrieval); the fork's licence not extracted | n/a | GPU-oriented figures (1500 img/s on a 3080; 50 ms back-end latency with memmapped faiss) | Architecture reference only: embeddings + memory-mapped index + kNN is the pattern FilmLook already uses in `service/filmlook_style_service.py`, at a scale where brute-force NumPy beats faiss (see oracle estimates). |
| SAAN, CVPR 2023 (BAID dataset) | **artistic** image aesthetics | n/a | n/a | n/a | BAID | n/a | The CVF page returned HTTP 403; nothing verified. Artistic aesthetics is again a preference task, not style identity. |
| VILA, CVPR 2023 (Ke et al.; google-research/vila) | aesthetics from user comments; zero-shot style classification; aesthetic captioning | CoCa-style image-text encoder-decoder (size not stated) | yes (GCS bucket, TF Hub) | not stated in the fetched README | AVA-Captions, AVA, LAION-pretrained CoCa | TensorFlow/JAX, large | Its captioning is the kind of LLM-adjacent text generation FilmLook must not depend on in production; its zero-shot style classification is interesting but the model is far too large for the target. |
| IQA-PyTorch (pyiqa) | technical IQA and some IAA (NIMA retrained) | many | yes (HF) | **PolyForm Noncommercial 1.0.0** + NTU S-Lab | AVA etc. (academic use) | PyTorch | **Not usable commercially.** Also the wrong task (quality, not style). |

Newer work (2024 to 2026) was not surveyed with a tool in this session; nothing is claimed about it.

## What the review settles

1. **No public method identifies film stock.** Every "aesthetic" model predicts preference; every "style" model
   predicts community or genre labels. FilmLook's own fresh-data readings (E53, E57) agree: film-stock lines are
   not separable across photographers by any representation tried (balanced accuracy 0.03 to 0.12 against chance
   0.043 over 23 lines). Stock *identity* must therefore stay out of the product's claims; *resemblance to a
   defined look* remains the honest framing.
2. **Style recognition in the Karayev sense works with linear heads on generic features plus colour statistics.**
   FilmLook's process-family task (albumen, gelatin silver, chromogenic, other; 4 classes, decade-matched,
   artist excluded) reaches 0.85 balanced accuracy with plain 5-NN on the 28 statistics and 0.84 to 0.85 with
   hybrids, versus 0.58 to 0.69 for learned embeddings alone (E57). The handcrafted descriptor is not the weak
   baseline; it is the strongest single representation for rendering-family retrieval.
3. **Generic semantic embeddings (DINOv2, CLIP) are content embeddings.** They are the right offline teacher for
   distillation and the wrong production feature for style retrieval; they are also too large for the target.
4. **Aesthetic predictors must not be used as film-look ground truth.** A linear head on CLIP can be reproduced
   in an hour, but it answers a different question, and its backbone cannot run on the instance.
5. **Licences**: DINOv2 Apache-2.0 is usable; the two LAION heads are MIT/Apache code with unlicensed weights;
   IQA-PyTorch is non-commercial; Flickr Style and AVA data licences are unverified. FilmLook's own data gate
   (`configs/datasets.yaml` and the per-record licence allowlist) stays the only admission rule.

## Reusable components, decided

- From the literature: the *architecture pattern* (embedding + colour/texture statistics → linear or kNN head),
  the *evaluation discipline* (grouped splits, class-balanced accuracy, calibration), and DINOv2 as an offline
  teacher only.
- From FilmLook's repository: the 28 statistics (`filmlook_infer` explicit features), the frozen MobileNetV3
  student for robustness to re-encoding, `service/filmlook_style_service.py` and `service/build_bank.py` as the
  serving skeleton, the fresh evaluation sets and their grouped metrics.
- Not reused: CLIP-based aesthetic heads (task mismatch, size), pyiqa (licence), VILA (size, text generation).
