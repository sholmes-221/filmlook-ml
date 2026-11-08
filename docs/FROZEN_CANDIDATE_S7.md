# Frozen candidate, session 7 (2026-11-08)

Frozen by `scripts/freeze_candidate.py` at 2026-11-08T15:29:33 on commit 17bc109b460b, before any fresh-holdout image was read. The directory
`checkpoints/frozen_session7/` (git-ignored) holds byte copies of the ONNX students, their `meta.json`, and `pca.npz` (mean,
64 nested PCA components fitted on the student's pooled 576-d features of the 601 training-allowed CC-BY train originals
(path-list SHA-256 `b204be3eadd10641`), int8 scales for the first 32). PCA-32 int8 = first 32 components,
scale max|component|/127 on the fit set (E47). `scripts/e53_fresh_holdout.py` verifies these hashes before it runs.

| tag | role | model.onnx SHA-256 | pca.npz SHA-256 | size | CPU 1 thread (median) |
|---|---|---|---|---|---|
| `style_fs_v2_real_f576_deg` | candidate (E52 seed) | `46b9b4f1e36440ff` | `a4a4b21f03f06965` | 4.40 MB | 1.33 ms + PCA 0.0784 ms |
| `style_fs_v2_real_f576_deg_s1` | candidate (E52 seed) | `5c60908f233ebb2a` | `e7890f3c0fd47777` | 4.40 MB | 1.6 ms + PCA 0.103 ms |
| `style_fs_v2_real_f576_deg_s2` | candidate (E52 seed) | `5831ff6e821b2b6c` | `c1b6439d0500601a` | 4.40 MB | 1.6 ms + PCA 0.0727 ms |
| `style_fs_v2_real_f576` | comparator (E23b seed, read E44) | `534fa15bf3140928` | `1c031dec4072c041` | 4.40 MB | 1.61 ms + PCA 0.0797 ms |
| `style_fs_v2_real_f576_s1` | comparator (E23b seed, read E44) | `d2ca8cbd50a5a0a1` | `4b0380df51c2086c` | 4.40 MB | 1.84 ms + PCA 0.0716 ms |
| `style_fs_v2_real_f576_s2` | comparator (E23b seed, read E44) | `5d91bda6443d4e9a` | `779a8270ba908f45` | 4.40 MB | 1.57 ms + PCA 0.0744 ms |
| `imagenet_init` | untrained control | `748c81c57b9fe5c2` | `4bb81d997a227138` | 4.40 MB | 1.56 ms + PCA 0.0794 ms |

Primary reading: the seed-mean of the three candidate seeds' PCA-32 int8 (32 bytes per image). The full hashes are in
`checkpoints/frozen_session7/freeze.json`. Nothing here is deployed; the production FilmLook application is unchanged.
