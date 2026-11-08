#!/usr/bin/env python
"""Session 7: freeze the recommended lightweight representation before the fresh holdouts are read.

  python scripts/freeze_candidate.py            # -> checkpoints/frozen_session7/{freeze.json, <tag>/...}; refuses to overwrite

Frozen artifacts (per tag): a byte copy of the ONNX student and its meta.json, and `pca.npz` holding the PCA fitted on the
student's pooled 576-d features of the 601 training-allowed CC-BY train originals (mean, the first 64 components, the
per-component int8 scales for the first 32). PCA-32 is the first 32 of the 64 components (nested), int8 quantised with
the scale max|value|/127 fitted on the same set (E47). `freeze.json` records the SHA-256 of every file, of the fit-set
path list, the git commit, the single-thread CPU latency of the student and of the reduction, and the artifact sizes.
Candidate: the three degrade-augmented seeds of E52 (`style_fs_v2_real_f576_deg{,_s1,_s2}`). Comparators frozen with
the same procedure: the E23b seeds (`style_fs_v2_real_f576{,_s1,_s2}`, held-out reading E44) and the untrained
ImageNet backbone (`imagenet_init`). The evaluation script (scripts/e53_fresh_holdout.py) loads only this directory.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import fit_embed  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

FROZEN = P.CHECKPOINTS / "frozen_session7"
CANDIDATE = ["style_fs_v2_real_f576_deg", "style_fs_v2_real_f576_deg_s1", "style_fs_v2_real_f576_deg_s2"]
COMPARATORS = ["style_fs_v2_real_f576", "style_fs_v2_real_f576_s1", "style_fs_v2_real_f576_s2", "imagenet_init"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Frozen:
    """Loads one frozen tag: ONNX session + PCA; `embed(arrays, k, int8)` returns the reduced embedding."""
    def __init__(self, tag: str, threads: int | None = None):
        import onnxruntime as ort
        d = FROZEN / tag; so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.oi = json.loads((d / "meta.json").read_text())["outputs"].index("features_576")
        z = np.load(d / "pca.npz"); self.mu, self.V, self.scale32 = z["mu"], z["V64"], z["scale32"]

    def features(self, arrays) -> np.ndarray:
        from PIL import Image
        from filmlook_infer.compact import preprocess
        acc = []
        for i in range(0, len(arrays), 64):
            acc.append(self.sess.run(None, {"pixel_values": np.concatenate([preprocess(Image.fromarray(x)) for x in arrays[i:i + 64]])})[self.oi])
        return np.concatenate(acc).astype(np.float32)

    def reduce(self, F: np.ndarray, k: int = 32, int8: bool = True) -> np.ndarray:
        Z = (F - self.mu) @ self.V[:, :k]
        if int8:
            assert k <= 32; s = self.scale32[:k]; Z = np.clip(np.round(Z / s), -127, 127).astype(np.int8).astype(np.float32) * s
        return Z


def latency(tag: str) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(FROZEN / tag / "model.onnx"), so, providers=["CPUExecutionProvider"]); x = np.zeros((1, 3, 224, 224), np.float32); sess.run(None, {"pixel_values": x})
    ts = []
    for _ in range(50):
        t = time.perf_counter(); sess.run(None, {"pixel_values": x}); ts.append(time.perf_counter() - t)
    z = np.load(FROZEN / tag / "pca.npz"); f = np.zeros((1, 576), np.float32); tp = []
    for _ in range(200):
        t = time.perf_counter(); ((f - z["mu"]) @ z["V64"][:, :32]); tp.append(time.perf_counter() - t)
    return {"student_ms_single_thread_median": round(1000 * float(np.median(ts)), 2), "pca32_ms_median": round(1000 * float(np.median(tp)), 4)}


def main():
    if (FROZEN / "freeze.json").exists():
        raise SystemExit(f"{FROZEN / 'freeze.json'} exists: the candidate is already frozen; it is not re-frozen")
    t0 = time.time(); FROZEN.mkdir(parents=True, exist_ok=True)
    fit_rows = [r for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    fit_paths = [r["path"] for r in fit_rows]; fit_sha = hashlib.sha256("\n".join(sorted(fit_paths)).encode()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=P.ROOT if hasattr(P, "ROOT") else Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    out = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit, "candidate": CANDIDATE, "comparators": COMPARATORS, "primary": "pca32_int8 of the candidate seeds (seed-mean reported)",
           "fit_set": {"dataset": "commoncatalog_ccby", "split": "train", "edit_type": "original", "n": len(fit_paths), "paths_sha256": fit_sha},
           "reduction": "PCA fitted on the student's pooled features of the fit set; PCA-32 = first 32 of 64 nested components; int8 scale = max|component|/127 on the fit set", "tags": {}}
    for tag in CANDIDATE + COMPARATORS:
        src = P.CHECKPOINTS / f"student_{tag}"; dst = FROZEN / tag; dst.mkdir(exist_ok=True)
        shutil.copyfile(src / "model.onnx", dst / "model.onnx"); shutil.copyfile(src / "meta.json", dst / "meta.json")
        Zfit = fit_embed(tag, fit_paths); pca = PCA64(Zfit, 64); Z32 = pca(Zfit)[:, :32]; scale32 = (np.abs(Z32).max(0) / 127.0 + 1e-9).astype(np.float32)
        np.savez(dst / "pca.npz", mu=pca.mu.astype(np.float32), V64=pca.V.astype(np.float32), scale32=scale32)
        meta = json.loads((src / "meta.json").read_text())
        out["tags"][tag] = {"source_checkpoint": str(src), "model_sha256": sha(dst / "model.onnx"), "meta_sha256": sha(dst / "meta.json"), "pca_sha256": sha(dst / "pca.npz"),
                            "model_bytes": (dst / "model.onnx").stat().st_size, "pca_bytes": (dst / "pca.npz").stat().st_size, "pca32_int8_vector_bytes": 32,
                            "recipe": {k: meta.get(k) for k in ("kd_target", "kd_weight", "vic_pooled", "vic_mode", "aug_degrade", "seed", "synth", "train_generators", "real_distil")}, "latency": latency(tag)}
        print(tag, out["tags"][tag]["model_sha256"][:12], out["tags"][tag]["latency"], flush=True)
    for d in FROZEN.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                f.chmod(0o444)
    out["seconds"] = round(time.time() - t0, 1); (FROZEN / "freeze.json").write_text(json.dumps(out, indent=1)); (FROZEN / "freeze.json").chmod(0o444)
    print(json.dumps({k: v for k, v in out.items() if k != "tags"}))


if __name__ == "__main__":
    main()
