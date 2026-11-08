#!/usr/bin/env python
"""FilmLook photographic-style similarity service (experimental MVP): CPU-only, ONNX Runtime, standard-library HTTP.

  FLS_MODEL_DIR=checkpoints/frozen_session7/style_fs_v2_real_f576_deg FLS_PORT=8765 python service/filmlook_style_service.py

Endpoints
  GET  /health            liveness: {"status": "ok", "model_loaded": bool, "inflight": n, ...}; never loads the model
  GET  /ready             readiness: loads the model lazily (first call) and returns 200 once it answers a warm-up input
  POST /embed[?explicit=0] body = image bytes (JPEG/PNG/WebP; <= FLS_MAX_BYTES, <= FLS_MAX_PIXELS) ->
                          {"embedding_int8": [32 ints], "scale": [32 floats], "explicit28": [28 floats], "model": sha, "ms": ..}
  POST /similar?k=5       same body -> the k nearest bank entries by cosine on the 32-d embedding, with the abstention rule:
                          "similar look" neighbours are always returned (exploratory); a label claim is attached only when
                          at least 3 of the top 5 bank labels agree AND the top-1 cosine >= FLS_TAU; otherwise "abstain".
Design for a 1 GB host (Oracle Always Free E2 Micro): one ORT intra-op thread, lazy model load guarded by a lock, bounded
in-flight requests (FLS_MAX_INFLIGHT, HTTP 503 when exceeded), request body and pixel limits, JPEG draft decoding, no
third-party web framework (ThreadingHTTPServer), graceful fallback to the deterministic 28 statistics when the network
fails (`"fallback": "explicit28"`), JSON errors for bad input. Nothing here identifies a film stock or a photographer; the
labels in a bank are whatever the bank's owner recorded, and a claim is only ever "looks like", gated as above.
The production FilmLook scoring and its LLM analysis are not touched; this service is meant to run beside them.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_infer.compact import preprocess  # noqa: E402
from filmlook_infer.features import FEATURE_NAMES, compute_features, feature_vector  # noqa: E402
from filmlook_infer.imageio import ImageValidationError, load_image  # noqa: E402

VERSION = "0.1.0-mvp"
CFG = {
    "model_dir": Path(os.environ.get("FLS_MODEL_DIR", ROOT / "checkpoints" / "frozen_session7" / "style_fs_v2_real_f576_deg")),
    "bank": os.environ.get("FLS_BANK", ""),                        # optional npz: ids, emb_int8 (n,32), scale (32), labels (n) or "", attribution (n)
    "port": int(os.environ.get("FLS_PORT", "8765")), "host": os.environ.get("FLS_HOST", "127.0.0.1"),
    "max_inflight": int(os.environ.get("FLS_MAX_INFLIGHT", "2")), "max_bytes": int(os.environ.get("FLS_MAX_BYTES", str(8 * 1024 * 1024))),
    "max_pixels": int(os.environ.get("FLS_MAX_PIXELS", str(30_000_000))), "threads": int(os.environ.get("FLS_THREADS", "1")),
    "tau": float(os.environ.get("FLS_TAU", "0.90")), "short_side": int(os.environ.get("FLS_SHORT_SIDE", "384")), "request_timeout": float(os.environ.get("FLS_TIMEOUT", "20")),
}


class Model:
    """Lazy ONNX session + PCA-32 int8 reduction; one instance per process."""
    def __init__(self, model_dir: Path, threads: int):
        self.dir, self.threads = Path(model_dir), threads; self.sess = None; self.lock = threading.Lock(); self.sha = None; self.load_ms = None; self.loaded_at = None

    def load(self):
        with self.lock:
            if self.sess is not None:
                return
            import onnxruntime as ort
            t = time.perf_counter(); so = ort.SessionOptions(); so.intra_op_num_threads = self.threads; so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.enable_mem_pattern = False; so.enable_cpu_mem_arena = False   # small, predictable RSS
            model = self.dir / "model.onnx"; self.sha = hashlib.sha256(model.read_bytes()).hexdigest()
            sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
            self.oi = json.loads((self.dir / "meta.json").read_text())["outputs"].index("features_576")
            z = np.load(self.dir / "pca.npz"); self.mu, self.V32, self.scale = z["mu"].astype(np.float32), z["V64"][:, :32].astype(np.float32), z["scale32"].astype(np.float32)
            sess.run(None, {"pixel_values": np.zeros((1, 3, 224, 224), np.float32)})                 # warm-up
            self.sess = sess; self.load_ms = round(1000 * (time.perf_counter() - t), 1); self.loaded_at = time.time()

    def embed(self, im) -> tuple[np.ndarray, np.ndarray]:
        """PIL image (already size-limited) -> (int8 codes (32,), dequantised float (32,))."""
        if self.sess is None:
            self.load()
        f = self.sess.run(None, {"pixel_values": preprocess(im)})[self.oi][0]
        z = (f - self.mu) @ self.V32; q = np.clip(np.round(z / self.scale), -127, 127).astype(np.int8)
        return q, q.astype(np.float32) * self.scale


class Bank:
    def __init__(self, path: str):
        self.n = 0; self.ids = self.labels = self.attr = None; self.E = None
        if path and Path(path).exists():
            z = np.load(path, allow_pickle=True); emb = z["emb_int8"].astype(np.float32) * z["scale"].astype(np.float32)
            self.E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9); self.ids = z["ids"]; self.labels = z["labels"] if "labels" in z else None; self.attr = z["attribution"] if "attribution" in z else None; self.n = len(self.ids)

    def similar(self, v: np.ndarray, k: int, tau: float) -> dict:
        if self.n == 0:
            return {"neighbours": [], "label_claim": None, "decision": "no_bank"}
        u = v / (np.linalg.norm(v) + 1e-9); s = self.E @ u; order = np.argsort(-s)[:max(k, 5)]
        nb = [{"id": str(self.ids[i]), "cosine": round(float(s[i]), 4), **({"label": str(self.labels[i])} if self.labels is not None and str(self.labels[i]) else {}), **({"attribution": str(self.attr[i])} if self.attr is not None else {})} for i in order]
        claim, decision = None, "similar_look_only"
        if self.labels is not None:
            top5 = [str(self.labels[i]) for i in order[:5] if str(self.labels[i])]
            if top5:
                lab, cnt = max(((l, top5.count(l)) for l in set(top5)), key=lambda x: x[1])
                if cnt >= 3 and float(s[order[0]]) >= tau:
                    claim, decision = {"label": lab, "agreement": f"{cnt}/5", "top1_cosine": round(float(s[order[0]]), 4), "kind": "looks_like (exploratory, not an identification)"}, "label_claim"
                else:
                    decision = "abstain"
        return {"neighbours": nb[:k], "label_claim": claim, "decision": decision}


STATE = {"model": Model(CFG["model_dir"], CFG["threads"]), "bank": None, "inflight": 0, "lock": threading.Lock(), "started": time.time(), "requests": 0, "errors": 0, "fallbacks": 0, "rejected_busy": 0}
STATE["bank"] = Bank(CFG["bank"])


def read_image(body: bytes):
    im = load_image(io.BytesIO(body), max_side=2 * CFG["short_side"])
    if im.width * im.height > CFG["max_pixels"]:
        raise ImageValidationError("too many pixels")
    s = CFG["short_side"] / min(im.size)
    if s < 1:                                                                  # the evaluation pipeline's 384 px short side
        from PIL import Image
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    return im


def analyse(body: bytes, want_similar: bool, k: int, explicit: bool = True) -> tuple[int, dict]:
    t0 = time.perf_counter(); im = read_image(body); out = {"version": VERSION, "ms": None}
    if explicit:                                                               # the 28 deterministic statistics (about 100 ms on one core; ?explicit=0 skips them)
        try:
            ex = feature_vector(compute_features(im)); out["explicit28"] = [round(float(x), 5) for x in ex]; out["explicit28_names"] = FEATURE_NAMES
        except Exception as e:  # noqa: BLE001
            out["explicit28"] = None; out["explicit28_error"] = repr(e)[:120]
    try:
        q, v = STATE["model"].embed(im); out.update({"embedding_int8": q.tolist(), "scale": [round(float(x), 7) for x in STATE["model"].scale], "model": STATE["model"].sha, "fallback": None})
        if want_similar:
            out["similar"] = STATE["bank"].similar(v, k, CFG["tau"])
    except Exception as e:  # noqa: BLE001  graceful degradation: the deterministic statistics still describe the image
        STATE["fallbacks"] += 1; out.update({"embedding_int8": None, "fallback": "explicit28", "model_error": repr(e)[:160]})
        if want_similar:
            out["similar"] = {"neighbours": [], "label_claim": None, "decision": "model_unavailable"}
    out["ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return 200, out


class Handler(BaseHTTPRequestHandler):
    server_version = "FilmLookStyle/" + VERSION; protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):                                        # quiet by default; one line per request to stderr when FLS_LOG=1
        if os.environ.get("FLS_LOG"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, obj: dict):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path; m = STATE["model"]
        if path == "/health":
            return self._json(200, {"status": "ok", "version": VERSION, "model_loaded": m.sess is not None, "model_dir": str(m.dir), "bank_size": STATE["bank"].n, "inflight": STATE["inflight"], "max_inflight": CFG["max_inflight"],
                                    "uptime_s": round(time.time() - STATE["started"], 1), "requests": STATE["requests"], "errors": STATE["errors"], "fallbacks": STATE["fallbacks"], "rejected_busy": STATE["rejected_busy"]})
        if path == "/ready":
            try:
                m.load(); return self._json(200, {"ready": True, "model": m.sha, "load_ms": m.load_ms})
            except Exception as e:  # noqa: BLE001
                return self._json(503, {"ready": False, "error": repr(e)[:160]})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path); path = u.path
        if path not in ("/embed", "/similar"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > CFG["max_bytes"]:
            return self._json(413 if n > CFG["max_bytes"] else 400, {"error": f"body must be 1..{CFG['max_bytes']} bytes of image data"})
        with STATE["lock"]:
            if STATE["inflight"] >= CFG["max_inflight"]:
                STATE["rejected_busy"] += 1; self.send_response(503); self.send_header("Retry-After", "1"); self.send_header("Content-Length", "0"); self.end_headers(); self.rfile.read(n); return
            STATE["inflight"] += 1; STATE["requests"] += 1
        try:
            body = self.rfile.read(n); qs = parse_qs(u.query); k = max(1, min(20, int((qs.get("k") or ["5"])[0])))
            code, out = analyse(body, path == "/similar", k, (qs.get("explicit") or ["1"])[0] not in ("0", "false")); self._json(code, out)
        except ImageValidationError as e:
            STATE["errors"] += 1; self._json(400, {"error": f"invalid image: {e}"})
        except Exception as e:  # noqa: BLE001
            STATE["errors"] += 1; self._json(500, {"error": "internal error", "detail": repr(e)[:160]})
        finally:
            with STATE["lock"]:
                STATE["inflight"] -= 1


def main():
    srv = ThreadingHTTPServer((CFG["host"], CFG["port"]), Handler); srv.daemon_threads = True; srv.timeout = CFG["request_timeout"]
    sys.stderr.write(f"filmlook style service {VERSION} on {CFG['host']}:{CFG['port']} model={CFG['model_dir']} bank={CFG['bank'] or '-'} max_inflight={CFG['max_inflight']} (model loads on first request or /ready)\n")
    if os.environ.get("FLS_PRELOAD"):
        STATE["model"].load()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
