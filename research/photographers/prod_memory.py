"""C19b: peak memory of the website's real analysis path, with and without the SigLIP int8 session.

    OMP_NUM_THREADS=1 ~/misc/film-evaluation/.venv/bin/python -B research/photographers/prod_memory.py [--siglip] [--out x.json]

Runs in the WEBSITE's virtual environment (its production dependencies) and imports its code read-only; nothing is
written inside film-evaluation (bytecode off; the DNA store goes to a temporary directory; no .env is read; no
network). One fresh process per reading. Path, as wired in filmlook/api/app.py and photographer_dna/service.py with
ANALYSIS_BACKEND=filmlook_ml, photo gate on, reference matching on, local vision on:
  prepare_image -> photo gate judge -> local vision signals -> DnaService.analyze (measured matching + reference
  retrieval through photo_gate.embedding) -> LocalFilmLookAnalyzer.analyze (measurements, simulations, local_report with
  the film head on photo_gate.embedding_layers).
With --siglip the SigLIP-B/16 int8 session (external, memory-mapped weights; lean options; 1 thread) is created
before the analyses and the prepared JPEG is embedded with it, as an integration would.
Reported: ru_maxrss (peak RSS of the process, MiB), RSS after imports, RSS at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

WEB = Path.home() / "misc" / "film-evaluation"
ML = Path(__file__).resolve().parents[2]
FIXTURE = WEB / "tests" / "fixtures" / "cc0_portrait_curly_man.jpg"
SIGLIP = ML / "checkpoints" / "c19" / "siglip_base_int8_ext.onnx"


def mib_rss() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def peak() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def siglip_pixels(img):
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    return ((a - 0.5) / 0.5).transpose(2, 0, 1)[None].copy()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--siglip", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(WEB))
    t0 = time.perf_counter()
    from filmlook import local_film, local_vision
    from filmlook import photo_gate as pg
    from filmlook.analyzer import LocalFilmLookAnalyzer
    from filmlook.config import Settings
    from filmlook.image_utils import prepare_image
    from filmlook.photographer_dna.service import create_dna_service
    from filmlook.photographer_dna.storage import DnaStore
    import io

    import onnxruntime as ort
    from PIL import Image
    out = {"siglip": a.siglip, "rss_after_imports_mib": round(mib_rss(), 1), "peak_after_imports_mib": round(peak(), 1)}
    tmp = Path(tempfile.mkdtemp(prefix="c19mem-"))
    settings = Settings(mode="local", analysis_backend="filmlook_ml", photo_gate="on", reference_match=True, local_vision=True,
                        results_dir=tmp, require_login=False)
    local_film.set_embedding_provider(lambda jpeg: (lambda gate: gate.embedding_layers(jpeg) if gate is not None else None)(pg.gate_for(settings.photo_gate_dir)))
    sig = None
    if a.siglip:
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False; so.add_session_config_entry("session.disable_prepacking", "1")
        sig = ort.InferenceSession(str(SIGLIP), so, providers=["CPUExecutionProvider"])
    raw = FIXTURE.read_bytes()
    image = prepare_image(raw, settings.max_long_edge, settings.jpeg_quality)
    gate = pg.gate_for(settings.photo_gate_dir)
    verdict = gate.judge(image.data)
    heads = local_vision.heads_for(settings.local_vision_dir)
    signals = sorted(local_vision.signals(gate.embedding(image.data), heads)) if heads else []
    if sig is not None:
        v = sig.run(None, {"pixel_values": siglip_pixels(Image.open(io.BytesIO(image.data)))})[0]
        out["siglip_dim"] = int(v.shape[1])
    dna = create_dna_service(settings, DnaStore(settings.db_path)).analyze(raw, FIXTURE.name, force=True, language="en", track=False)
    film = LocalFilmLookAnalyzer().analyze(image, "en")
    out.update({"gate": verdict.outcome, "local_vision_signals": len(signals), "dna_status": dna.status,
                "dna_reference": (dna.result.reference or {}).get("status") if dna.result else None,
                "film_score": film.analysis.film_look_score, "film_matches": len(film.analysis.simulation_matches or []),
                "peak_rss_mib": round(peak(), 1), "rss_end_mib": round(mib_rss(), 1), "seconds": round(time.perf_counter() - t0, 2),
                "ort": ort.__version__, "omp_threads": os.getenv("OMP_NUM_THREADS")})
    print(json.dumps(out))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
