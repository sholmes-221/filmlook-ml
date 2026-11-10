"""Standalone film-style matching engine (NumPy / SciPy / OpenCV only; no torch, no ONNX at inference).

    from research.lut.engine import FilmStyleEngine
    engine = FilmStyleEngine()                      # loads research/lut/db (precomputed LUT feature database)
    result = engine.analyze("photo.jpg")            # dict: top-3 styles with similarity 0-100, explanations, descriptors

Modules: `lut3d` (3D LUT container, .cube I/O, trilinear application), `descriptors` (photo descriptors),
`lut_db` (LUT feature database builder + loader), `matching` (approaches A-E), `engine` (public API).
Every style is a LUT-derived aesthetic (a Lightroom preset re-implemented approximately); none is a claim about the
film stock that exposed the photograph.
"""
from pathlib import Path

DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "db"

from research.lut.engine.engine import FilmStyleEngine, analyze_image  # noqa: E402, F401
