#!/usr/bin/env python
"""Launch the local Streamlit label-review app (binds to localhost only)."""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--reviewer", required=True)
ap.add_argument("--port", type=int, default=8501)
a = ap.parse_args()
app = Path(__file__).resolve().parents[1] / "src/filmlook_ml/labeling/review_app.py"
sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "127.0.0.1",
                          "--server.port", str(a.port), "--", "--reviewer", a.reviewer]))
