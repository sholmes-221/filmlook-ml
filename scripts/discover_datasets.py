#!/usr/bin/env python
"""Search the HF Hub for candidate datasets. Never enables training."""
import argparse
import json

from filmlook_ml import paths as P
from filmlook_ml.datasets.discovery import discover_hf
from filmlook_ml.datasets.registry import Registry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-query", type=int, default=15)
    a = ap.parse_args()
    cands = discover_hf(limit_per_query=a.limit_per_query)
    out = P.MANIFESTS / "discovered_datasets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cands, indent=1))
    reg = Registry.load()
    print(f"{len(cands)} candidates -> {out}")
    print("Registered & approved for training:", [d for d in reg.datasets if reg.dataset_allowed(d)])
    for c in cands[:20]:
        if "id" in c:
            print(f"  {c['id']:<55} license={c['license']:<22} commercial_compatible={c['license_commercial_compatible']}")


if __name__ == "__main__":
    main()
