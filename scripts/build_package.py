#!/usr/bin/env python
"""Build the deployable inference package (no training code, no torch):

  dist/filmlook-infer-<version>/
    filmlook_infer/            runtime code (+ schema/filmlook_output_v1.json)
    model/                     model.onnx or model.int8.onnx + meta.json
    requirements.txt           pinned CPU runtime deps
    pyproject.toml             `pip install .` works
    examples/                  sample outputs (en, zh) produced by THIS package
    README.md                  integration notes
    TRAINING_DATA_ATTRIBUTION.jsonl   CC-BY credits for photos used in training

  python scripts/build_package.py --ckpt checkpoints/best [--verify]
"""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from filmlook_ml import paths as P

REQS = ["numpy==2.5.4", "pillow==13.0.0", "scipy==1.18.1", "onnxruntime==1.31.1"]

PYPROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "filmlook-infer"
version = "{version}"
requires-python = ">=3.10"
dependencies = {deps}

[tool.setuptools]
packages = ["filmlook_infer"]

[tool.setuptools.package-data]
filmlook_infer = ["schema/*.json"]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(P.CHECKPOINTS / "best"))
    ap.add_argument("--int8", action="store_true", help="ship model.int8.onnx instead of model.onnx")
    ap.add_argument("--verify", action="store_true", help="install into a fresh venv and run the CLI")
    a = ap.parse_args()
    ck = Path(a.ckpt)
    meta = json.loads((ck / "meta.json").read_text())
    ver = meta["model_version"].removeprefix("filmlook-ml-v")
    out = P.ROOT / "dist" / f"filmlook-infer-{ver}"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(P.ROOT / "src" / "filmlook_infer", out / "filmlook_infer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (out / "model").mkdir(parents=True)
    onnx_name = "model.int8.onnx" if a.int8 else "model.onnx"
    shutil.copy(ck / onnx_name, out / "model" / onnx_name)
    (out / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "requirements.txt").write_text("\n".join(REQS) + "\n")
    (out / "pyproject.toml").write_text(PYPROJECT.format(version=ver, deps=json.dumps(REQS)))
    shutil.copy(P.ROOT / "deployment" / "INTEGRATION.md", out / "README.md")
    attr = P.RAW / "commoncatalog" / "ATTRIBUTION.jsonl"
    if attr.exists():
        shutil.copy(attr, out / "TRAINING_DATA_ATTRIBUTION.jsonl")
    # sample outputs produced by the packaged code itself (subprocess, isolated from training code)
    ex = out / "examples"
    ex.mkdir()
    sample = next(iter(sorted((P.ROOT / "tests" / "assets").glob("*.jpg"))), None)
    if sample:
        shutil.copy(sample, ex / "sample.jpg")
        for lang in ("en", "zh"):
            r = subprocess.run([sys.executable, "-I", "-m", "filmlook_infer.cli", str(ex / "sample.jpg"), "--lang", lang,
                                "--model", str(out / "model")], cwd=out, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(r.stderr)
            (ex / f"sample_output_{lang}.json").write_text(
                json.dumps(json.loads(r.stdout), indent=2, ensure_ascii=False) + "\n")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    tar = out.parent / f"{out.name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=out.name)
    print(f"package -> {out} ({size / 1e6:.1f} MB unpacked, {tar.stat().st_size / 1e6:.1f} MB tar.gz)")
    if a.verify:
        venv = P.ROOT / "dist" / ".verify-venv"
        if venv.exists():
            shutil.rmtree(venv)
        subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(venv)], check=True)
        py = venv / "bin" / "python"
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(py), str(out)], check=True)
        r = subprocess.run([str(py), "-I", "-c",
                            "import sys, json, subprocess; import filmlook_infer; "
                            "assert 'torch' not in sys.modules; print(filmlook_infer.MODEL_VERSION)"],
                           capture_output=True, text=True, cwd="/")
        print("fresh-venv import:", r.stdout.strip() or r.stderr)
        r = subprocess.run([str(py), "-I", "-m", "filmlook_infer.cli", str(ex / "sample.jpg"), "--lang", "zh",
                            "--model", str(out / "model")], capture_output=True, text=True, cwd="/")
        print("fresh-venv CLI exit", r.returncode, r.stdout[:300])
        pkgs = subprocess.run(["uv", "pip", "list", "--python", str(py)], capture_output=True, text=True).stdout
        print(pkgs)
        site = subprocess.run([str(py), "-c", "import site;print(site.getsitepackages()[0])"], capture_output=True,
                              text=True).stdout.strip()
        du = subprocess.run(["du", "-sm", site], capture_output=True, text=True).stdout.split()[0]
        print(f"fresh-venv site-packages: {du} MB")


if __name__ == "__main__":
    main()
