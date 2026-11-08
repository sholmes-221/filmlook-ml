#!/usr/bin/env python
"""E51: can the pooled layer's rank be raised by expanding treatment variance only (source-centred VICReg)?

  python scripts/e51_source_vicreg.py   # retrains three students (~1 GPU-minute each) -> reports/e51_source_vicreg.json

E46: a VICReg term on the pooled layer raises its effective rank to ~22 and its real-photograph metrics, but re-admits
scene content (own-source rank-1 0.61 vs 0.28), the trade-off of claim 25 reproduced end to end. Here the same term is
computed on features centred on their synthetic source's in-batch mean (`--vic-mode source`: each batch is led by 12
sources x 4 of their rows), so the variance it expands and decorrelates is the variance *between treatments of the same
photograph*, never between photographs. Same recipe otherwise (E23b, `--kd 0`, weight 1), three seeds, judged on the
real-photograph matrix next to the plain no-teacher seeds and the E43 teacher-distilled reference (both from E46).
Pre-registered rule (research/backlog.yaml E51): the E46 rule: for every seed the PCA-64 has effective rank >= 8,
own-source rank-1 <= 0.45, museum p@5 lower CI above chance, within-uploader different-day AUC lower CI above 0.5 and
synthetic family p@5 within 0.10 of the E43 reference seed-mean; VICReg seed-mean museum p@5 not more than 0.02 below
the plain no-teacher seed-mean. No test half, no BlueNeg.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import e46_pooled_vicreg as E46  # noqa: E402

VIC = {"style_fs_v2_real_svic1": 0, "style_fs_v2_real_svic1_s1": 1, "style_fs_v2_real_svic1_s2": 2}

if __name__ == "__main__":
    E46.main(VIC=VIC, extra_args=("--vic-mode", "source"), name="E51_source_vicreg", out="e51_source_vicreg.json")
