"""LUT / Lightroom-preset film-style research (2026-11-10).

Question: can film-simulation presets identify the film colour style of a user photograph?
Code: `xmp_parser` (Adobe Camera Raw XMP -> parameters), `forward_model` (documented numpy approximation of the
Lightroom rendering of those parameters), `inverse_analysis` (monotonicity, Jacobian, grid inverse, ambiguity),
`features` (CIELAB statistics, histograms, DINOv2 ONNX), `synthetic` (preset-labelled renders of licensed digital
photographs), `baselines`, `train`, `evaluate`, `real_eval`, `robustness`, `bench`. Report: `REPORT.md`.

The owner's XMP files stay in /home/shanmo/misc/lut and are never copied into the repository; parsed numeric tables
go to data/lut/ (git-ignored).
"""
