"""Lightweight autonomous-research orchestration: registry, runner with resource ledger, evaluator, planner.

Single-process, file-based; no LLM calls, no cloud. State lives in research/registry.jsonl, research/ledger.json
and research/backlog.yaml (tracked in git) plus per-experiment logs under reports/experiments/ (git-ignored).
"""
