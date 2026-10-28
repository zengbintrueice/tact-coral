#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-anchored residual head — three-domain PFV evaluation (orchestrator).

Runs the two PROCESS-SEPARATED phases so netCDF4/mmap cleanup in the inference
process cannot crash the analysis step:

  Phase 1  compute_physics_anchored_infer.py   — loads data+models, PFV
             inference, ocean-masked record sets -> physics_anchored_records.npz
  Phase 2  compute_physics_anchored_analysis.py — reads the npz (no netCDF),
             metrics + paired week-block bootstrap -> eval.json/.csv/.md

Outputs -> results/physics_anchored/
  physics_anchored_records.npz, physics_anchored_infer_meta.json
  physics_anchored_eval.{json,csv,md}

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_physics_anchored_eval.py
"""
import os
import sys
import io
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(script):
    print(f"\n=== {script} ===", flush=True)
    rc = subprocess.call([PY, str(HERE / script)], cwd=str(HERE))
    if rc != 0:
        print(f"[FAIL] {script} exited {rc}", flush=True)
        raise SystemExit(rc)
    print(f"[OK] {script}", flush=True)


if __name__ == "__main__":
    run("compute_physics_anchored_infer.py")
    run("compute_physics_anchored_analysis.py")
    print("\nEVALUATION COMPLETE", flush=True)
