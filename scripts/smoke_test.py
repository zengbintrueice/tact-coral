#!/usr/bin/env python3
"""LTTS smoke test: imports, model load, forward pass, metric utils."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 1. infra import
from paths import A6_WEIGHT, DATA_DIR, XGB_WEIGHT_DIR
print(f"paths OK: A6_WEIGHT exists={A6_WEIGHT.exists()}, XGB dir exists={XGB_WEIGHT_DIR.exists()}")
print(f"DATA_DIR={DATA_DIR} exists={DATA_DIR.exists()}")

# 2. src package
from src.data.aligned_dataset import SCSPIECFDataset
from src.models import PIECFModel
print("src package import OK")

# 3. model definition + weight load
import importlib, torch
mod = importlib.import_module("ablate_a_dh_stopgrad")
model = mod.FinalSingleHeadModel(dynamic_channels=13)
sd = torch.load(A6_WEIGHT, map_location="cpu", weights_only=False)
model.load_state_dict(sd)
model.eval()
print(f"a6 model load OK ({sum(p.numel() for p in model.parameters())} params)")

# 4. forward smoke test
xd = torch.randn(2, 24, 21, 23, 13)
xs = torch.randn(2, 21, 23, 5)
lm = torch.zeros(21, 23, dtype=torch.bool)
out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
print(f"forward OK: y_final {tuple(out[0].shape)}, y_main {tuple(out[1].shape)}")

# 5. pfv_utils
from pfv_utils import build_pfv_arrays
print("pfv_utils import OK")

# 6. XGBoost weights loadable
import joblib
m = joblib.load(XGB_WEIGHT_DIR / "xgb_lead4.joblib")
print(f"XGB lead4 load OK ({type(m).__name__})")

# 7. key result JSONs intact
import json
with open(Path(__file__).resolve().parents[1] / "results" / "scs_benchmark" / "scs_threshold_calibration.json") as f:
    tc = json.load(f)
print(f"threshold_calibration.json OK: a6 best thr={tc['a6']['best_csi']['threshold']}, "
      f"test CSI={tc['a6']['best_csi']['test']['csi']:.4f}")

print("\nALL SMOKE TESTS PASSED")

# ---- evidence-lock verification (--locks) ----
def check_locks():
    import hashlib as _h, re as _re
    locks_md = Path(__file__).resolve().parents[1] / "EVIDENCE_LOCKS.md"
    if not locks_md.exists():
        print("EVIDENCE_LOCKS.md missing - skip")
        return True
    root = locks_md.parent
    ok = True
    for m in _re.finditer(r"`([^`]+)`\s+MD5\s+`([0-9a-f]{32})`", locks_md.read_text(encoding="utf-8")):
        rel, want = m.group(1), m.group(2)
        p = root / rel
        if not p.exists():
            print(f"LOCK MISSING  {rel}")
            ok = False
            continue
        got = _h.md5(p.read_bytes()).hexdigest()
        if got != want:
            print(f"LOCK FAIL    {rel}  got {got}")
            ok = False
        else:
            print(f"LOCK OK      {rel}")
    print("locks:", "ALL OK" if ok else "FAILURES PRESENT")
    return ok

if "--locks" in sys.argv:
    check_locks()
