#!/usr/bin/env python
# =============================================================================
# check_runcard.py -- 10-second pre-flight check BEFORE starting a training
# round. Run this FIRST in Colab (after cell-2/cell-3):
#     !python scripts/check_runcard.py
# It verifies every name/config/dependency the round's run card depends on,
# so a missing file or a typo in EXP_NAME is caught here instead of two
# hours into a training run.
# =============================================================================
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


print("== 1. experiment-name <-> config consistency ==")
# Parse EXP_CONFIGS from the notebook (single source of truth for names).
import json
nb = json.load(open("notebooks/colab_train.ipynb", encoding="utf-8"))
exp_block = ""
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if src.startswith("# 5)"):
        exp_block = src
        break
names = {}
for line in exp_block.splitlines():
    if "'" in line and ".yaml" in line and ":" in line:
        try:
            k = line.split("'")[1]
            v = line.split("'")[3]
            names[k] = v
        except IndexError:
            pass
check(f"EXP_CONFIGS parsed ({len(names)} experiments)", len(names) >= 24,
      f"only {len(names)} found -- notebook is outdated, re-upload the new zip")

missing_cfg = {k: v for k, v in names.items() if not os.path.isfile(v)}
check("every EXP_CONFIGS name has its yaml on disk", not missing_cfg,
      f"missing: {missing_cfg}")
import os.path as _op
yaml_orphans = [f for f in sorted(os.listdir("configs")) if f.endswith(".yaml")
                and f not in {_op.basename(v) for v in names.values()}]
if yaml_orphans:
    print(f"  [note] {len(yaml_orphans)} yaml(s) not wired into EXP_CONFIGS "
          f"(informational): {yaml_orphans}")

print("== 2. duplicate out_dir (checkpoint overwrite risk) ==")
try:
    import yaml
    outdirs = {}
    for k, v in names.items():
        cfg = yaml.safe_load(open(v, encoding="utf-8"))
        outdirs.setdefault(cfg.get("train", {}).get("out_dir", "?"), []).append(k)
    dups = {d: ks for d, ks in outdirs.items() if len(ks) > 1}
    check("no two experiments share one out_dir", not dups, f"duplicates: {dups}")
except Exception as e:
    check("out_dir scan", False, str(e))

print("== 3. scripts & notebook dependencies ==")
for f in ["scripts/build_pranet_split.py", "scripts/eval.py",
          "scripts/benchmark_speed.py", "scripts/benchmark_competitors.py",
          "scripts/benchmark_one.py", "scripts/make_figures.py",
          "scripts/train.py", "src/model.py", "src/fanet.py"]:
    check(f, os.path.isfile(f))
try:
    import timm, thop, yaml, cv2  # noqa
    from py_sod_metrics import Smeasure  # noqa
    check("python deps (timm/thop/yaml/cv2/py_sod_metrics)", True)
except Exception as e:
    check("python deps", False, f"{type(e).__name__}: {e} -- re-run the pip cell")

print("== 4. Drive checkpoint conflicts (run BEFORE each training) ==")
drive_root = "/content/drive/MyDrive/polyp_paper"
if os.path.isdir(drive_root):
    stale = []
    for k in names:
        ck = f"{k}_ckpt"
        if os.path.isdir(os.path.join(drive_root, ck)):
            stale.append(ck)
    keep = {"exp8_combo_ckpt"}  # R8 seed-42 result -- DO NOT DELETE
    risky = [s for s in stale if s not in keep]
    check("no conflicting Drive checkpoints for this round", not risky,
          f"delete these first: {risky} (keep {sorted(set(stale) & keep)})")
else:
    print("  [note] Drive not mounted or polyp_paper folder missing -- "
          "run the mount cell first.")

print()
if FAIL:
    print(f"RESULT: {len(FAIL)} problem(s) found: {FAIL}")
    sys.exit(1)
print("RESULT: all checks passed -- safe to start the round.")
