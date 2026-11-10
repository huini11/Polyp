#!/usr/bin/env python
# =============================================================================
# benchmark_one.py -- benchmark ONE competitor architecture in an isolated
# subprocess (each vendor repo has its own `lib` package; a single process
# cannot import three different `lib` packages simultaneously).
#
# Called by benchmark_competitors.py with --method {pranet|polyp-pvt|hardnet}.
# Latency/FPS depend only on architecture and input size, NOT on weight
# values, so the benchmark runs with random weights; the paper states this
# explicitly ("architecture-level re-benchmark, latency is weight-
# independent"). The measurement protocol is identical to
# scripts/benchmark_speed.py: torch.cuda.synchronize before/after every
# timed frame, warmup iterations excluded, mean +/- std and P95 latency.
#
# Output: a single JSON line on stdout, e.g.
#   {"method": "PraNet", "params_m": 26.67, "macs_g": 61.15, "fps": 12.3,
#    "latency_mean_ms": 81.3, "latency_std_ms": 2.1, "latency_p95_ms": 85.0,
#    "input_size": 352, "device": "cuda"}
# =============================================================================
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch


def measure(model, size, device, iters, warmup, fp16):
    model = model.eval().to(device)
    x = torch.randn(1, 3, size, size, device=device)
    use_amp = fp16 and device == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.autocast("cuda"):
                    model(x)
            else:
                model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        lats = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_amp:
                with torch.autocast("cuda"):
                    model(x)
            else:
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)

    lats = np.asarray(lats)
    fps = 1000.0 / float(np.mean(lats))
    return {
        "fps": round(fps, 1),
        "latency_mean_ms": round(float(np.mean(lats)), 2),
        "latency_std_ms": round(float(np.std(lats, ddof=1)) if iters > 1 else 0.0, 2),
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
    }


def count_macs(model, size, device):
    """MACs via thop (same convention as our benchmark_speed.py)."""
    try:
        from thop import profile
        x = torch.randn(1, 3, size, size, device=device)
        model = model.eval().to(device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        return round(macs / 1e9, 2)
    except Exception as e:  # thop sometimes chokes on exotic modules
        print(f"[macs] thop failed: {e}", file=sys.stderr)
        return None


def patch_missing_weight_loads():
    """Vendor repos hard-code their authors' local ImageNet-weight paths and
    load them inside the constructor (e.g. hardnet68.pth, pvt_v2_b2.pth,
    res2net50_v1b_26w_4s-*.pth). Latency depends only on architecture, not
    on weight values, so we silently skip unavailable weight files and keep
    random initialization instead of hunting down dead download links."""
    import torch.nn as nn
    orig_load = torch.load

    def patched_load(path, *a, **k):
        p = str(path)
        if not os.path.exists(p):
            print(f"[bench] skipping unavailable weight file: {p}",
                  file=sys.stderr)
            return {}
        return orig_load(path, *a, **k)

    torch.load = patched_load

    orig_lsd = nn.Module.load_state_dict

    def patched_lsd(self, state_dict, *a, **k):
        if not state_dict:  # empty result from a skipped weight file
            return None
        try:
            return orig_lsd(self, state_dict, *a, **k)
        except Exception as e:  # partial/imcompatible vendor state dicts
            print(f"[bench] load_state_dict skipped ({type(e).__name__})",
                  file=sys.stderr)
            return None

    nn.Module.load_state_dict = patched_lsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["pranet", "polyp-pvt", "hardnet"])
    ap.add_argument("--repo", required=True, help="path to the cloned repo")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)  # some repos resolve weight paths relative to CWD

    patch_missing_weight_loads()

    name = {"pranet": "PraNet", "polyp-pvt": "Polyp-PVT",
            "hardnet": "HarDNet-MSEG"}[args.method]
    # All three vendors test at 352x352 under the PraNet protocol (each
    # repo's Test.py uses testsize=352), so every speed number below is
    # directly comparable at the same input resolution.
    size = 352

    if args.method == "pranet":
        from lib.PraNet_Res2Net import PraNet
        model = PraNet()
    elif args.method == "polyp-pvt":
        from lib.pvt import PolypPVT
        model = PolypPVT()
    else:
        from lib.HarDMSEG import HarDMSEG
        model = HarDMSEG()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    macs = count_macs(model, size, args.device)
    res = measure(model, size, args.device, args.iters, args.warmup, args.fp16)

    out = {"method": name, "params_m": round(params_m, 2),
           "macs_g": macs, "input_size": size, "device": args.device,
           "iters": args.iters, **res}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
