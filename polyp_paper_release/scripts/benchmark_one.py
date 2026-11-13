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
import platform
import re
import sys
import time

import numpy as np
import torch


def platform_info(device):
    info = {"cpu": platform.processor() or platform.machine(),
            "threads": os.cpu_count()}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def make_forward(model, method, device):
    """Uniform forward(x) wrapper: FANet needs the feedback mask argument."""
    if method == "fanet":
        prev = torch.zeros(1, 1, 352, 352, device=device)
        return lambda x: model(x, prev)
    return lambda x: model(x)


def measure_e2e(model, size, device, iters, warmup, fp16, method):
    """End-to-end latency: CPU-side preprocessing (resize + normalize) ->
    device inference -> output postprocessing (sigmoid + resize back to a
    'original resolution' target), mirroring a deployment pipeline."""
    model = model.eval().to(device)
    use_amp = fp16 and device == "cuda"
    raw = torch.randint(0, 256, (480, 620, 3), dtype=torch.uint8).numpy()
    fwd = make_forward(model, method, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def one_pass():
        x = torch.from_numpy(raw).permute(2, 0, 1)[None].float().to(device) / 255.0
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                            align_corners=False)
        x = (x - mean) / std
        if use_amp:
            with torch.autocast("cuda"):
                out = fwd(x)
        else:
            out = fwd(x)
        seg = out["seg"] if isinstance(out, dict) else out[-1] if isinstance(out, (list, tuple)) else out
        prob = torch.sigmoid(seg.float())
        prob = torch.nn.functional.interpolate(prob, size=(480, 620), mode="bilinear",
                                               align_corners=False)
        return prob

    with torch.no_grad():
        for _ in range(warmup):
            one_pass()
        if device == "cuda":
            torch.cuda.synchronize()
        lats = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_pass()
            if device == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def measure(fwd, size, device, iters, warmup, fp16):
    """fwd: a callable taking an input tensor (model must already be in
    eval mode and on the right device)."""
    x = torch.randn(1, 3, size, size, device=device)
    use_amp = fp16 and device == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.autocast("cuda"):
                    fwd(x)
            else:
                fwd(x)
        if device == "cuda":
            torch.cuda.synchronize()

        lats = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_amp:
                with torch.autocast("cuda"):
                    fwd(x)
            else:
                fwd(x)
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


def count_macs(model, size, device, method=""):
    # MACs via thop (same convention as benchmark_speed.py). Runs on a
    # DEEPCOPY of the model: when thop fails mid-profile it leaves its
    # hooks and total_ops buffers attached to the model, which poison
    # every later forward pass (cpu/cuda mismatch inside the stale hook).
    # Two-input models (FANet.forward(image, mask)) get both inputs.
    import copy
    x = torch.randn(1, 3, size, size, device=device)
    if method == "fanet":
        mask = torch.zeros(1, 1, size, size, device=device)
        inputs = (x, mask)
    else:
        inputs = (x,)
    try:
        from thop import profile
        mc = copy.deepcopy(model).eval().to(device)
        with torch.no_grad():
            macs, _ = profile(mc, inputs=inputs, verbose=False)
        return round(macs / 1e9, 2)
    except Exception as e:
        print(f"[macs] thop failed ({e}); falling back to flop_counter",
              file=sys.stderr)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        mc = copy.deepcopy(model).eval().to(device)
        if method == "fanet":
            fwd = lambda: mc(x, mask)
        else:
            fwd = lambda: mc(x)
        with torch.no_grad(), FlopCounterMode(display=False) as fcm:
            fwd()
        return round(fcm.get_total_flops() / 1e9, 2)
    except Exception as e:
        print(f"[macs] flop_counter failed: {e}", file=sys.stderr)
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
                    choices=["pranet", "polyp-pvt", "hardnet", "fanet", "ours"])
    ap.add_argument("--repo", default="", help="path to the cloned repo "
                    "(empty for fanet, which lives in this project's src/)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--e2e", action="store_true",
                    help="end-to-end latency (pre+inference+postprocessing)")
    ap.add_argument("--paper-root", default="..",
                    help="project root (for the fanet method)")
    ap.add_argument("--our-config", default="configs/exp8_combo.yaml",
                    help="our model config (for the ours method)")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo) if args.repo else ""
    if repo:
        sys.path.insert(0, repo)
        os.chdir(repo)  # some repos resolve weight paths relative to CWD
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(script_dir, args.paper_root))
        os.chdir(root)
        sys.path.insert(0, root)

    patch_missing_weight_loads()

    name = {"pranet": "PraNet", "polyp-pvt": "Polyp-PVT",
            "hardnet": "HarDNet-MSEG", "fanet": "FANet",
            "ours": "LightPolySegNet"}[args.method]
    # All vendors test at 352x352 under the PraNet protocol (each repo's
    # Test.py uses testsize=352), so every speed number below is directly
    # comparable at the same input resolution.
    size = 352

    if args.method == "pranet":
        from lib.PraNet_Res2Net import PraNet
        model = PraNet()
    elif args.method == "polyp-pvt":
        from lib.pvt import PolypPVT
        model = PolypPVT()
    elif args.method == "fanet":
        from src.fanet import FANet
        model = FANet()
    elif args.method == "ours":
        import yaml as _yaml
        from src.model import build_model_from_cfg
        with open(args.our_config, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f)
        model = build_model_from_cfg(cfg, load_pretrained=False)
    else:
        from lib.HarDMSEG import HarDMSEG
        model = HarDMSEG()

    model = model.eval().to(args.device)   # GPU path: keep weights on-device

    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    out = {"method": name, "params_m": round(params_m, 2),
           "input_size": size, "device": args.device,
           "iters": args.iters, "platform": platform_info(args.device)}

    if args.e2e:
        lats = measure_e2e(model, size, args.device, args.iters,
                           args.warmup, args.fp16, args.method)
        out["e2e"] = True
        out["fps"] = round(1000.0 / float(np.mean(lats)), 1)
        out["latency_mean_ms"] = round(float(np.mean(lats)), 2)
        out["latency_std_ms"] = round(float(np.std(lats, ddof=1))
                                      if args.iters > 1 else 0.0, 2)
        out["latency_p95_ms"] = round(float(np.percentile(lats, 95)), 2)
    else:
        out["macs_g"] = count_macs(model, size, args.device, args.method)
        fwd = make_forward(model, args.method, args.device)
        out.update(measure(fwd, size, args.device, args.iters,
                           args.warmup, args.fp16))
        if args.method == "fanet":
            # Official FANet inference: 10 iterative feedback passes.
            model = model.eval().to(args.device)
            x = torch.randn(1, 3, size, size, device=args.device)
            prev = torch.zeros(1, 1, size, size, device=args.device)
            with torch.no_grad():
                for _ in range(max(2, args.warmup // 10)):
                    o = model(x, prev)
                    prev = (torch.sigmoid(o[0, 0]) > 0.5).float()[None, None]
                    for _ in range(10):
                        o = model(x, prev)
                        prev = (torch.sigmoid(o[0, 0]) > 0.5).float()[None, None]
                if args.device == "cuda":
                    torch.cuda.synchronize()
                lats = []
                for _ in range(max(10, args.iters // 10)):
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        o = model(x, prev)
                        prev = (torch.sigmoid(o[0, 0]) > 0.5).float()[None, None]
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    lats.append((time.perf_counter() - t0) * 1000.0)
            out["fps_10iter"] = round(1000.0 / float(np.mean(lats)), 1)
            out["latency10_mean_ms"] = round(float(np.mean(lats)), 2)

    print(json.dumps(out))


if __name__ == "__main__":
    main()
