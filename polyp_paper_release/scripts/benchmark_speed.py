"""
Efficiency benchmark: Params (M), MACs (G), FPS, per-frame latency stats, fp16.

Protocol (paper-consistent, fixed; upgraded 2026-11-02 per external review):
  - input 352x352, batch size 1, fp32 (and optional fp16 pass), model.eval()
  - FPS measured on the SAME hardware for all compared methods
  - two timing modes:
      * throughput FPS: 50 warmup + N timed iterations, one
        torch.cuda.synchronize at the end (standard protocol)
      * per-frame latency: torch.cuda.synchronize BEFORE AND AFTER every
        frame; reports mean / std / P95 in milliseconds
  - MACs counted with thop AND cross-checked with
    torch.utils.flop_counter (FLOPs/2) -- thop misses functional matmul
    operators (e.g. attention), so the two numbers bound the true cost
  - FANet: zero feedback mask = first-iteration latency; its official
    protocol performs 10 such iterations per inference

Usage:
    python scripts/benchmark_speed.py [--checkpoint runs/main/last.pth] \
        [--iters 300] [--fp16]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model_from_cfg  # noqa: E402


def _make_inputs(model, device, size, half, arch):
    dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
    x = torch.randn(1, 3, size, size, device=device, dtype=dtype)
    if arch == "fanet":
        m = torch.zeros(1, 1, size, size, device=device, dtype=dtype)
        fwd = lambda: model(x, m)
    else:
        fwd = lambda: model(x)
    return fwd


def count_flops_thop(model, device, size=352, arch="lightpolysegnet") -> float:
    """Return GMACs via thop (multiply-accumulates; the common convention)."""
    try:
        from thop import profile
    except ImportError:
        print("[warn] thop not installed -> `pip install thop`; thop MACs skipped.")
        return float("nan")
    x = torch.randn(1, 3, size, size).to(device)
    if arch == "fanet":
        m = torch.zeros(1, 1, size, size, device=device)
        flops, _ = profile(model, inputs=(x, m), verbose=False)
    else:
        flops, _ = profile(model, inputs=(x,), verbose=False)
    return flops / 1e9


def count_flops_flop_counter(model, device, size=352, arch="lightpolysegnet"):
    """Cross-check GMACs with torch.utils.flop_counter (counts FLOPs; /2 = MACs).

    Catches functional matmul operators (attention etc.) that thop misses.
    Returns None if torch is too old.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    model = model.copy() if hasattr(model, "copy") else model
    x = torch.randn(1, 3, size, size).to(device)
    if arch == "fanet":
        m = torch.zeros(1, 1, size, size, device=device)
        inputs = (x, m)
    else:
        inputs = (x,)
    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad():
        with flop_counter:
            model(*inputs)
    total_flops = flop_counter.get_total_flops()
    return total_flops / 2 / 1e9   # FLOPs -> MACs


def measure_fps(model, device, iters=300, warmup=50, size=352,
                arch="lightpolysegnet", half=False):
    """Throughput FPS (single synchronize at the end)."""
    fwd = _make_inputs(model, device, size, half, arch)
    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            fwd()
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fwd()
        if is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return iters / dt


def measure_latency(model, device, iters=100, warmup=20, size=352,
                    arch="lightpolysegnet", half=False):
    """Per-frame latency (ms): synchronize before AND after every frame."""
    fwd = _make_inputs(model, device, size, half, arch)
    is_cuda = device.type == "cuda"
    lats = []
    with torch.no_grad():
        for _ in range(warmup):
            fwd()
        if is_cuda:
            torch.cuda.synchronize()
        for _ in range(iters):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fwd()
            if is_cuda:
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
    lats = np.array(lats)
    return {"mean": float(lats.mean()), "std": float(lats.std(ddof=1)),
            "p95": float(np.percentile(lats, 95))}


def main():
    ap = argparse.ArgumentParser(description="Params / MACs / FPS / latency benchmark")
    ap.add_argument("--checkpoint", default=None,
                    help="optional: load trained weights before benchmarking")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=352)
    ap.add_argument("--fp16", action="store_true",
                    help="also benchmark an fp16 (half precision) pass")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = "lightpolysegnet"
    model = build_model_from_cfg({"model": {}}, load_pretrained=False).to(device).eval()

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config") or {"model": {}}
        arch = cfg.get("model", {}).get("arch", "lightpolysegnet")
        model = build_model_from_cfg(cfg, load_pretrained=False).to(device).eval()
        model.load_state_dict(ckpt["model"])
        print(f"[bench] loaded weights from {args.checkpoint} | arch={arch}")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    gmacs_thop = count_flops_thop(model, device, args.image_size, arch)
    gmacs_fc = count_flops_flop_counter(model, device, args.image_size, arch)
    fps = measure_fps(model, device, args.iters, args.warmup, args.image_size, arch)
    lat = measure_latency(model, device, max(args.iters // 3, 50), 20,
                          args.image_size, arch)

    hw = (torch.cuda.get_device_name(0) if device.type == "cuda"
          else f"{os.cpu_count()}-core CPU (torch {torch.__version__})")
    print("\n===== Efficiency report =====")
    print(f"Device            : {hw}")
    print(f"Input             : {args.image_size}x{args.image_size}, batch=1, fp32")
    print(f"Params            : {params_m:.2f} M")
    print(f"MACs (thop)       : {gmacs_thop:.2f} G")
    if gmacs_fc is not None:
        print(f"MACs (flop_counter): {gmacs_fc:.2f} G   <- catches functional matmuls thop misses")
    print(f"FPS (throughput)  : {fps:.1f} ({args.warmup} warmup + {args.iters} timed iters)")
    print(f"Latency per frame : {lat['mean']:.2f} ± {lat['std']:.2f} ms "
          f"(P95 {lat['p95']:.2f} ms, n={max(args.iters // 3, 50)})")
    if args.fp16:
        if device.type != "cuda":
            print("[warn] fp16 requires CUDA; skipped.")
        else:
            fps16 = measure_fps(model, device, args.iters, args.warmup,
                                args.image_size, arch, half=True)
            lat16 = measure_latency(model, device, max(args.iters // 3, 50), 20,
                                    args.image_size, arch, half=True)
            print(f"FPS (fp16)        : {fps16:.1f}")
            print(f"Latency fp16      : {lat16['mean']:.2f} ± {lat16['std']:.2f} ms "
                  f"(P95 {lat16['p95']:.2f} ms)")
    print("=============================")


if __name__ == "__main__":
    main()
