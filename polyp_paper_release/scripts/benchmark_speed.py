"""
Efficiency benchmark: Params (M), FLOPs (G), FPS.

Protocol (paper-consistent, fixed):
  - input 352x352, batch size 1, fp32, model.eval()
  - FPS measured on the SAME hardware for all compared methods
    (report the GPU/device name in the output); CUDA if available, else CPU
  - FLOPs counted with thop (MACs; we report GMACs as "GFLOPs", the common
    convention in the polyp-segmentation literature)
  - warmup 50 iterations, then timed over 300 iterations with torch.cuda.synchronize

Usage:
    python scripts/benchmark_speed.py [--checkpoint runs/exp1/best.pth] [--iters 300]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model_from_cfg  # noqa: E402


def count_flops(model, device, size=352, arch="lightpolysegnet") -> float:
    """Return GMACs via thop (a.k.a. 'GFLOPs' in most segmentation papers)."""
    try:
        from thop import profile
    except ImportError:
        print("[warn] thop not installed -> `pip install thop`; FLOPs skipped.")
        return float("nan")
    x = torch.randn(1, 3, size, size).to(device)
    if arch == "fanet":
        # zero feedback mask = first-iteration latency (protocol: batch=1, fp32)
        m = torch.zeros(1, 1, size, size, device=device)
        flops, _ = profile(model, inputs=(x, m), verbose=False)
    else:
        flops, _ = profile(model, inputs=(x,), verbose=False)
    return flops / 1e9


def measure_fps(model, device, iters=300, warmup=50, size=352,
                arch="lightpolysegnet") -> float:
    x = torch.randn(1, 3, size, size).to(device)
    if arch == "fanet":
        m = torch.zeros(1, 1, size, size, device=device)
        fwd = lambda: model(x, m)
    else:
        fwd = lambda: model(x)
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


def main():
    ap = argparse.ArgumentParser(description="Params / FLOPs / FPS benchmark")
    ap.add_argument("--checkpoint", default=None,
                    help="optional: load trained weights before benchmarking")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=352)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = "lightpolysegnet"
    model = build_model_from_cfg({"model": {}}, load_pretrained=False).to(device).eval()

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Build from the checkpoint's own config (arch may differ, e.g. exp3
        # timm-aligned backbone or the FANet baseline), then load weights.
        cfg = ckpt.get("config") or {"model": {}}
        arch = cfg.get("model", {}).get("arch", "lightpolysegnet")
        model = build_model_from_cfg(cfg, load_pretrained=False).to(device).eval()
        model.load_state_dict(ckpt["model"])
        print(f"[bench] loaded weights from {args.checkpoint} | arch={arch}")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = count_flops(model, device, args.image_size, arch)
    fps = measure_fps(model, device, args.iters, args.warmup, args.image_size, arch)

    hw = (torch.cuda.get_device_name(0) if device.type == "cuda"
          else f"{os.cpu_count()}-core CPU (torch {torch.__version__})")
    print("\n===== Efficiency report =====")
    print(f"Device        : {hw}")
    print(f"Input         : {args.image_size}x{args.image_size}, batch=1, fp32")
    print(f"Params        : {params_m:.2f} M")
    print(f"FLOPs (GMACs) : {gflops:.2f} G")
    print(f"FPS           : {fps:.1f} ({args.warmup} warmup + {args.iters} timed iters)")
    print("=============================")


if __name__ == "__main__":
    main()
