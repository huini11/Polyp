#!/usr/bin/env python
# =============================================================================
# benchmark_competitors.py -- Same-GPU inference benchmark for competitor
# architectures (review round 2, item B6 / reviewer item "real-time is the
# biggest shortcoming").
#
# Benchmarks PraNet, Polyp-PVT, and HarDNet-MSEG on the SAME GPU with the
# SAME measurement protocol as our scripts/benchmark_speed.py, so the
# speed-accuracy table can finally be compared on equal terms.
#
# Design notes:
#  * Latency and FPS depend only on ARCHITECTURE + INPUT SIZE, not on weight
#    values. We therefore benchmark with random weights (no Google-Drive
#    weight hunts, no link rot). The paper reports these as
#    "architecture-level re-benchmark (latency is weight-independent)".
#  * Each vendor repo ships its own `lib` package, so every competitor is
#    benchmarked in an isolated subprocess (scripts/benchmark_one.py) to
#    avoid package-name collisions.
#  * Repos are shallow-cloned on first use into --repo-root.
#
# Usage (Colab, T4):
#   !python scripts/benchmark_competitors.py --iters 200 --warmup 20
# Output: results/competitor_speed.csv (+ per-method JSON lines on stdout)
# =============================================================================
import argparse
import csv
import json
import os
import subprocess
import sys

REPOS = {
    "pranet": "https://github.com/DengPingFan/PraNet.git",
    "polyp-pvt": "https://github.com/DengPingFan/Polyp-PVT.git",
    "hardnet": "https://github.com/james128333/HarDNet-MSEG.git",
}


def ensure_repo(name, root):
    d = os.path.join(root, name)
    if not os.path.isdir(os.path.join(d, ".git")):
        os.makedirs(root, exist_ok=True)
        url = REPOS[name]
        print(f"[clone] {url} -> {d}")
        subprocess.run(["git", "clone", "--depth", "1", url, d],
                       check=True)
    return d


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    paper_root = os.path.dirname(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200,
                    help="timed iterations per model (default 200)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (default: auto)")
    ap.add_argument("--fp16", action="store_true",
                    help="also run an fp16 autocast pass on cuda")
    ap.add_argument("--repo-root", default=os.path.join(paper_root, "external_repos"))
    ap.add_argument("--out", default=os.path.join(paper_root, "results",
                                                  "competitor_speed.csv"))
    args = ap.parse_args()

    device = args.device or ("cuda" if
                             __import__("torch").cuda.is_available() else "cpu")
    rows = []
    for key in ["pranet", "polyp-pvt", "hardnet"]:
        repo = ensure_repo(key, args.repo_root)
        cmd = [sys.executable, os.path.join(here, "benchmark_one.py"),
               "--method", key, "--repo", repo,
               "--iters", str(args.iters), "--warmup", str(args.warmup),
               "--device", device]
        if args.fp16:
            cmd.append("--fp16")
        print(f"\n=== benchmarking {key} ===")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] {key}:\n{r.stderr[-800:]}")
            rows.append({"method": key, "status": "FAILED",
                         "error": r.stderr.strip().splitlines()[-1][:200]
                         if r.stderr.strip() else "unknown"})
            continue
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        row = json.loads(line[-1])
        rows.append(row)
        print(f"[ok] {row['method']}: {row['params_m']}M params, "
              f"{row.get('macs_g')} GMACs, {row['fps']} FPS "
              f"@ {row['input_size']}x{row['input_size']} ({row['device']})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = ["method", "params_m", "macs_g", "input_size", "device", "iters",
            "fps", "latency_mean_ms", "latency_std_ms", "latency_p95_ms",
            "status", "error"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[saved] {args.out}")
    for r in rows:
        print(f"  {r.get('method')}: FPS={r.get('fps')} "
              f"latency={r.get('latency_mean_ms')}±{r.get('latency_std_ms')}ms "
              f"P95={r.get('latency_p95_ms')}ms")


if __name__ == "__main__":
    main()
