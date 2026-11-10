#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_runs.py — Phase 4 (robustness-auditor)
Summarize all results/exp*_main_results.csv into one comparison table
(CSV + Markdown), with:

  1. Multi-seed support: files named like exp3_seed43_main_results.csv are
     grouped under the base name `exp3` and reported as mean±std per metric.
  2. Paired per-dataset deltas between experiments, with human-readable
     interpretation (e.g. `exp3 - exp5` = ImageNet-pretraining contribution).

Usage (from the polyp_paper/ root):
    python scripts/aggregate_runs.py                       # default: all CSVs, base = last exp
    python scripts/aggregate_runs.py --base exp3           # deltas vs exp3
    python scripts/aggregate_runs.py --metric mIoU         # switch metric
    python scripts/aggregate_runs.py --pairs "exp3-exp5=ImageNet预训练贡献,exp3-exp4=边界损失贡献"

Outputs:
    results/comparison_table.csv
    results/comparison_table.md

Stdlib only (no pandas) so it runs anywhere, incl. the local venv.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

# Paper-table dataset order (PraNet-protocol literature standard).
DATASET_ORDER = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-Larib", "Mean"]
DATASET_ALIASES = {"CVC-300": ["CVC-300", "CVC-ClinicDB-300", "CVC300"],
                   "CVC-ClinicDB": ["CVC-ClinicDB", "ClinicDB"],
                   "CVC-ColonDB": ["CVC-ColonDB", "ColonDB"],
                   "ETIS-Larib": ["ETIS-Larib", "ETIS"],
                   "Kvasir-SEG": ["Kvasir-SEG", "Kvasir"]}
METRICS = ["mDice", "mIoU", "BF score (352)", "Boundary IoU (352)"]

# Experiment interpretation registry (extended as new exps finish).
EXP_MEANING = {
    "exp1": "v1 backbone, random init, no color aug (baseline)",
    "exp2": "v1 backbone, random init, + color aug",
    "exp3": "FINAL: timm-aligned backbone + ImageNet pretrain + color aug",
    "exp4_no_boundary": "exp3 config but w_boundary=0 (boundary-loss ablation)",
    "exp5_randinit": "exp3 config but pretrained=false (pretrain ablation)",
    "fanet_baseline": "FANet retrained by us under the identical protocol "
                      "(352/bs8/100ep/AdamW), evaluated with its official "
                      "10-iteration feedback protocol",
}
PAIR_MEANING = {
    ("exp2", "exp1"): "颜色增强贡献（v1 骨干上）",
    ("exp3", "exp2"): "骨干对齐+预训练+（保留颜色增强）的合并增益",
    ("exp5", "exp2"): "timm 骨干重设计贡献（均为随机初始化+颜色增强）",
    ("exp3", "exp5"): "ImageNet 预训练贡献（timm 骨干上）",
    ("exp3", "exp4"): "边界感知损失贡献（exp3 完整配置）",
    ("exp3", "fanet_baseline"): "Ours vs 受控重训 FANet（同协议）",
}


def canon_dataset(name: str) -> str:
    name = name.strip()
    for canon, aliases in DATASET_ALIASES.items():
        if name in aliases:
            return canon
    return name


def parse_csv(path: str):
    """Return {dataset: {metric: float-or-None}} from one eval CSV."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        ds = canon_dataset(r["Dataset"])
        rec = {}
        for m in METRICS:
            v = r.get(m)
            try:
                rec[m] = float(v) if v not in (None, "", "--") else None
            except (TypeError, ValueError):
                rec[m] = None
        out[ds] = rec
    return out


def exp_name_from_path(path: str) -> str:
    """results/exp3_seed43_main_results.csv -> (base='exp3', seed=43);
    results/exp4_no_boundary_main_results.csv -> ('exp4_no_boundary', None);
    results/fanet_baseline_main_results.csv  -> ('fanet_baseline', None)."""
    stem = os.path.basename(path)
    m = re.match(r"(?P<base>.+?)_seed(?P<seed>\d+)_main_results\.csv$", stem)
    if m:
        return m.group("base"), int(m.group("seed"))
    m = re.match(r"(?P<base>.+)_main_results\.csv$", stem)
    if m:
        return m.group("base"), None
    return os.path.splitext(stem)[0], None


def mean_std(vals):
    n = len(vals)
    mu = sum(vals) / n
    if n == 1:
        return mu, None
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)  # sample std
    return mu, math.sqrt(var)


def fmt(v, std=None, digits=4):
    if v is None:
        return "--"
    if std is None:
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f}±{std:.{digits}f}"


def resolve_base(name: str, known) -> str:
    """Resolve shorthand ('exp4', 'exp5') to actual run-base names
    ('exp4_no_boundary', 'exp5_randinit', ...). Exact match wins."""
    if name in known:
        return name
    hits = [b for b in known if b.startswith(name)]
    if len(hits) == 1:
        return hits[0]
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--metric", default="mDice", choices=METRICS + ["BF score", "Boundary IoU"],
                    help="headline metric for the comparison table")
    ap.add_argument("--base", default=None, help="experiment that others are compared to (deltas)")
    ap.add_argument("--pairs", default=None,
                    help='extra deltas, e.g. "exp3-exp5=ImageNet预训练贡献,exp3-exp4=边界损失贡献"')
    ap.add_argument("--digits", type=int, default=4)
    ap.add_argument("--out-prefix", default=None,
                    help="output file prefix (default: results/comparison_table)")
    args = ap.parse_args()

    if args.metric == "BF score":
        args.metric = "BF score (352)"
    elif args.metric == "Boundary IoU":
        args.metric = "Boundary IoU (352)"

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*_main_results.csv")))
    if not paths:
        sys.exit(f"[aggregate_runs] no exp*_main_results.csv found in {args.results_dir}/")

    # ---- group runs: base name -> {seed: parsed} -------------------------
    runs = {}
    for p in paths:
        base, seed = exp_name_from_path(p)
        runs.setdefault(base, {})[seed if seed is not None else 0] = parse_csv(p)

    # ---- aggregate per experiment (mean±std across seeds) ----------------
    agg = {}   # base -> {dataset: {metric: (mean, std)}}
    for base, seeds in runs.items():
        agg[base] = {}
        for ds in DATASET_ORDER:
            agg[base][ds] = {}
            for m in METRICS:
                vals = [rec[ds][m] for rec in seeds.values()
                        if ds in rec and rec[ds][m] is not None]
                if vals:
                    mu, sd = mean_std(vals)
                    agg[base][ds][m] = (mu, sd)
                else:
                    agg[base][ds][m] = (None, None)

    exp_order = sorted(agg.keys())
    base = args.base or exp_order[-1]
    n_seeds = {b: len(runs[b]) for b in exp_order}

    # ---- build comparison table ------------------------------------------
    out_prefix = args.out_prefix or os.path.join(args.results_dir, "comparison_table")
    mt = args.metric
    cols = ["Experiment", "Description", "Seeds"] + \
           [f"{ds} ({mt})" for ds in DATASET_ORDER]

    def row_for(b):
        desc = EXP_MEANING.get(b, "")
        row = [b, desc, str(n_seeds[b]) if n_seeds[b] > 1 else "1"]
        for ds in DATASET_ORDER:
            mu, sd = agg[b][ds][mt]
            row.append(fmt(mu, sd, args.digits))
        return row

    csv_rows = [cols] + [row_for(b) for b in exp_order]

    # ---- paired deltas ----------------------------------------------------
    pair_specs = []
    if args.pairs:
        for spec in args.pairs.split(","):
            lhs, _, label = spec.partition("=")
            a, _, b = lhs.strip().partition("-")
            pair_specs.append((a.strip(), b.strip(), label.strip() or ""))
    else:
        # default: every experiment vs --base
        for b in exp_order:
            if b != base:
                pair_specs.append((base, b, PAIR_MEANING.get((base, b), "")))

    delta_rows = []
    for a, b, label in pair_specs:
        a = resolve_base(a, exp_order)
        b = resolve_base(b, exp_order)
        if a not in agg or b not in agg:
            missing = a if a not in agg else b
            delta_rows.append([f"{a} - {b}", label or "", f"[missing {missing}]"] +
                              ["--"] * len(DATASET_ORDER))
            continue
        row = [f"{a} - {b}", label or PAIR_MEANING.get((a, b), ""), ""]
        for ds in DATASET_ORDER:
            ma, _ = agg[a][ds][mt]
            mb, _ = agg[b][ds][mt]
            row.append(fmt(None if (ma is None or mb is None) else ma - mb,
                           None, args.digits))
        delta_rows.append(row)

    # ---- write CSV ---------------------------------------------------------
    with open(out_prefix + ".csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)
        w.writerow([])
        w.writerow(["Delta", "Interpretation", ""] + [ds for ds in DATASET_ORDER])
        w.writerows(delta_rows)

    # ---- write Markdown ----------------------------------------------------
    def md_table(header, rows):
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join(["---"] * len(header)) + "|"]
        lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(lines)

    md = []
    md.append(f"# Run aggregation — {args.metric} (PraNet protocol, 5 test sets)\n")
    md.append(f"Source: `{args.results_dir}/exp*_main_results.csv` | "
              f"generated by `scripts/aggregate_runs.py`\n")
    md.append("## 1. Per-experiment results\n")
    md.append(md_table(cols, [row_for(b) for b in exp_order]))
    md.append("\n*多 seed 的实验以 mean±std 报告；`--` 表示该 CSV 未包含此列（如 exp1 无 352 协议边界指标）。*\n")
    md.append("## 2. Paired per-dataset deltas\n")
    md.append(md_table(["Delta", "Interpretation"] + DATASET_ORDER,
                       [[r[0], r[1]] + r[3:] for r in delta_rows]))
    md.append("\n## 3. Interpretation guide\n")
    md.append("- `exp3 - exp5` = **ImageNet 预训练贡献**（同为 timm 骨干 + 颜色增强）")
    md.append("- `exp3 - exp4` = **边界感知损失贡献**（exp4 仅把 w_boundary 0.5→0.0）")
    md.append("- `exp2 - exp1` = **颜色增强贡献**（同为 v1 骨干、随机初始化）")
    md.append("- `exp5 - exp2` = **骨干重设计贡献**（同为随机初始化 + 颜色增强）")
    md.append("- 预期方向：跨域集（CVC-300/ColonDB/ETIS）对增强与预训练最敏感；"
              "若 exp4 ≥ exp3（去边界损失反而更好），需检查边界头是否过拟合 ClinicDB 平滑边界，"
              "并考虑把边界分支降级为可选模块（见 phase4_ablation_plan.md 预案 B）。")

    with open(out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- console echo ------------------------------------------------------
    print("\n".join(md))
    print(f"\n[aggregate_runs] wrote {out_prefix}.csv and {out_prefix}.md")


if __name__ == "__main__":
    main()
