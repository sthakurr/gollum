"""
Statistical summary of benchmark experiments.
Works with single-branch results (shows what's available) AND with
two-branch results (runs paired statistical tests for comparison).

Usage:
    python benchmark/analysis/summarize.py
"""

import json
import os
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── helpers ────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_results(prefix: str):
    """
    Returns (orig_path, optim_path) where:
      orig_path  = *_main.json  (original branch)
      optim_path = *_scale.json (optimized branch)
    Either may be None if that branch hasn't been run yet.
    """
    if not RESULTS_DIR.exists():
        return None, None
    files = {p.name: p for p in RESULTS_DIR.glob(f"{prefix}_*.json")}
    orig = next((v for k, v in files.items() if "main" in k), None)
    optim = next((v for k, v in files.items() if "scale" in k), None)
    return orig, optim


def _header(title: str):
    print(f"\n{'='*60}")
    print(title)
    print("=" * 60)


def _warn_missing(orig, optim):
    missing = []
    if not orig:
        missing.append("main (original) branch")
    if not optim:
        missing.append("scale (optimized) branch")
    if missing:
        print(f"  [!] Missing results for: {', '.join(missing)}")
        print(f"      Run: python benchmark/run_benchmarks.py")
        if missing == ["scale (optimized) branch"]:
            print(f"      (switch to scale branch first: git checkout scale)")
    return bool(missing)


def _available_data(orig_path, optim_path):
    """Return which data is available as (orig_data, optim_data), either may be None."""
    orig = load_json(orig_path) if orig_path else None
    optim = load_json(optim_path) if optim_path else None
    return orig, optim


def print_table(rows, headers):
    if not rows:
        print("  (no data)")
        return
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


def wilcoxon_test(a, b):
    try:
        from scipy.stats import wilcoxon
        if len(a) < 2 or len(b) < 2:
            return None, None, None
        stat, p = wilcoxon(a, b)
        n = len(a)
        r = 1 - (2 * stat) / (n * (n + 1))
        return float(stat), float(p), float(r)
    except ImportError:
        return None, None, None


def paired_ttest(a, b):
    try:
        from scipy.stats import ttest_rel
        if len(a) < 2 or len(b) < 2:
            return None, None, None
        t, p = ttest_rel(a, b)
        diff = np.array(a) - np.array(b)
        d = diff.mean() / (diff.std(ddof=1) + 1e-12)
        return float(t), float(p), float(d)
    except ImportError:
        return None, None, None


def fmt_mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


# ── Experiment 1: VRAM ─────────────────────────────────────────────────────────

def summarize_exp1():
    _header("EXPERIMENT 1 — Peak VRAM Usage (MB)")
    orig_path, optim_path = find_results("exp1_vram")
    partial = _warn_missing(orig_path, optim_path)
    orig, optim = _available_data(orig_path, optim_path)

    if not orig and not optim:
        return

    # Collect per (config, seed) VRAM values
    def group_by_config(data):
        d = {}
        if data is None:
            return d
        for r in data:
            if r.get("peak_vram_mb") is not None:
                d.setdefault(r["config"], []).append(r["peak_vram_mb"])
        return d

    orig_d = group_by_config(orig)
    optim_d = group_by_config(optim)
    all_configs = sorted(set(list(orig_d.keys()) + list(optim_d.keys())))

    rows = []
    for cfg in all_configs:
        o_vals = orig_d.get(cfg, [])
        p_vals = optim_d.get(cfg, [])
        paired = list(zip(o_vals, p_vals))
        if paired:
            a, b = zip(*paired)
            reduction = 100 * (np.mean(a) - np.mean(b)) / (np.mean(a) + 1e-9)
            _, p, r = wilcoxon_test(list(a), list(b))
            rows.append([
                cfg,
                fmt_mean_std(list(a)),
                fmt_mean_std(list(b)),
                f"{reduction:.1f}%",
                f"{p:.4f}" if p is not None else "n/a",
                f"{r:.3f}" if r is not None else "n/a",
            ])
        elif o_vals:
            rows.append([cfg, fmt_mean_std(o_vals), "not run", "—", "—", "—"])
        elif p_vals:
            rows.append([cfg, "not run", fmt_mean_std(p_vals), "—", "—", "—"])

    print_table(rows, ["Config", "Original MB", "Optimized MB", "Reduction", "p (Wilcoxon)", "Effect r"])
    if partial:
        print("\n  Run on scale branch to enable comparison statistics.")


# ── Experiment 2: Iter Time ────────────────────────────────────────────────────

def summarize_exp2():
    _header("EXPERIMENT 2 — Wall-Clock Time per BO Iteration")
    orig_path, optim_path = find_results("exp2_iter_time")
    partial = _warn_missing(orig_path, optim_path)
    orig, optim = _available_data(orig_path, optim_path)

    if not orig and not optim:
        return

    def group(data):
        d = {}
        if data is None:
            return d
        for r in data:
            if r.get("mean_iter_time_s") is not None:
                d.setdefault(r["config"], []).append(r["mean_iter_time_s"])
        return d

    orig_d = group(orig)
    optim_d = group(optim)
    all_configs = sorted(set(list(orig_d.keys()) + list(optim_d.keys())))

    rows = []
    for cfg in all_configs:
        o = orig_d.get(cfg, [])
        p = optim_d.get(cfg, [])
        if o and p:
            n = min(len(o), len(p))
            speedup = np.mean(o[:n]) / (np.mean(p[:n]) + 1e-12)
            _, pval, _ = paired_ttest(o[:n], p[:n])
            rows.append([
                cfg,
                fmt_mean_std(o),
                fmt_mean_std(p),
                f"{speedup:.2f}x",
                f"{pval:.4f}" if pval is not None else "n/a",
            ])
        elif o:
            rows.append([cfg, fmt_mean_std(o), "not run", "—", "—"])
        elif p:
            rows.append([cfg, "not run", fmt_mean_std(p), "—", "—"])

    print_table(rows, ["Config", "Original s", "Optimized s", "Speedup", "p (t-test)"])


# ── Experiment 3: Throughput ───────────────────────────────────────────────────

def summarize_exp3():
    _header("EXPERIMENT 3 — Featurization Throughput Scaling")
    print("  b = power-law exponent: b≈0 is ideal (flat throughput), b<0 = degradation")
    orig_path, optim_path = find_results("exp3_throughput")
    partial = _warn_missing(orig_path, optim_path)
    orig, optim = _available_data(orig_path, optim_path)

    if not orig and not optim:
        return

    def get_pl(data, featurizer):
        if data is None:
            return None
        return next((r for r in data if r.get("summary") == "power_law" and r["featurizer"] == featurizer), None)

    def get_points(data, featurizer):
        if data is None:
            return []
        return [(r["n"], r["throughput_samples_per_sec"]) for r in data
                if r.get("n") and r.get("featurizer") == featurizer
                and r.get("throughput_samples_per_sec") is not None]

    featurizers = set()
    for d in [orig, optim]:
        if d:
            featurizers.update(r.get("featurizer") for r in d if r.get("featurizer"))

    rows = []
    for feat in sorted(featurizers):
        o_pl = get_pl(orig, feat)
        p_pl = get_pl(optim, feat)

        def fmt_pl(pl):
            if pl is None:
                return "not run"
            return f"b={pl['b']:.3f} [{pl['b_ci_low']:.3f}, {pl['b_ci_high']:.3f}]"

        # Per-N throughput comparison
        o_pts = dict(get_points(orig, feat))
        p_pts = dict(get_points(optim, feat))
        all_ns = sorted(set(o_pts) | set(p_pts))
        if all_ns:
            for n in all_ns:
                o_tp = f"{o_pts[n]:.1f}" if n in o_pts else "—"
                p_tp = f"{p_pts[n]:.1f}" if n in p_pts else "—"
                rows.append([feat[:25], str(n), o_tp, p_tp])

    print_table(rows, ["Featurizer", "N", "Original samp/s", "Optimized samp/s"])

    # Power-law summary
    print("\n  Power-law exponents (b):")
    for feat in sorted(featurizers):
        o_pl = get_pl(orig, feat)
        p_pl = get_pl(optim, feat)
        print(f"    {feat[:30]}: original={fmt_pl(o_pl)}  optimized={fmt_pl(p_pl)}")


# ── Experiment 4: Quality ──────────────────────────────────────────────────────

def summarize_exp4():
    _header("EXPERIMENT 4 — BO Quality (Numerical Correctness)")
    print("  Null: bfloat16/checkpointing does not change BO performance")
    print(f"  Bonferroni-corrected α = 0.05/{4*1} ≈ {0.05/4:.4f}")
    orig_path, optim_path = find_results("exp4_quality")
    partial = _warn_missing(orig_path, optim_path)
    orig, optim = _available_data(orig_path, optim_path)

    if not orig and not optim:
        return

    def group_by_checkpoint(data):
        d = {}
        if data is None:
            return d
        for r in data:
            for cp, vals in (r.get("checkpoints") or {}).items():
                if vals.get("best_y") is not None:
                    d.setdefault(cp, []).append(vals["best_y"])
        return d

    orig_d = group_by_checkpoint(orig)
    optim_d = group_by_checkpoint(optim)
    all_cps = sorted(set(list(orig_d.keys()) + list(optim_d.keys())), key=lambda x: int(x))
    alpha_c = 0.05 / max(len(all_cps), 1)

    rows = []
    for cp in all_cps:
        o = orig_d.get(cp, [])
        p = optim_d.get(cp, [])
        if o and p:
            n = min(len(o), len(p))
            _, pval, d = paired_ttest(o[:n], p[:n])
            sig = "YES *" if (pval is not None and pval < alpha_c) else "no"
            rows.append([
                f"iter {cp}", fmt_mean_std(o), fmt_mean_std(p),
                f"{pval:.4f}" if pval is not None else "n/a", sig,
            ])
        elif o:
            rows.append([f"iter {cp}", fmt_mean_std(o), "not run", "—", "—"])
        elif p:
            rows.append([f"iter {cp}", "not run", fmt_mean_std(p), "—", "—"])

    print_table(rows, ["Checkpoint", "Original best_y", "Optimized best_y", "p", "Significant?"])
    print(f"\n  * = significant at Bonferroni-corrected α={alpha_c:.4f}")
    print("  Goal: NO significant differences (numerical equivalence confirmed)")


# ── Experiment 5: OOM Threshold ────────────────────────────────────────────────

def summarize_exp5():
    _header("EXPERIMENT 5 — OOM Threshold (Scalability Multiplier)")
    orig_path, optim_path = find_results("exp5_oom")
    partial = _warn_missing(orig_path, optim_path)
    orig, optim = _available_data(orig_path, optim_path)

    if not orig and not optim:
        return

    def get_summary(data):
        if data is None:
            return None
        return next((r for r in data if r.get("summary")), None)

    def get_entries(data):
        if data is None:
            return {}
        return {r["n"]: r for r in data if not r.get("summary") and r.get("n")}

    orig_s = get_summary(orig)
    optim_s = get_summary(optim)
    orig_e = get_entries(orig)
    optim_e = get_entries(optim)

    n_orig = orig_s["oom_threshold_n"] if orig_s else None
    n_optim = optim_s["oom_threshold_n"] if optim_s else None

    if n_orig is not None and n_optim is not None:
        multiplier = n_optim / max(n_orig, 1)
        print(f"  Original OOM threshold:  N = {n_orig}")
        print(f"  Optimized OOM threshold: N = {n_optim}")
        print(f"  Scalability multiplier:  {multiplier:.2f}x")
    elif n_orig is not None:
        print(f"  Original OOM threshold: N = {n_orig}  (optimized not yet run)")
    elif n_optim is not None:
        print(f"  Optimized OOM threshold: N = {n_optim}  (original not yet run)")

    all_ns = sorted(set(list(orig_e.keys()) + list(optim_e.keys())))
    rows = []
    for n in all_ns:
        o = orig_e.get(n, {})
        p = optim_e.get(n, {})
        rows.append([
            n,
            f"{o.get('success_rate', '—'):.0%}" if o else "—",
            f"{p.get('success_rate', '—'):.0%}" if p else "—",
        ])
    print_table(rows, ["N", "Original success rate", "Optimized success rate"])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\nGOLLuM Scalability Benchmark Summary")
    print("Original branch: main  |  Optimized branch: scale")
    print(f"Results directory: {RESULTS_DIR}")
    summarize_exp1()
    summarize_exp2()
    summarize_exp3()
    summarize_exp4()
    summarize_exp5()
    print("\nDone. To generate figures: python benchmark/analysis/plots.py")


if __name__ == "__main__":
    main()
