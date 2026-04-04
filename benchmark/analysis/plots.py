"""
Generates analysis figures from benchmark results.
Works with single-branch data (baseline profile) and two-branch data (comparison).
Outputs saved to benchmark/analysis/figures/

Usage:
    python benchmark/analysis/plots.py
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

MAIN_COLOR = "#E74C3C"
SCALE_COLOR = "#2ECC71"
ACCENT = "#3498DB"
MAIN_LABEL = "Original (main)"
SCALE_LABEL = "Optimized (scale)"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_results(prefix):
    if not RESULTS_DIR.exists():
        return None, None
    files = {p.name: p for p in RESULTS_DIR.glob(f"{prefix}_*.json")}
    orig = next((v for k, v in files.items() if "main" in k), None)
    optim = next((v for k, v in files.items() if "scale" in k), None)
    return orig, optim


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Peak VRAM per seed (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_exp1():
    orig_path, optim_path = find_results("exp1_vram")
    if not orig_path and not optim_path:
        print("[plots] Exp1 — no results, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for path, color, label, offset in [
        (orig_path, MAIN_COLOR, MAIN_LABEL, -0.2),
        (optim_path, SCALE_COLOR, SCALE_LABEL, 0.2),
    ]:
        if path is None:
            continue
        data = load_json(path)
        entries = [r for r in data if r.get("peak_vram_mb") and not r.get("error")]
        if not entries:
            continue
        seeds = [r["seed"] for r in entries]
        vrams = [r["peak_vram_mb"] for r in entries]
        ax.bar([s + offset for s in seeds], vrams, width=0.35,
               color=color, alpha=0.85, label=label, edgecolor="white")
        # Annotate mean
        mean_v = np.mean(vrams)
        ax.axhline(mean_v, color=color, linestyle="--", linewidth=1, alpha=0.6)
        ax.text(max(seeds) + 0.7, mean_v, f"mean={mean_v:.0f} MB",
                color=color, fontsize=9, va="center")

    ax.set_xlabel("Seed")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Exp 1: Peak GPU Memory — DeepGP + t5-base + LoRA", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xticks(sorted({r["seed"] for p in [orig_path, optim_path] if p for r in load_json(p) if r.get("seed")}))

    out = FIGURES_DIR / "exp1_vram.png"
    plt.savefig(out)
    print(f"[plots] Saved {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: BO iteration time distribution (violin + strip)
# ─────────────────────────────────────────────────────────────────────────────

def plot_exp2():
    orig_path, optim_path = find_results("exp2_iter_time")
    if not orig_path and not optim_path:
        print("[plots] Exp2 — no results, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: per-iteration time series across all seeds
    ax = axes[0]
    for path, color, label in [
        (orig_path, MAIN_COLOR, MAIN_LABEL),
        (optim_path, SCALE_COLOR, SCALE_LABEL),
    ]:
        if path is None:
            continue
        data = load_json(path)
        all_times = {}
        for r in data:
            for i, t in enumerate(r.get("iter_times_s", [])):
                all_times.setdefault(i, []).append(t)
        if not all_times:
            continue
        iters = sorted(all_times.keys())
        means = [np.mean(all_times[i]) for i in iters]
        stds = [np.std(all_times[i]) for i in iters]
        ax.plot([i + 1 for i in iters], means, "o-", color=color, label=label, markersize=5)
        ax.fill_between(
            [i + 1 for i in iters],
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )
    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_title("Time per BO Iteration (mean ± std over seeds)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: violin of all iteration times
    ax2 = axes[1]
    violins = []
    labels_v = []
    colors_v = []
    for path, color, label in [
        (orig_path, MAIN_COLOR, "main"),
        (optim_path, SCALE_COLOR, "scale"),
    ]:
        if path is None:
            continue
        data = load_json(path)
        all_t = [t for r in data for t in r.get("iter_times_s", [])]
        if all_t:
            violins.append(all_t)
            labels_v.append(label)
            colors_v.append(color)

    if violins:
        parts = ax2.violinplot(violins, showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors_v[i])
            pc.set_alpha(0.6)
        ax2.set_xticks(range(1, len(labels_v) + 1))
        ax2.set_xticklabels(labels_v)
        ax2.set_ylabel("Time (s)")
        ax2.set_title("Distribution")
        # Add mean annotation
        for i, vs in enumerate(violins):
            ax2.text(i + 1, np.mean(vs) + 0.1, f"μ={np.mean(vs):.2f}s",
                     ha="center", fontsize=9, color=colors_v[i])

    fig.suptitle("Exp 2: Wall-Clock Time per BO Iteration", fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "exp2_iter_time.png"
    plt.savefig(out)
    print(f"[plots] Saved {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Featurization throughput vs N (log-log with power-law fit)
# ─────────────────────────────────────────────────────────────────────────────

def plot_exp3():
    orig_path, optim_path = find_results("exp3_throughput")
    if not orig_path and not optim_path:
        print("[plots] Exp3 — no results, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    featurizer_names = {"get_tokens": "get_tokens()", "get_huggingface_embeddings": "get_huggingface_embeddings()"}

    for ax_idx, feat in enumerate(["get_tokens", "get_huggingface_embeddings"]):
        ax = axes[ax_idx]
        for path, color, label, marker in [
            (orig_path, MAIN_COLOR, MAIN_LABEL, "o"),
            (optim_path, SCALE_COLOR, SCALE_LABEL, "s"),
        ]:
            if path is None:
                continue
            data = load_json(path)
            pts = sorted(
                [(r["n"], r["throughput_samples_per_sec"]) for r in data
                 if r.get("featurizer") == feat and r.get("n") and r.get("throughput_samples_per_sec")],
                key=lambda x: x[0],
            )
            pl = next((r for r in data if r.get("summary") == "power_law" and r.get("featurizer") == feat), None)
            if not pts:
                continue
            ns, tps = zip(*pts)
            ax.plot(ns, tps, f"{marker}-", color=color, label=label, markersize=7, linewidth=2)

            # Power-law fit line
            if pl:
                b = pl["b"]
                log_ns = np.linspace(np.log(min(ns)), np.log(max(ns)), 50)
                a = np.mean(np.log(tps)) - b * np.mean(np.log(ns))
                fit_tps = np.exp(a + b * log_ns)
                ax.plot(np.exp(log_ns), fit_tps, "--", color=color, alpha=0.5, linewidth=1)
                ax.text(
                    ns[-1], tps[-1] * 0.7,
                    f"b={b:.2f}\n[{pl['b_ci_low']:.2f}, {pl['b_ci_high']:.2f}]",
                    fontsize=8, color=color, ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dataset Size N")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title(featurizer_names.get(feat, feat))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Exp 3: Featurization Throughput Scaling (log-log)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "exp3_throughput.png"
    plt.savefig(out)
    print(f"[plots] Saved {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: BO quality curves (best_y vs iteration)
# ─────────────────────────────────────────────────────────────────────────────

def plot_exp4():
    orig_path, optim_path = find_results("exp4_quality")
    if not orig_path and not optim_path:
        print("[plots] Exp4 — no results, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for path, color, label, marker in [
        (orig_path, MAIN_COLOR, MAIN_LABEL, "o"),
        (optim_path, SCALE_COLOR, SCALE_LABEL, "s"),
    ]:
        if path is None:
            continue
        data = load_json(path)

        # Collect best_y per checkpoint per seed
        cp_values = {}
        for r in data:
            for cp_str, vals in (r.get("checkpoints") or {}).items():
                cp = int(cp_str)
                if vals.get("best_y") is not None:
                    cp_values.setdefault(cp, []).append(vals["best_y"])

        if not cp_values:
            continue
        cps = sorted(cp_values.keys())
        means = [np.mean(cp_values[c]) for c in cps]
        stds = [np.std(cp_values[c]) for c in cps]

        ax.errorbar(cps, means, yerr=stds, fmt=f"{marker}-", color=color,
                     label=label, capsize=4, linewidth=2, markersize=8)

        # Annotate final value
        ax.annotate(
            f"{means[-1]:.1f} ± {stds[-1]:.1f}",
            xy=(cps[-1], means[-1]),
            xytext=(cps[-1] + 1.5, means[-1] - 1.5),
            fontsize=9, color=color,
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.5),
        )

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Best Observed Yield")
    ax.set_title("Exp 4: BO Optimization Quality\n(Buchwald-Hartwig, 5 seeds, mean ± std)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    out = FIGURES_DIR / "exp4_quality.png"
    plt.savefig(out)
    print(f"[plots] Saved {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: OOM threshold heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_exp5():
    orig_path, optim_path = find_results("exp5_oom")
    if not orig_path and not optim_path:
        print("[plots] Exp5 — no results, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35

    all_ns = set()
    branch_data = {}
    for path, label, color in [
        (orig_path, MAIN_LABEL, MAIN_COLOR),
        (optim_path, SCALE_LABEL, SCALE_COLOR),
    ]:
        if path is None:
            continue
        data = load_json(path)
        rates = {r["n"]: r["success_rate"] for r in data if r.get("n") and not r.get("summary")}
        branch_data[label] = (rates, color)
        all_ns.update(rates.keys())

    ns = sorted(all_ns)
    x = np.arange(len(ns))

    for i, (label, (rates, color)) in enumerate(branch_data.items()):
        offset = (i - 0.5 * (len(branch_data) - 1)) * w
        vals = [rates.get(n, 0) for n in ns]
        bars = ax.bar(x + offset, vals, w * 0.9, color=color, alpha=0.85, label=label, edgecolor="white")
        # Color bars red if failed
        for bar, v in zip(bars, vals):
            if v < 1.0:
                bar.set_facecolor("#E74C3C")
                bar.set_alpha(0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Dataset Size N")
    ax.set_ylabel("Success Rate (no OOM)")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend()
    ax.set_title("Exp 5: OOM Threshold — Success Rate by Dataset Size", fontweight="bold")

    # Annotate threshold
    for label, (rates, color) in branch_data.items():
        threshold = max((n for n, r in rates.items() if r == 1.0), default=0)
        if threshold:
            ax.annotate(
                f"Threshold: N={threshold}",
                xy=(ns.index(threshold), 1.0),
                xytext=(ns.index(threshold) + 0.5, 1.08),
                fontsize=9, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color),
            )

    out = FIGURES_DIR / "exp5_oom.png"
    plt.savefig(out)
    print(f"[plots] Saved {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: combined summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard():
    """4-panel summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("GOLLuM Scalability Benchmark — Baseline Profile (main branch)",
                 fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: VRAM
    ax1 = fig.add_subplot(2, 2, 1)
    orig_path, _ = find_results("exp1_vram")
    if orig_path:
        data = load_json(orig_path)
        entries = [r for r in data if r.get("peak_vram_mb") and not r.get("error")]
        if entries:
            seeds = [r["seed"] for r in entries]
            vrams = [r["peak_vram_mb"] for r in entries]
            ax1.bar(seeds, vrams, color=MAIN_COLOR, alpha=0.85, edgecolor="white")
            mean_v = np.mean(vrams)
            ax1.axhline(mean_v, color="black", linestyle="--", linewidth=1)
            ax1.text(max(seeds) + 0.3, mean_v, f"{mean_v:.0f} MB", fontsize=10, va="center")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Peak VRAM (MB)")
    ax1.set_title("Peak GPU Memory (DeepGP + LoRA)")

    # Panel 2: Iter time
    ax2 = fig.add_subplot(2, 2, 2)
    orig_path, _ = find_results("exp2_iter_time")
    if orig_path:
        data = load_json(orig_path)
        all_times = {}
        for r in data:
            for i, t in enumerate(r.get("iter_times_s", [])):
                all_times.setdefault(i, []).append(t)
        if all_times:
            iters = sorted(all_times.keys())
            means = [np.mean(all_times[i]) for i in iters]
            stds = [np.std(all_times[i]) for i in iters]
            ax2.plot([i+1 for i in iters], means, "o-", color=MAIN_COLOR, markersize=5)
            ax2.fill_between([i+1 for i in iters],
                             [m-s for m,s in zip(means,stds)],
                             [m+s for m,s in zip(means,stds)],
                             alpha=0.15, color=MAIN_COLOR)
            overall = np.mean([t for ts in all_times.values() for t in ts])
            ax2.axhline(overall, color="black", linestyle="--", linewidth=1)
            ax2.text(max(iters)+1, overall, f"μ={overall:.2f}s", fontsize=10, va="center")
    ax2.set_xlabel("BO Iteration")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("BO Iteration Time (GP + HuggingFace)")
    ax2.grid(alpha=0.3)

    # Panel 3: Throughput
    ax3 = fig.add_subplot(2, 2, 3)
    orig_path, _ = find_results("exp3_throughput")
    if orig_path:
        data = load_json(orig_path)
        for feat, color, marker in [
            ("get_tokens", ACCENT, "s"),
            ("get_huggingface_embeddings", MAIN_COLOR, "o"),
        ]:
            pts = sorted(
                [(r["n"], r["throughput_samples_per_sec"]) for r in data
                 if r.get("featurizer") == feat and r.get("n") and r.get("throughput_samples_per_sec")],
            )
            if pts:
                ns, tps = zip(*pts)
                short = feat.replace("get_", "").replace("_embeddings", "")
                ax3.plot(ns, tps, f"{marker}-", color=color, label=short, markersize=6, linewidth=2)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Dataset Size N")
    ax3.set_ylabel("Throughput (samp/s)")
    ax3.set_title("Featurization Throughput (log-log)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, which="both")

    # Panel 4: BO Quality
    ax4 = fig.add_subplot(2, 2, 4)
    orig_path, _ = find_results("exp4_quality")
    if orig_path:
        data = load_json(orig_path)
        cp_values = {}
        for r in data:
            for cp_str, vals in (r.get("checkpoints") or {}).items():
                if vals.get("best_y") is not None:
                    cp_values.setdefault(int(cp_str), []).append(vals["best_y"])
        if cp_values:
            cps = sorted(cp_values.keys())
            means = [np.mean(cp_values[c]) for c in cps]
            stds = [np.std(cp_values[c]) for c in cps]
            ax4.errorbar(cps, means, yerr=stds, fmt="o-", color=MAIN_COLOR,
                         capsize=4, linewidth=2, markersize=8)
            ax4.text(cps[-1]+0.5, means[-1], f"{means[-1]:.1f}±{stds[-1]:.1f}",
                     fontsize=9, color=MAIN_COLOR, va="center")
    ax4.set_xlabel("BO Iteration")
    ax4.set_ylabel("Best Observed Yield")
    ax4.set_title("BO Quality (Buchwald-Hartwig)")
    ax4.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "dashboard.png"
    plt.savefig(out)
    print(f"[plots] Saved {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating figures...")
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
    plot_dashboard()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
