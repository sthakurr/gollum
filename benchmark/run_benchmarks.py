"""
GOLLuM Scalability Benchmark Orchestrator
==========================================
Run all 5 experiments sequentially (or selectively) and produce a summary report.

Usage:
    # Run all experiments on the current branch, then summarize:
    python benchmark/run_benchmarks.py

    # Run only specific experiments:
    python benchmark/run_benchmarks.py --experiments 1 3 5

    # Summary only (after both branches have been run):
    python benchmark/run_benchmarks.py --summary-only

Workflow:
    1. git checkout main  &&  python benchmark/run_benchmarks.py
    2. git checkout scale &&  python benchmark/run_benchmarks.py
    3. python benchmark/run_benchmarks.py --summary-only
       (or: python benchmark/analysis/summarize.py)
"""

import argparse
import subprocess
import sys
import os

EXPERIMENTS = {
    1: ("Peak VRAM Usage",               "experiments/exp1_vram.py"),
    2: ("Wall-Clock Time per Iteration", "experiments/exp2_iter_time.py"),
    3: ("Featurization Throughput",      "experiments/exp3_throughput.py"),
    4: ("GP Quality / Correctness",      "experiments/exp4_quality.py"),
    5: ("OOM Threshold",                 "experiments/exp5_oom_threshold.py"),
}

HERE = os.path.dirname(os.path.abspath(__file__))


def run_experiment(exp_id: int):
    name, script = EXPERIMENTS[exp_id]
    script_path = os.path.join(HERE, script)
    print(f"\n{'='*60}")
    print(f"Running Experiment {exp_id}: {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(HERE),
    )
    if result.returncode != 0:
        print(f"[warn] Experiment {exp_id} exited with code {result.returncode}")
    return result.returncode


def run_summary():
    summary_script = os.path.join(HERE, "analysis", "summarize.py")
    plots_script = os.path.join(HERE, "analysis", "plots.py")
    subprocess.run([sys.executable, summary_script], cwd=os.path.dirname(HERE))
    subprocess.run([sys.executable, plots_script], cwd=os.path.dirname(HERE))


def main():
    parser = argparse.ArgumentParser(description="GOLLuM benchmark runner")
    parser.add_argument(
        "--experiments", "-e",
        nargs="+", type=int,
        choices=list(EXPERIMENTS.keys()),
        default=list(EXPERIMENTS.keys()),
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip experiments and only run statistical summary + plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating figures",
    )
    args = parser.parse_args()

    if not args.summary_only:
        for exp_id in sorted(args.experiments):
            run_experiment(exp_id)

    print("\nGenerating statistical summary...")
    summary_script = os.path.join(HERE, "analysis", "summarize.py")
    subprocess.run([sys.executable, summary_script], cwd=os.path.dirname(HERE))

    if not args.no_plots:
        print("\nGenerating figures...")
        plots_script = os.path.join(HERE, "analysis", "plots.py")
        subprocess.run([sys.executable, plots_script], cwd=os.path.dirname(HERE))

    print("\nBenchmark run complete.")
    print("Results: benchmark/results/")
    print("Figures: benchmark/analysis/figures/")
    print("Summary: run 'python benchmark/analysis/summarize.py' to compare branches")


if __name__ == "__main__":
    main()
