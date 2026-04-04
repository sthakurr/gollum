"""
Experiment 3 — Featurization Throughput Scaling
Directly imports gollum (no subprocess needed — purely in-process timing).
Measures samples/second for get_tokens and get_huggingface_embeddings across N values.
Fits a power-law exponent b via bootstrap; b ≈ 0 = good, b < 0 = degradation.

Results: benchmark/results/exp3_throughput_{branch}.json
"""

import json
import os
import random
import subprocess
import sys
import time

# Add src to path so gollum is importable without pip install
GOLLUM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(GOLLUM_ROOT, "src"))
os.chdir(GOLLUM_ROOT)
# Stub OPENAI_API_KEY so text.py doesn't crash on import
os.environ.setdefault("OPENAI_API_KEY", "x")

import numpy as np
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_CSV = os.path.join(GOLLUM_ROOT, "data", "buchwald-hartwig", "bh_reaction_1.csv")

N_VALUES = [50, 100, 200, 500, 1000]
N_REPEATS = 3
MODEL_NAME = "t5-base"


def get_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=GOLLUM_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=GOLLUM_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def load_real_texts(n: int, seed: int = 42) -> list:
    """Sample n procedure texts from the BH dataset (with replacement if needed)."""
    import pandas as pd
    df = pd.read_csv(DATA_CSV)
    texts = df["procedure"].dropna().tolist()
    rng = random.Random(seed)
    return [rng.choice(texts) for _ in range(n)]


def time_fn(fn, *args, n_repeats=N_REPEATS, warmup=1) -> list:
    """Time fn(*args) n_repeats times after warmup calls. Returns list of wall times."""
    for _ in range(warmup):
        try:
            fn(*args)
        except Exception:
            pass
    timings = []
    for _ in range(n_repeats):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        fn(*args)
        timings.append(time.perf_counter() - t0)
    return timings


def fit_power_law_bootstrap(ns, throughputs, n_boot: int = 500):
    """Fit throughput ~ a * N^b via log-linear OLS. Bootstrap 95% CI on b."""
    log_n = np.log(np.array(ns, dtype=float))
    log_t = np.log(np.array(throughputs, dtype=float))
    b_obs = np.polyfit(log_n, log_t, 1)[0]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ns), size=len(ns))
        boots.append(np.polyfit(log_n[idx], log_t[idx], 1)[0])
    boots.sort()
    return float(b_obs), float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot)])


def run(
    model_name: str = MODEL_NAME,
    n_values: list = None,
    n_repeats: int = N_REPEATS,
    seed: int = 42,
):
    if n_values is None:
        n_values = N_VALUES

    from gollum.featurization.text import get_tokens, get_huggingface_embeddings

    branch = get_branch()
    commit = get_commit()
    gpu = get_gpu_name()

    featurizers = {
        "get_tokens": lambda texts: get_tokens(texts, model_name=model_name),
        "get_huggingface_embeddings": lambda texts: get_huggingface_embeddings(
            texts, model_name=model_name, batch_size=32
        ),
    }

    all_results = []
    for feat_name, feat_fn in featurizers.items():
        print(f"\n[Exp3] Featurizer: {feat_name}")
        ns_list, tp_list = [], []

        for n in n_values:
            texts = load_real_texts(n, seed=seed)
            try:
                timings = time_fn(feat_fn, texts, n_repeats=n_repeats)
                mean_t = float(np.mean(timings))
                throughput = n / mean_t
            except Exception as e:
                print(f"  N={n}: ERROR {e}")
                timings = []
                mean_t = None
                throughput = None

            ns_list.append(n)
            tp_list.append(throughput)

            entry = {
                "branch": branch,
                "commit_sha": commit,
                "gpu": gpu,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "featurizer": feat_name,
                "model_name": model_name,
                "n": n,
                "n_repeats": n_repeats,
                "times_s": timings,
                "mean_time_s": mean_t,
                "throughput_samples_per_sec": throughput,
            }
            all_results.append(entry)
            if throughput is not None:
                print(f"  N={n:5d}  throughput={throughput:.1f} samp/s")

        # Power-law fit (skip if any None throughputs)
        valid = [(n, t) for n, t in zip(ns_list, tp_list) if t is not None and t > 0]
        if len(valid) >= 3:
            ns_v, tp_v = zip(*valid)
            b, b_lo, b_hi = fit_power_law_bootstrap(list(ns_v), list(tp_v))
            print(f"  Power-law exponent b={b:.3f} (95% CI [{b_lo:.3f}, {b_hi:.3f}])")
            all_results.append({
                "branch": branch,
                "commit_sha": commit,
                "featurizer": feat_name,
                "model_name": model_name,
                "summary": "power_law",
                "b": b,
                "b_ci_low": b_lo,
                "b_ci_high": b_hi,
            })

    out_path = os.path.join(RESULTS_DIR, f"exp3_throughput_{branch}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Exp3] Saved → {out_path}")
    return all_results


if __name__ == "__main__":
    run()
