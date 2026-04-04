"""
Experiment 5 — OOM Threshold Test (Scalability Boundary)
Each subprocess tries featurization + 1 BO iteration on a synthetic dataset of
size N. OOM is detected from the subprocess exit code / stderr.

Scalability multiplier = N_optimized_threshold / N_original_threshold

Results: benchmark/results/exp5_oom_{branch}.json
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import torch
from _utils import get_python, get_branch, get_commit, get_gpu_name, GOLLUM_ROOT

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_CSV = os.path.join(GOLLUM_ROOT, "data", "buchwald-hartwig", "bh_reaction_1.csv")

N_VALUES = [50, 100, 200, 400, 800, 1600]
N_REPEATS = 3

_WRAPPER = textwrap.dedent("""
import sys, os, json, warnings
sys.path.insert(0, {src!r})
os.chdir({root!r})
os.environ.setdefault("OPENAI_API_KEY", "x")
os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
np.random.seed({seed})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Build synthetic dataset of size N by repeat-sampling the real data
df = pd.read_csv({data_csv!r})
rng = np.random.default_rng({seed})
idx = rng.integers(0, len(df), size={n})
synthetic = df.iloc[idx].reset_index(drop=True)

synth_path = {synth_csv!r}
synthetic.to_csv(synth_path, index=False)

result = {{"n": {n}, "seed": {seed}, "outcome": "unknown", "error": None}}

try:
    from gollum.featurization.base import Featurizer
    from gollum.initialization.initializers import BOInitializer
    from gollum.data.module import BaseDataModule
    from gollum.bo.optimizer import BotorchOptimizer
    import gpytorch

    featurizer = Featurizer(
        representation="get_huggingface_embeddings",
        model_name="t5-base",
    )
    initializer = BOInitializer(method="true_random", n_clusters=min(10, {n} - 1), seed={seed})
    dm = BaseDataModule(
        data_path=synth_path,
        input_column="procedure",
        target_column="objective",
        maximize=True,
        featurizer=featurizer,
        initializer=initializer,
        normalize_input="original",
    )

    surrogate_config = {{
        "class_path": "gollum.surrogate_models.gp.GP",
        "init_args": {{
            "covar_module": {{
                "class_path": "gpytorch.kernels.ScaleKernel",
                "init_args": {{
                    "base_kernel": {{"class_path": "gpytorch.kernels.MaternKernel",
                                    "init_args": {{"nu": 2.5}}}},
                }},
            }},
            "likelihood": {{"class_path": "gpytorch.likelihoods.GaussianLikelihood", "init_args": {{}}}},
        }},
    }}
    acq_config = {{
        "class_path": "botorch.acquisition.analytic.ExpectedImprovement",
        "init_args": {{"maximize": True}},
    }}
    bo = BotorchOptimizer(
        design_space=dm.heldout_x.to(DEVICE),
        surrogate_model_config=surrogate_config,
        acq_function_config=acq_config,
        batch_size=1,
    )

    train_x = dm.train_x.clone().to(DEVICE)
    train_y = dm.train_y.clone().to(DEVICE)
    design_space = dm.heldout_x.clone().to(DEVICE)
    _ = bo.suggest_next_experiments(train_x, train_y, design_space)

    result["outcome"] = "success"

except torch.cuda.OutOfMemoryError as e:
    result["outcome"] = "oom"
    result["error"] = str(e)
except Exception as e:
    import traceback
    result["outcome"] = "error"
    result["error"] = traceback.format_exc()
finally:
    if os.path.exists(synth_path):
        os.unlink(synth_path)

with open({out!r}, "w") as f:
    json.dump(result, f)
""")


def run_one(n: int, seed: int, timeout: int = 600) -> dict:
    src = os.path.join(GOLLUM_ROOT, "src")
    python = get_python()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        synth_csv = f.name

    code = _WRAPPER.format(
        src=src,
        root=GOLLUM_ROOT,
        seed=seed,
        n=n,
        data_csv=DATA_CSV,
        synth_csv=synth_csv,
        out=out_path,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.run(
            [python, script_path],
            capture_output=True, text=True, timeout=timeout,
        )
        # Also detect OOM from stderr even if json wasn't written
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path) as f:
                result = json.load(f)
        else:
            combined = proc.stdout + proc.stderr
            oom_signals = ["CUDA out of memory", "OutOfMemoryError", "out of memory"]
            if any(s in combined for s in oom_signals):
                result = {"outcome": "oom", "error": "detected from stderr"}
            else:
                result = {"outcome": f"error_{proc.returncode}", "error": proc.stderr[-300:]}
        return result
    except subprocess.TimeoutExpired:
        return {"outcome": "timeout", "error": "timeout"}
    finally:
        os.unlink(script_path)
        for p in [out_path, synth_csv]:
            if os.path.exists(p):
                os.unlink(p)


def run(n_values: list = None, n_repeats: int = N_REPEATS):
    if n_values is None:
        n_values = N_VALUES

    branch = get_branch()
    commit = get_commit()
    gpu = get_gpu_name()
    all_results = []

    print(f"\n[Exp5] OOM threshold — N values={n_values}")
    for n in n_values:
        outcomes = []
        for rep in range(n_repeats):
            print(f"  N={n:5d}  rep={rep} ...", flush=True)
            res = run_one(n, seed=rep + 1)
            outcomes.append(res.get("outcome", "unknown"))
            print(f"    outcome={outcomes[-1]}")

        success_rate = outcomes.count("success") / n_repeats
        entry = {
            "branch": branch,
            "commit_sha": commit,
            "gpu": gpu,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n": n,
            "n_repeats": n_repeats,
            "outcomes": outcomes,
            "success_rate": success_rate,
        }
        all_results.append(entry)

    # OOM threshold = largest N with 100% success
    threshold = 0
    for entry in all_results:
        if entry["success_rate"] == 1.0:
            threshold = entry["n"]

    summary = {
        "branch": branch,
        "commit_sha": commit,
        "oom_threshold_n": threshold,
        "summary": True,
    }
    all_results.append(summary)
    print(f"\n  OOM threshold (branch='{branch}'): N={threshold}")

    out_path = os.path.join(RESULTS_DIR, f"exp5_oom_{branch}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Exp5] Saved → {out_path}")
    return all_results


if __name__ == "__main__":
    run()
