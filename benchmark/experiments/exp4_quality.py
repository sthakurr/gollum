"""
Experiment 4 — BO Quality (Numerical Correctness Check)
Runs the full BO loop without wandb (WANDB_MODE=disabled) and captures
best observed yield at checkpoints. Null hypothesis: optimizations don't
change BO performance.

Results: benchmark/results/exp4_quality_{branch}.json
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

CHECKPOINTS = [5, 10, 20, 30]

_WRAPPER = textwrap.dedent("""
import sys, os, json, warnings
sys.path.insert(0, {src!r})
os.chdir({root!r})
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("OPENAI_API_KEY", "x")
warnings.filterwarnings("ignore")

import torch
import numpy as np
torch.manual_seed({seed})
np.random.seed({seed})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS = {checkpoints!r}

from gollum.featurization.base import Featurizer
from gollum.initialization.initializers import BOInitializer
from gollum.data.module import BaseDataModule
from gollum.bo.optimizer import BotorchOptimizer
import gpytorch

result = {{"seed": {seed}, "checkpoints": {{}}, "error": None}}

try:
    featurizer = Featurizer(representation="get_huggingface_embeddings", model_name="t5-base")
    initializer = BOInitializer(method="true_random", n_clusters=10, seed={seed})
    dm = BaseDataModule(
        data_path={data_csv!r},
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

    for i in range({n_iters}):
        train_x = dm.train_x.clone().to(DEVICE)
        train_y = dm.train_y.clone().to(DEVICE)
        design_space = dm.heldout_x.clone().to(DEVICE)

        x_next_list = bo.suggest_next_experiments(train_x, train_y, design_space)
        x_next = torch.stack(x_next_list)

        matches = (design_space.unsqueeze(0) == x_next).all(dim=-1)
        indices = matches.nonzero(as_tuple=True)[1].cpu()
        if len(indices) == 0:
            break

        dm.train_x = torch.cat([dm.train_x, dm.heldout_x[indices]])
        dm.train_y = torch.cat([dm.train_y, dm.heldout_y[indices]])
        keep = torch.ones(dm.heldout_x.size(0), dtype=torch.bool)
        keep[indices] = False
        dm.heldout_x = dm.heldout_x[keep]
        dm.heldout_y = dm.heldout_y[keep]

        step = i + 1
        if step in CHECKPOINTS:
            best_y = float(dm.train_y.max().item())
            result["checkpoints"][str(step)] = {{"best_y": best_y}}

        if dm.heldout_x.size(0) == 0:
            break

except Exception as e:
    import traceback
    result["error"] = traceback.format_exc()

with open({out!r}, "w") as f:
    json.dump(result, f)
""")


def run_one(seed: int, n_iters: int = 30, timeout: int = 900) -> dict:
    src = os.path.join(GOLLUM_ROOT, "src")
    python = get_python()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    code = _WRAPPER.format(
        src=src,
        root=GOLLUM_ROOT,
        seed=seed,
        checkpoints=CHECKPOINTS,
        n_iters=n_iters,
        data_csv=DATA_CSV,
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
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path) as f:
                return json.load(f)
        return {"checkpoints": {}, "error": f"No output (rc={proc.returncode}): {proc.stderr[-500:]}"}
    except subprocess.TimeoutExpired:
        return {"checkpoints": {}, "error": "timeout"}
    finally:
        os.unlink(script_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def run(seeds: list = None, n_iters: int = 30):
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    branch = get_branch()
    commit = get_commit()
    gpu = get_gpu_name()
    all_results = []

    print(f"\n[Exp4] BO quality — {n_iters} iters, seeds={seeds}")
    for seed in seeds:
        print(f"  seed={seed} ...", flush=True)
        res = run_one(seed, n_iters)
        entry = {
            "branch": branch,
            "commit_sha": commit,
            "gpu": gpu,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": seed,
            "n_iters": n_iters,
            **res,
        }
        all_results.append(entry)
        if res.get("checkpoints"):
            cps = "  ".join(
                f"@{cp}: {v['best_y']:.4f}" for cp, v in sorted(res["checkpoints"].items())
            )
            print(f"    {cps}")
        elif res.get("error"):
            print(f"    error: {str(res['error'])[:100]}")

    out_path = os.path.join(RESULTS_DIR, f"exp4_quality_{branch}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Exp4] Saved → {out_path}")
    return all_results


if __name__ == "__main__":
    run()
