"""
Experiment 2 — Wall-Clock Time per BO Iteration
Each subprocess runs N_ITERS BO iterations, monkey-patches suggest_next_experiments
to record per-iteration timing, and writes a JSON result.

fp config = negative control (fingerprints only — LLM changes shouldn't affect it).
llm config = uses get_huggingface_embeddings (affected by model caching + pre-alloc).

Results: benchmark/results/exp2_iter_time_{branch}.json
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

_WRAPPER = textwrap.dedent("""
import sys, os, json, time
sys.path.insert(0, {src!r})
os.chdir({root!r})
os.environ.setdefault("OPENAI_API_KEY", "x")
os.environ["WANDB_MODE"] = "disabled"

import torch
import numpy as np
torch.manual_seed({seed})
np.random.seed({seed})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from gollum.featurization.base import Featurizer
from gollum.initialization.initializers import BOInitializer
from gollum.data.module import BaseDataModule
from gollum.surrogate_models.gp import GP
from gollum.bo.optimizer import BotorchOptimizer
import gpytorch

result = {{"seed": {seed}, "config": {config_name!r}, "iter_times_s": [], "error": None}}

try:
    featurizer = Featurizer(representation={representation!r}, model_name={model_name!r})
    initializer = BOInitializer(method="true_random", n_clusters=10, seed={seed})
    dm = BaseDataModule(
        data_path={data_csv!r},
        input_column={input_column!r},
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
            "likelihood": {{"class_path": "gpytorch.likelihoods.GaussianLikelihood",
                           "init_args": {{}}}},
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

        t0 = time.perf_counter()
        x_next_list = bo.suggest_next_experiments(train_x, train_y, design_space)
        iter_time = time.perf_counter() - t0
        result["iter_times_s"].append(iter_time)

        # Find candidate in heldout and update dm
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

        if dm.heldout_x.size(0) == 0:
            break

    import statistics
    if result["iter_times_s"]:
        result["mean_iter_time_s"] = statistics.mean(result["iter_times_s"])
        result["std_iter_time_s"] = (
            statistics.stdev(result["iter_times_s"])
            if len(result["iter_times_s"]) > 1 else 0.0
        )

except Exception as e:
    import traceback
    result["error"] = traceback.format_exc()

with open({out!r}, "w") as f:
    json.dump(result, f)
""")

CONFIGS = {
    "GP_huggingface": {
        "representation": "get_huggingface_embeddings",
        "model_name": "t5-base",
        "input_column": "procedure",
        "desc": "HuggingFace embeddings + GP (model caching + pre-alloc path)",
    },
}


def run_one(config_name: str, cfg: dict, seed: int, n_iters: int = 20, timeout: int = 600) -> dict:
    src = os.path.join(GOLLUM_ROOT, "src")
    python = get_python()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    code = _WRAPPER.format(
        src=src,
        root=GOLLUM_ROOT,
        seed=seed,
        config_name=config_name,
        representation=cfg["representation"],
        model_name=cfg["model_name"],
        input_column=cfg.get("input_column", "procedure"),
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
        return {"iter_times_s": [], "error": f"No output (rc={proc.returncode}): {proc.stderr[-500:]}"}
    except subprocess.TimeoutExpired:
        return {"iter_times_s": [], "error": "timeout"}
    finally:
        os.unlink(script_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def run(configs: list = None, seeds: list = None, n_iters: int = 20):
    if configs is None:
        configs = list(CONFIGS.keys())
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    branch = get_branch()
    commit = get_commit()
    gpu = get_gpu_name()
    all_results = []

    for config_name in configs:
        cfg = CONFIGS[config_name]
        print(f"\n[Exp2] Config: {config_name}  ({cfg['desc']})")
        for seed in seeds:
            print(f"  seed={seed} ...", flush=True)
            res = run_one(config_name, cfg, seed, n_iters)
            entry = {
                "branch": branch,
                "commit_sha": commit,
                "gpu": gpu,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config_name,
                "seed": seed,
                "n_iters": n_iters,
                **res,
            }
            all_results.append(entry)
            if res.get("mean_iter_time_s") is not None:
                print(f"    mean={res['mean_iter_time_s']:.3f}s  std={res.get('std_iter_time_s', 0):.3f}s")
            elif res.get("error"):
                print(f"    error: {res['error'][:100]}")

    out_path = os.path.join(RESULTS_DIR, f"exp2_iter_time_{branch}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Exp2] Saved → {out_path}")
    return all_results


if __name__ == "__main__":
    run()
