"""
Experiment 1 — Peak VRAM Usage
Measures maximum GPU memory allocated during featurization + BO fitting for
each seed/config. Each seed runs in its own subprocess for GPU isolation.

Run on BOTH branches, then compare with benchmark/analysis/summarize.py.
Results: benchmark/results/exp1_vram_{branch}.json
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

# Wrapper template — injected into each subprocess.
# Uses t5-base (small LLM) so it works on any GPU.
_WRAPPER = textwrap.dedent("""
import sys, os, json, time
sys.path.insert(0, {src!r})
os.chdir({root!r})
os.environ.setdefault("OPENAI_API_KEY", "x")
os.environ["WANDB_MODE"] = "disabled"

import torch
import numpy as np
import pandas as pd

torch.manual_seed({seed})
np.random.seed({seed})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Set up featurizer ──────────────────────────────────────────────────────────
from gollum.featurization.base import Featurizer
from gollum.initialization.initializers import BOInitializer
from gollum.data.module import BaseDataModule
from gollum.surrogate_models.gp import {surrogate_class} as SurrogateModel
from gollum.bo.optimizer import BotorchOptimizer
from gollum.utils.config import instantiate_class
import gpytorch

# ── Peak VRAM ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

result = {{"seed": {seed}, "config": {config_name!r}, "error": None}}

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

    # Build GP surrogate model
    train_x = dm.train_x.to(DEVICE)
    train_y = dm.train_y.to(DEVICE)
    design_space = dm.heldout_x.to(DEVICE)

{surrogate_setup}

    # Fit the model (this is where LLM backprop happens for DeepGP)
    surrogate.fit()

    # Acquisition + suggest (triggers GP posterior computation over design space)
    from gollum.utils.config import instantiate_class
    bo = BotorchOptimizer(
        design_space=design_space,
        surrogate_model_config={surrogate_config!r},
        acq_function_config={{
            "class_path": "botorch.acquisition.analytic.ExpectedImprovement",
            "init_args": {{"maximize": True}},
        }},
        batch_size=1,
    )
    bo.surrogate_model = surrogate
    bo.acquisition_function = instantiate_class(
        bo.acq_function_config,
        **bo.update_acquisition_function_params(train_y)
    )
    x_next = bo.optimize_acquisition_function(design_space)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available() else 0.0
    )
    result["peak_vram_mb"] = peak_vram_mb

except Exception as e:
    result["error"] = str(e)
    result["peak_vram_mb"] = 0.0

with open({out!r}, "w") as f:
    json.dump(result, f)
""")


# Lightweight GP surrogate (fingerprints config)
_GP_SETUP = """
covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
likelihood = gpytorch.likelihoods.GaussianLikelihood()
surrogate = SurrogateModel(
    train_x=train_x, train_y=train_y,
    covar_module=covar, likelihood=likelihood,
)
"""

_GP_CONFIG = {
    "class_path": "gollum.surrogate_models.gp.GP",
    "init_args": {
        "covar_module": {
            "class_path": "gpytorch.kernels.ScaleKernel",
            "init_args": {"base_kernel": {"class_path": "gpytorch.kernels.MaternKernel", "init_args": {"nu": 2.5}}},
        },
        "likelihood": {"class_path": "gpytorch.likelihoods.GaussianLikelihood", "init_args": {}},
    },
}

# DeepGP + LLMFeaturizer (trainable t5-base) — shows bfloat16 + grad checkpoint effect
_DEEPGP_SETUP = """
from gollum.featurization.deep import LLMFeaturizer
lora_model = LLMFeaturizer(
    model_name="t5-base",
    input_dim=768,
    projection_dim=64,
    trainable=True,
    pooling_method="average",
)
covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
likelihood = gpytorch.likelihoods.GaussianLikelihood()
mean = gpytorch.means.ConstantMean()
surrogate = SurrogateModel(
    train_x=train_x, train_y=train_y,
    covar_module=covar, likelihood=likelihood,
    mean_module=mean, finetuning_model=lora_model,
    scale_embeddings=True,
)
"""

_DEEPGP_CONFIG = {
    "class_path": "gollum.surrogate_models.gp.DeepGP",
    "init_args": {
        "covar_module": {
            "class_path": "gpytorch.kernels.ScaleKernel",
            "init_args": {"base_kernel": {"class_path": "gpytorch.kernels.MaternKernel", "init_args": {"nu": 2.5}}},
        },
        "likelihood": {"class_path": "gpytorch.likelihoods.GaussianLikelihood", "init_args": {}},
        "finetuning_model": {
            "class_path": "gollum.featurization.deep.LLMFeaturizer",
            "init_args": {"model_name": "t5-base", "input_dim": 768, "projection_dim": 64, "trainable": True},
        },
    },
}

CONFIGS = {
    "DeepGP_t5base": {
        "representation": "get_tokens",
        "model_name": "t5-base",
        "input_column": "procedure",
        "surrogate_class": "DeepGP",
        "surrogate_setup": _DEEPGP_SETUP,
        "surrogate_config": _DEEPGP_CONFIG,
    },
}


def run_one(config_name: str, cfg: dict, seed: int, timeout: int = 600) -> dict:
    src = os.path.join(GOLLUM_ROOT, "src")
    python = get_python()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    # Indent the surrogate_setup block to match try-body indentation (4 spaces)
    setup_code = cfg["surrogate_setup"].strip()
    indented_setup = "\n".join("    " + line for line in setup_code.splitlines())

    code = _WRAPPER.format(
        src=src,
        root=GOLLUM_ROOT,
        seed=seed,
        config_name=config_name,
        representation=cfg["representation"],
        model_name=cfg["model_name"],
        input_column=cfg.get("input_column", "procedure"),
        surrogate_class=cfg["surrogate_class"],
        surrogate_setup=indented_setup,
        surrogate_config=cfg["surrogate_config"],
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
        return {
            "peak_vram_mb": 0.0,
            "error": f"No output (rc={proc.returncode}): {proc.stderr[-500:]}",
        }
    except subprocess.TimeoutExpired:
        return {"peak_vram_mb": 0.0, "error": "timeout"}
    finally:
        os.unlink(script_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def run(configs: list = None, seeds: list = None):
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
        print(f"\n[Exp1] Config: {config_name}")
        for seed in seeds:
            print(f"  seed={seed} ...", flush=True)
            res = run_one(config_name, cfg, seed)
            entry = {
                "branch": branch,
                "commit_sha": commit,
                "gpu": gpu,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config_name,
                "seed": seed,
                **res,
            }
            all_results.append(entry)
            status = f"peak_vram_mb={res.get('peak_vram_mb', 0):.1f}"
            if res.get("error"):
                status += f"  [error: {res['error'][:80]}]"
            print(f"    {status}")

    out_path = os.path.join(RESULTS_DIR, f"exp1_vram_{branch}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Exp1] Saved → {out_path}")
    return all_results


if __name__ == "__main__":
    run()
