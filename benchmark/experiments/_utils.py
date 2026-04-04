"""Shared utilities for benchmark experiment scripts."""

import os
import subprocess
import sys

GOLLUM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Known locations for the gollum conda env Python, in priority order
_GOLLUM_PYTHON_CANDIDATES = [
    os.path.join(GOLLUM_ROOT, ".venv", "bin", "python"),
    os.path.expanduser("~/miniconda3/envs/gollum/bin/python"),
    os.path.expanduser("~/anaconda3/envs/gollum/bin/python"),
    os.path.expanduser("~/miniforge3/envs/gollum/bin/python"),
]


def get_python() -> str:
    """
    Return the Python executable that can import gollum.
    Priority: current sys.executable → gollum conda env → fallback.
    """
    src_path = os.path.join(GOLLUM_ROOT, "src")

    # Check if current Python can import gollum (after sys.path fix)
    probe = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, {src_path!r}); "
            "import os; os.environ.setdefault('OPENAI_API_KEY', 'x'); "
            "from gollum.data.module import BaseDataModule; print('ok')",
        ],
        capture_output=True, text=True,
    )
    if probe.returncode == 0 and "ok" in probe.stdout:
        return sys.executable

    # Try known gollum env locations
    for candidate in _GOLLUM_PYTHON_CANDIDATES:
        if not os.path.isfile(candidate):
            continue
        probe = subprocess.run(
            [
                candidate, "-c",
                f"import sys; sys.path.insert(0, {src_path!r}); "
                "import os; os.environ.setdefault('OPENAI_API_KEY', 'x'); "
                "from gollum.data.module import BaseDataModule; print('ok')",
            ],
            capture_output=True, text=True,
        )
        if probe.returncode == 0 and "ok" in probe.stdout:
            return candidate

    # Last resort: current Python (will fail with a clear error message)
    return sys.executable


def get_branch(root: str = GOLLUM_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_commit(root: str = GOLLUM_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"
