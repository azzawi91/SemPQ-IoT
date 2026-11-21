"""Seed locking and thread pinning (ml-paper-pipeline section 1 and section 13).

Call ``set_seed`` at the very top of every entry point, before any tensor is created.
"""
from __future__ import annotations

import os
import random

import numpy as np


def pin_threads(n: int = 1) -> None:
    """Pin BLAS/OpenMP thread counts; required for bitwise-reproducible CPU runs."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:  # torch is optional for WP0 data and channel code
        pass


def set_seed(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) torch. Returns a NumPy Generator for local use."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=False)
    except ImportError:
        pass
    return np.random.default_rng(seed)
