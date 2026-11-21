"""Provenance block attached to every results file: commit, dirty flag, config, seed, platform."""
from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _torch_threads():
    try:
        import torch

        return torch.get_num_threads()
    except ImportError:
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def provenance(config: dict | None = None, seed: int | None = None) -> dict:
    import numpy

    versions = {"python": platform.python_version(), "numpy": numpy.__version__}
    for mod in ("scipy", "pandas", "wfdb", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "torch_threads": _torch_threads(),
        "versions": versions,
        "seed": seed,
        "config": config,
    }
