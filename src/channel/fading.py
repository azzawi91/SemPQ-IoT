"""Block Rician fading.

The complex gain is h = sqrt(K/(K+1)) e^{j phi} + sqrt(1/(K+1)) g, with g ~ CN(0, 1) and
phi ~ U[0, 2 pi), so E|h|^2 = 1 for every Rician factor K (K = 0 is Rayleigh). The gain is
held constant over a block of ``block_len`` consecutive slots and drawn independently
across blocks. The instantaneous SNR is the average SNR scaled by |h|^2.

Distribution used by the unit tests: 2 (K + 1) |h|^2 is noncentral chi-square with
2 degrees of freedom and noncentrality 2K.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rician_power_gains(n_blocks: int, k_factor: float, rng: np.random.Generator) -> np.ndarray:
    """|h|^2 for ``n_blocks`` independent blocks (unit mean)."""
    if k_factor < 0:
        raise ValueError("Rician K must be >= 0")
    k = float(k_factor)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_blocks)
    los = np.sqrt(k / (k + 1.0)) * np.exp(1j * phi)
    g = (rng.standard_normal(n_blocks) + 1j * rng.standard_normal(n_blocks)) / np.sqrt(2.0)
    h = los + np.sqrt(1.0 / (k + 1.0)) * g
    return np.abs(h) ** 2


def block_fading_trace(n_slots: int, block_len: int, k_factor: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Per-slot |h|^2, constant within blocks of ``block_len`` slots."""
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    n_blocks = -(-n_slots // block_len)
    return np.repeat(rician_power_gains(n_blocks, k_factor, rng), block_len)[:n_slots]


def power_gain_cdf(x: np.ndarray, k_factor: float) -> np.ndarray:
    """Analytic CDF of |h|^2 under unit-mean Rician fading."""
    k = float(k_factor)
    return stats.ncx2.cdf(2.0 * (k + 1.0) * np.asarray(x), df=2, nc=2.0 * k)


def db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)


def lin_to_db(lin):
    return 10.0 * np.log10(np.asarray(lin, dtype=np.float64))
