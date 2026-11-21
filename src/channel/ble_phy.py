"""BLE LE 1M PHY: bit-error and packet-error models versus SNR.

Two BER models are provided; the choice is a configuration switch, recorded with every run.

* ``noncoherent_fsk`` (analytic): BER = 0.5 exp(-gamma_b / 2), the textbook result for
  noncoherent orthogonal binary FSK in AWGN (reference to be pinned in WP5). BLE GFSK with
  modulation index 0.5 is not orthogonal FSK, so this is an approximation.
* ``gfsk_mc`` (link-level Monte Carlo): baseband GFSK (BT = 0.5, h = 0.5) through AWGN,
  detected by a limiter-discriminator with integrate-and-dump at ideal timing. The BER curve
  is generated once per seed by ``gfsk_ber_curve`` and interpolated in log domain.

SNR convention: gamma_b = Eb/N0. With a noise bandwidth equal to the 1 Mb/s symbol rate,
Eb/N0 equals the in-band SNR; a different receiver bandwidth B shifts it by 10 log10(B/Rb).

Packet error: bits are assumed independent, PER = 1 - (1 - BER)^n, counted over the bits
whose corruption causes loss (access address, PDU header, payload, CRC). The independence
assumption is optimistic for discriminator receivers, whose errors cluster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fading import db_to_lin

BIT_RATE = 1e6  # LE 1M PHY


@dataclass(frozen=True)
class BleFrame:
    """LE uncoded PHY packet layout in bits (Bluetooth Core Spec, Vol 6 Part B, 2.1)."""
    preamble_bits: int = 8
    access_address_bits: int = 32
    header_bits: int = 16
    crc_bits: int = 24
    mic_bits: int = 0           # link-layer encryption off: confidentiality is in PQ-Lite
    max_payload_bytes: int = 251  # with LE Data Length Extension

    def protected_bits(self, payload_bytes) -> np.ndarray:
        pb = np.asarray(payload_bytes)
        if np.any(pb > self.max_payload_bytes) or np.any(pb < 0):
            raise ValueError(f"payload must be in [0, {self.max_payload_bytes}] bytes")
        return self.access_address_bits + self.header_bits + 8 * pb + self.mic_bits + self.crc_bits

    def airtime_s(self, payload_bytes) -> np.ndarray:
        return (self.preamble_bits + self.protected_bits(payload_bytes)) / BIT_RATE


def ber_noncoherent_fsk(ebn0_db) -> np.ndarray:
    return 0.5 * np.exp(-db_to_lin(ebn0_db) / 2.0)


def _gaussian_taps(bt: float, sps: int, span: int) -> np.ndarray:
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    alpha = np.sqrt(np.log(2.0) / 2.0) / bt
    g = (np.sqrt(np.pi) / alpha) * np.exp(-(np.pi * t / alpha) ** 2)
    return g / g.sum()


def gfsk_ber_curve(ebn0_db: np.ndarray, rng: np.random.Generator, n_bits: int = 200_000,
                   bt: float = 0.5, h: float = 0.5, sps: int = 8,
                   prefilter: bool = True) -> np.ndarray:
    """Monte Carlo BER of GFSK with limiter-discriminator detection at each Eb/N0 point."""
    taps = _gaussian_taps(bt, sps, span=4)
    bits = rng.integers(0, 2, n_bits)
    nrz = np.repeat(2.0 * bits - 1.0, sps)
    freq = np.convolve(nrz, taps, mode="same")
    phase = np.pi * h * np.cumsum(freq) / sps
    s = np.exp(1j * phase)  # unit power, Es = sps samples per bit at unit power
    out = []
    for e in np.atleast_1d(ebn0_db):
        n0 = sps / db_to_lin(e)  # Eb = sps (unit-power samples, sps per bit)
        noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size))
        r = s + noise
        if prefilter:  # boxcar pre-detection filter, noise bandwidth = bit rate
            r = np.convolve(r, np.ones(sps) / sps, mode="same")
        dphi = np.angle(r[1:] * np.conj(r[:-1]))
        dphi = np.concatenate([[0.0], dphi])
        stat = dphi.reshape(n_bits, sps).sum(axis=1)
        dec = (stat > 0).astype(int)
        guard = 4  # skip edge bits affected by filter start-up
        out.append(np.mean(dec[guard:-guard] != bits[guard:-guard]))
    return np.asarray(out)


class BerModel:
    """BER(Eb/N0 dB) from the analytic formula or an interpolated Monte Carlo table."""

    def __init__(self, kind: str = "noncoherent_fsk", table_db=None, table_ber=None,
                 floor: float = 1e-9):
        if kind not in ("noncoherent_fsk", "gfsk_mc"):
            raise ValueError(f"unknown BER model {kind}")
        if kind == "gfsk_mc" and (table_db is None or table_ber is None):
            raise ValueError("gfsk_mc needs a BER table (see gfsk_ber_curve)")
        self.kind, self.floor = kind, floor
        if table_db is not None:
            self.table_db = np.asarray(table_db, dtype=float)
            self.log_ber = np.log10(np.maximum(np.asarray(table_ber, dtype=float), floor))

    def __call__(self, ebn0_db) -> np.ndarray:
        if self.kind == "noncoherent_fsk":
            return np.maximum(ber_noncoherent_fsk(ebn0_db), 0.0)
        x = np.asarray(ebn0_db, dtype=float)
        # linear extrapolation in log-BER beyond the table, clipped to [floor, 0.5]
        slope_hi = (self.log_ber[-1] - self.log_ber[-2]) / (self.table_db[-1] - self.table_db[-2])
        y = np.interp(x, self.table_db, self.log_ber)
        y = np.where(x > self.table_db[-1], self.log_ber[-1] + slope_hi * (x - self.table_db[-1]), y)
        return np.clip(10.0 ** y, 0.0, 0.5)


def packet_error_rate(ebn0_db, payload_bytes, ber_model: BerModel, frame: BleFrame = BleFrame()):
    ber = ber_model(ebn0_db)
    n = frame.protected_bits(payload_bytes)
    return 1.0 - np.exp(n * np.log1p(-np.minimum(ber, 0.5)))
