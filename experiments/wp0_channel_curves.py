"""WP0 channel model check: GFSK Monte Carlo BER table and BLE PER vs Eb/N0 (AWGN and Rician).

    python experiments/wp0_channel_curves.py seed=0

Outputs: results/wp0/gfsk_ber_seed<seed>.json (BER table consumed later by BerModel('gfsk_mc')),
figures/wp0_ble_per.{pdf,png}. Model outputs only; no measured quantity is involved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.channel.ble_phy import BerModel, BleFrame, ber_noncoherent_fsk, gfsk_ber_curve, packet_error_rate  # noqa: E402
from src.channel.link import expected_delivery  # noqa: E402
from src.utils import plotting  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402
from src.utils.seed import pin_threads, set_seed  # noqa: E402


@hydra.main(config_path="../configs", config_name="wp0_channel", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rng = set_seed(cfg.seed)
    pin_threads(1)
    g = cfg.channel.gfsk_mc
    grid = np.asarray(g.ebn0_grid_db, dtype=float)
    ber = gfsk_ber_curve(grid, rng, n_bits=g.n_bits, bt=g.bt, h=g.h, sps=g.sps, prefilter=g.prefilter)
    frame = BleFrame(**cfg.channel.frame)
    out = {"provenance": provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed),
           "ebn0_db": grid.tolist(), "ber_gfsk_mc": ber.tolist(),
           "ber_noncoherent_fsk": ber_noncoherent_fsk(grid).tolist()}
    res = REPO / "results" / "wp0"
    res.mkdir(parents=True, exist_ok=True)

    model = BerModel("gfsk_mc", table_db=grid, table_ber=ber)
    snr = np.arange(0, 20.5, 0.5)
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(plotting.IEEE_COL_IN, 2.4))
    curves = {}
    for i, pb in enumerate(cfg.payloads):
        per_awgn = packet_error_rate(snr, pb, model, frame)
        ax.semilogy(snr, np.maximum(per_awgn, 1e-6), color=plotting.PALETTE[i], ls="-",
                    label=f"{pb} B, AWGN")
        per_ric = np.array([1 - expected_delivery(s, pb, cfg.channel.k_factor, model, frame,
                                                  n_mc=20_000, rng=np.random.default_rng(cfg.seed))
                            for s in snr])
        ax.semilogy(snr, np.maximum(per_ric, 1e-6), color=plotting.PALETTE[i], ls="--",
                    label=f"{pb} B, Rician K={cfg.channel.k_factor:g}")
        curves[str(pb)] = {"awgn": per_awgn.tolist(), "rician": per_ric.tolist()}
    ax.set_xlabel(r"Average $E_b/N_0$ (dB)")
    ax.set_ylabel("Packet error rate")
    ax.set_ylim(1e-5, 1.5)
    ax.legend(ncol=1, loc="lower left")
    plotting.save(fig, REPO / "figures" / "wp0_ble_per")
    out["per_snr_db"] = snr.tolist()
    out["per_curves"] = curves
    (res / f"gfsk_ber_seed{cfg.seed}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"ebn0_db": out["ebn0_db"], "ber_gfsk_mc": out["ber_gfsk_mc"]}))


if __name__ == "__main__":
    main()
