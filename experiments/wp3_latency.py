"""End-to-end beat latency model (decision 2026-10-13, external review point 6).

The L3 "delay" d(t) is a link-delay proxy (loss and handshake time per slot). This script reports the latency of
every delivered DS2 beat in the end-to-end trace (wp3_e2e: one DS2 beat per slot, seed k = encoder seed k and trace
seed k), from the R peak of the beat to the arrival of its record at the gateway:

  L = max(post-RR interval, window tail)   wait for the next R peak (post-RR feature) and the window end
    + t_enc + t_rec                         encoder inference and record protection on the node (cost model)
    + t_access                              wait for the next connection event, uniform on [0, CI)
    + t_event + t_air                       connection-event overhead (ramp-ups, central's packet) + record air time
    + 1[session switch] T_hs(s)             handshake latency of the new tier (WP2 link simulation)

Lost records (no link-layer retransmission is modelled, as in the scheduler) are counted separately. Gateway
inference time is excluded (gateway hardware not specified). CI is a deployment parameter: 7.5 ms (minimum
LE connection interval), 100 ms and 1 s are reported. A MODEL, not a measurement.

    python experiments/wp3_latency.py inputs_path=results/wp3/inputs.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402

CI_S = (0.0075, 0.1, 1.0)
POLICIES = ("DPP (proposed)", "Myopic greedy", "Fixed max. security", "Combined: 8 B, tier 2, DPP over power",
            "Combined: 8 B, tier 2, max. power")


def wp1_settings():
    """Window and classes of the WP1 runs (configs/data/mitbih.yaml, configs/wp1_mitbih.yaml)."""
    w = OmegaConf.load(REPO / "configs" / "data" / "mitbih.yaml").window
    classes = list(OmegaConf.load(REPO / "configs" / "wp1_mitbih.yaml").classes)
    return w, classes


def ds2_post_rr(w, classes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unquantised post-RR interval (s) of every DS2 beat in the order of the WP1 per-beat predictions."""
    import os

    from src.data import mitbih, physionet
    from src.data.datasets import mitbih_split_beats

    raw = Path(os.environ.get("SEMPQ_DATA_RAW", REPO / "data" / "raw"))
    root = physionet.fetch("mitdb", raw, REPO / "data" / "manifests", source="s3")
    beats = mitbih_split_beats(root, "test", w.pre, w.post, include_q=True)
    keep_idx = [mitbih.CLASS_INDEX[c] for c in classes]
    sel = np.isin(beats.y, keep_idx)
    remap = {old: new for new, old in enumerate(keep_idx)}
    return beats.rr[sel, 1].astype(float), np.vectorize(remap.get)(beats.y[sel]), beats.record[sel]


@hydra.main(config_path="../configs", config_name="wp3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.channel import ble_conn
    from src.channel.ble_phy import BerModel, gfsk_ber_curve
    from src.l3_scheduler import policies as pol
    from src.l3_scheduler.model import SystemModel
    from src.l3_scheduler.sim import run, traces
    from src.utils.provenance import provenance
    from src.utils.seed import pin_threads, set_seed

    set_seed(cfg.seed)
    pin_threads(1)
    w, classes = wp1_settings()
    post_rr, y_ref, rec_ref = ds2_post_rr(w, classes)
    wait = np.maximum(post_rr, w.post / 360.0)

    grid = np.asarray(cfg.ber.ebn0_grid_db, float)
    ber = BerModel("gfsk_mc", grid, gfsk_ber_curve(grid, np.random.default_rng(cfg.ber.seed), n_bits=cfg.ber.n_bits))
    m = SystemModel.build(cfg, ber)
    with open_dict(cfg):
        cfg.E_bar = float(m.e_enc + cfg.E_bar_extra)
    import yaml
    const = yaml.safe_load((REPO / "configs" / "constants.yaml").read_text())
    inp = yaml.safe_load((Path(cfg.inputs_path) if Path(cfg.inputs_path).is_absolute()
                          else REPO / cfg.inputs_path).read_text())
    t_enc = (inp.get("encoder_macs") or 0) * const["nn_inference"]["cycles_per_mac"]["value"] / m.f_hz
    t_rec = np.array([lv["e_rec"] / (m.i_run_a * m.v) for lv in m.levels])          # cycles / f
    t_air = np.array([lv["airtime"] for lv in m.levels])
    t_evt = ble_conn.event_energy_j(1, "nominal", const)["t_event_overhead_s"]
    t_hs = {s: m.tiers[s]["t_hs"] for s in m.tiers}

    rows = []
    for f in sorted((REPO / "results" / "wp1" / "semantic_mitbih").glob("seed*_pred.npz")):
        seed = int(f.stem.split("_")[0][4:])
        d = np.load(f)
        assert np.array_equal(d["y"], y_ref) and np.array_equal(d["record"], rec_ref), "DS2 beat order differs"
        with open_dict(cfg):
            cfg.slots = int(len(y_ref))
        tr = traces(cfg, np.random.default_rng(seed))
        tab = m.tables_all(tr["gain"], tr["outage"])
        bp = [d[lv["level"]] for lv in m.levels]
        for p in pol.make_policies(cfg, len(m.levels), len(m.powers_dbm)):
            if p.name not in POLICIES:
                continue
            out = run(p, m, tr, tab, beat_preds=bp, fallback=0, record=True)
            t = out["trace"]
            r, s, ok = t["r"].astype(int), t["s"].astype(int), t["ok"].astype(bool)
            prev = np.concatenate([[m.floor[int(tr["theta"][0])]], s[:-1]])
            hs = np.where(s != prev, np.vectorize(t_hs.get)(s), 0.0)
            rng = np.random.default_rng(1000 + seed)
            for ci in CI_S:
                lat = wait + t_enc + t_rec[r] + rng.uniform(0, ci, len(r)) + t_evt + t_air[r] + hs
                ld = lat[ok]
                rows.append({"seed": seed, "policy": p.name, "ci_s": ci, "delivered": float(ok.mean()),
                             "p50_s": float(np.percentile(ld, 50)), "p95_s": float(np.percentile(ld, 95)),
                             "p99_s": float(np.percentile(ld, 99)), "max_s": float(ld.max()),
                             "mean_s": float(ld.mean()), "mean_wait_postrr_s": float(wait[ok].mean()),
                             "handshakes": int((s != prev).sum())})
            print(seed, p.name, [(rw["ci_s"], round(rw["p50_s"], 3), round(rw["p99_s"], 3)) for rw in rows[-3:]],
                  flush=True)
    comp = {"t_enc_s": t_enc, "t_rec_s": t_rec.tolist(), "t_air_s": t_air.tolist(), "t_event_overhead_s": t_evt,
            "t_hs_s": t_hs, "window_tail_s": w.post / 360.0,
            "post_rr_s": {"mean": float(post_rr.mean()), "p50": float(np.median(post_rr)),
                          "p99": float(np.percentile(post_rr, 99))}}
    prov = provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed)
    res = REPO / "results" / "wp3"
    res.mkdir(parents=True, exist_ok=True)
    (res / "latency.json").write_text(json.dumps({"provenance": prov, "components": comp, "rows": rows},
                                                 indent=1, default=float))


if __name__ == "__main__":
    main()
