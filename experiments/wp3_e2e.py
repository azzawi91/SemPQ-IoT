"""End-to-end trace (decision 2026-10-12, review 1): every DS2 beat, in recording order, occupies
one slot; the scheduler chooses (r, p, s); a delivered beat is classified at the gateway with the
prediction of the proposed encoder at level r (per-beat DS2 predictions of the same seed); a lost
beat is assigned the training-majority class N. Reported: realised macro-F1 over all DS2 beats,
energy, delay, floor compliance, per policy and seed (seed k: encoder seed k and trace seed k).

    python experiments/wp3_e2e.py [inputs_path=...]
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


@hydra.main(config_path="../configs", config_name="wp3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.channel.ble_phy import BerModel, gfsk_ber_curve
    from src.l3_scheduler import policies as pol
    from src.l3_scheduler.model import SystemModel
    from src.l3_scheduler.sim import run, traces
    from src.utils.provenance import provenance
    from src.utils.seed import pin_threads, set_seed

    sys.path.insert(0, str(REPO / "experiments"))
    from wp1_stats import macro_f1

    set_seed(cfg.seed)
    pin_threads(1)
    grid = np.asarray(cfg.ber.ebn0_grid_db, float)
    ber = BerModel("gfsk_mc", grid, gfsk_ber_curve(grid, np.random.default_rng(cfg.ber.seed), n_bits=cfg.ber.n_bits))
    m = SystemModel.build(cfg, ber)
    with open_dict(cfg):
        cfg.E_bar = float(m.e_enc + cfg.E_bar_extra)
    files = sorted((REPO / "results" / "wp1" / "semantic_mitbih").glob("seed*_pred.npz"))
    if not files:
        raise SystemExit("no per-beat predictions (results/wp1/semantic_mitbih/seed*_pred.npz)")
    rows = []
    for f in files:
        seed = int(f.stem.split("_")[0][4:])
        d = np.load(f)
        y = d["y"]
        bp = [d[lv["level"]] for lv in m.levels]
        with open_dict(cfg):
            cfg.slots = int(len(y))
        tr = traces(cfg, np.random.default_rng(seed))
        tab = m.tables_all(tr["gain"], tr["outage"])
        for p in pol.make_policies(cfg, len(m.levels), len(m.powers_dbm)):
            r = run(p, m, tr, tab, beat_preds=bp, fallback=0)
            yhat = r.pop("yhat")
            r.update({"seed": seed, "realised_macro_f1": macro_f1(y, yhat),
                      "macro_f1_all_delivered_smallest_level": macro_f1(y, bp[0])})
            rows.append(r)
            print(f"seed {seed} {r['policy']:45s} realised F1 {r['realised_macro_f1']:.3f} "
                  f"utility {r['utility']:.3f} E {r['energy_per_slot_j'] * 1e6:.1f} uJ floor {r['floor_compliance']:.4f}")
    res = REPO / "results" / "wp3"
    res.mkdir(parents=True, exist_ok=True)
    prov = provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed)
    prov["e_enc_j"], prov["E_bar_effective_j"] = float(m.e_enc), float(cfg.E_bar)
    (res / "e2e.json").write_text(json.dumps({"provenance": prov, "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
