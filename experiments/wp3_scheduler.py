"""WP3: threat-aware Lyapunov scheduler vs baselines and ablations; V-sweep; link-budget sweep.

    python experiments/wp3_scheduler.py

Outputs: results/wp3/{main,vsweep,gamma}.json, tables/wp3_main.tex, figures/wp3_vsweep.*,
figures/wp3_gamma.*. Every policy sees the same exogenous traces per seed (paired comparison).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(config_path="../configs", config_name="wp3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from scipy import stats

    from src.channel.ble_phy import BerModel, gfsk_ber_curve
    from src.l3_scheduler import policies as pol
    from src.l3_scheduler.model import SystemModel
    from src.l3_scheduler.sim import run, traces
    from src.utils import plotting
    from src.utils.provenance import provenance
    from src.utils.seed import pin_threads, set_seed

    set_seed(cfg.seed)
    pin_threads(1)
    grid = np.asarray(cfg.ber.ebn0_grid_db, float)
    ber = BerModel("gfsk_mc", grid, gfsk_ber_curve(grid, np.random.default_rng(cfg.ber.seed), n_bits=cfg.ber.n_bits))

    def model_for(gamma0):
        c = OmegaConf.merge(cfg, {"gamma0_db": gamma0})
        return SystemModel.build(c, ber)

    m = model_for(cfg.gamma0_db)
    # decision 2026-10-12: the energy budget covers the encoder inference too: E_bar = E_enc + E_bar_extra
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.E_bar = float(m.e_enc + cfg.E_bar_extra)
    print(f"E_enc = {m.e_enc * 1e6:.1f} uJ/beat, E_bar = {cfg.E_bar * 1e6:.1f} uJ/slot")
    main_rows, vs_rows, g_rows = [], [], []
    for seed in cfg.seeds:
        tr = traces(cfg, np.random.default_rng(seed))
        tab = m.tables_all(tr["gain"], tr["outage"])
        for p in pol.make_policies(cfg, len(m.levels), len(m.powers_dbm)):
            main_rows.append({"seed": seed, **run(p, m, tr, tab)})
        for V in cfg.v_sweep:
            vs_rows.append({"seed": seed, "V": V, **run(pol.DPP(V, cfg.E_bar, cfg.D_bar), m, tr, tab)})
        for g0 in cfg.gamma0_sweep:
            mg = model_for(g0)
            tg = mg.tables_all(tr["gain"], tr["outage"])
            ps = pol.make_policies(cfg, len(mg.levels), len(mg.powers_dbm))
            for p in ps[:5] + ps[9:]:  # baselines + combined baselines (decision 2026-10-13)
                g_rows.append({"seed": seed, "gamma0_db": g0, **run(p, mg, tr, tg)})

    res = REPO / "results" / "wp3"
    res.mkdir(parents=True, exist_ok=True)
    prov = provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed)
    prov["e_enc_j"] = float(m.e_enc)
    prov["E_bar_effective_j"] = float(cfg.E_bar)
    df, vs, gd = pd.DataFrame(main_rows), pd.DataFrame(vs_rows), pd.DataFrame(g_rows)

    # paired tests: proposed vs each other policy on utility (seeds paired), Holm-corrected
    prop = df[df.policy == "DPP (proposed)"].set_index("seed").utility
    tests, pv = {}, {}
    for name, g in df[df.policy != "DPP (proposed)"].groupby("policy"):
        y = g.set_index("seed").utility.loc[prop.index]
        diff = prop.to_numpy() - y.to_numpy()
        if np.allclose(diff, 0):
            tests[name] = {"mean_diff": 0.0, "note": "identical utility on every seed; not tested"}
            continue
        t = stats.ttest_rel(prop.to_numpy(), y.to_numpy())
        pv[name] = float(t.pvalue)
        tests[name] = {"mean_diff": float(diff.mean()), "t": float(t.statistic), "p": float(t.pvalue)}
    keys = sorted(pv, key=pv.get)
    run_max = 0.0
    for i, k in enumerate(keys):
        run_max = max(run_max, min(1.0, (len(keys) - i) * pv[k]))
        tests[k]["p_holm"] = run_max

    # theoretical trend check: utility gap ~ 1/V, mean queue ~ V
    va = vs.groupby("V").agg(utility=("utility", "mean"), Q=("Q_mean", "mean"), Z=("Z_mean", "mean"),
                             Qf=("Q_final", "mean"), Zf=("Z_final", "mean"),
                             energy=("energy_per_slot_j", "mean"), delay=("delay_per_slot_s", "mean")).reset_index()
    big = va[va.V >= 100]
    # Theory (docs/theory/wp3_dpp.md): U* - U_avg <= B/V and mean backlog <= (B + V (U_max - U_min)) / eps;
    # time-average constraint excess <= Q(T)/T. Checks: utility gap linear in 1/V; backlog growth at most
    # linear in V (log-log slope in (0, 1]); measured constraint excess within Q(T)/T.
    lin_u = stats.linregress(1.0 / big.V, big.utility)
    lin_q = stats.linregress(np.log(big.V), np.log(big.Q))
    excess_e = np.maximum(va.energy / cfg.E_bar - 1.0, 0.0)
    excess_d = np.maximum(va.delay / cfg.D_bar - 1.0, 0.0)
    T = cfg.slots
    trend = {"utility_vs_invV_Vge100": {"slope": lin_u.slope, "intercept_U_inf": lin_u.intercept, "r2": lin_u.rvalue ** 2},
             "logQ_vs_logV_Vge100": {"slope": lin_q.slope, "r2": lin_q.rvalue ** 2},
             "utility_nondecreasing_in_V": bool(np.all(np.diff(va.utility) >= -1e-3)),
             "Q_growth_at_most_linear": bool(0.0 < lin_q.slope <= 1.0),
             "energy_excess_within_QT_over_T": bool(np.all(excess_e <= va.Qf / T + 1e-9)),
             "delay_excess_within_ZT_over_T": bool(np.all(excess_d <= va.Zf / T + 1e-9))}
    for name, obj in (("main", {"rows": main_rows, "paired_tests_utility": tests}),
                      ("vsweep", {"rows": vs_rows, "aggregate": va.to_dict("list"), "trend": trend}),
                      ("gamma", {"rows": g_rows})):
        (res / f"{name}.json").write_text(json.dumps({"provenance": prov, **obj}, indent=1, default=float))

    agg = df.groupby("policy", sort=False).agg(
        u=("utility", "mean"), u_sd=("utility", "std"), e=("energy_per_slot_j", "mean"), e_sd=("energy_per_slot_j", "std"),
        d=("delay_per_slot_s", "mean"), dr=("delivery_ratio", "mean"), fl=("floor_compliance", "mean"),
        fl_min=("floor_compliance", "min")).reset_index()
    tab = pd.DataFrame({
        "Policy": agg.policy,
        "Utility": [f"{a:.4f} $\\pm$ {b:.4f}" for a, b in zip(agg.u, agg.u_sd)],
        "Energy/slot ($\\mu$J)": [f"{a * 1e6:.2f} $\\pm$ {b * 1e6:.2f}" for a, b in zip(agg.e, agg.e_sd)],
        "Delay/slot (ms)": (agg.d * 1e3).map("{:.1f}".format),
        "Delivery": agg.dr.map("{:.3f}".format),
        "Floor met": agg.fl.map("{:.4f}".format)})
    tdir = REPO / "tables"
    tdir.mkdir(exist_ok=True)
    (tdir / "wp3_main.tex").write_text(tab.to_latex(index=False, escape=False))
    pd.set_option("display.width", 220)
    print(agg.to_string(index=False))
    print(json.dumps(tests, indent=1))
    print(va.to_string(index=False))
    print(json.dumps(trend, indent=1, default=float))

    plotting.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(plotting.IEEE_PAGE_IN * 0.75, 2.3))
    fig.subplots_adjust(wspace=0.45)
    sv = vs.groupby("V").agg(u=("utility", "mean"), us=("utility", "std"), q=("Q_mean", "mean"), qs=("Q_mean", "std"))
    ax[0].errorbar(sv.index, sv.u, yerr=sv.us, marker="o", capsize=2, color=plotting.PALETTE[0])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("V")
    ax[0].set_ylabel("Time-average utility")
    ax[1].errorbar(sv.index, sv.q, yerr=sv.qs, marker="s", capsize=2, color=plotting.PALETTE[1])
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("V")
    ax[1].set_ylabel("Mean energy backlog Q (budget units)")
    plotting.save(fig, REPO / "figures" / "wp3_vsweep")

    fig, ax = plt.subplots(figsize=(plotting.IEEE_COL_IN, 2.5))
    for i, (name, g) in enumerate(gd.groupby("policy", sort=False)):
        a = g.groupby("gamma0_db").utility.agg(["mean", "std"]).reset_index()
        ax.errorbar(a.gamma0_db, a["mean"], yerr=a["std"], marker=plotting.MARKERS[i], capsize=2,
                    color=plotting.PALETTE[i], label=name)
    ax.set_xlabel(r"Average $E_b/N_0$ at 0 dBm (dB)")
    ax.set_ylabel("Time-average utility")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    plotting.save(fig, REPO / "figures" / "wp3_gamma")


if __name__ == "__main__":
    main()
