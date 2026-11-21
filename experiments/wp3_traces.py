"""Per-slot traces of the L3 policies (decision 2026-10-13: presentation figures). Same model, budgets,
seeds and traces as experiments/wp3_scheduler.py; additionally records the per-slot decisions and queues.

Outputs: results/wp3/traces.json (running averages at log-spaced slots, DPP queue paths, DPP decision
counts) and figures/wp3_convergence.*, figures/wp3_policy_heatmap.*. The final running averages equal the
time averages of results/wp3/main.json for the same inputs (checked and recorded as "check_vs_main").

    python experiments/wp3_traces.py inputs_path=results/wp3/inputs.yaml
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

SHOW = {"DPP (proposed)": "DPP", "Myopic greedy": "Greedy", "Fixed rate (DPP over power, tier)": "Fixed rate",
        "Fixed max. security": "Fixed max. security", "Threshold heuristic": "Threshold",
        "Ablation: tier fixed at 2 (threat-unaware)": "Tier fixed at 2"}


@hydra.main(config_path="../configs", config_name="wp3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt

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
    m = SystemModel.build(cfg, ber)
    with open_dict(cfg):
        cfg.E_bar = float(m.e_enc + cfg.E_bar_extra)
    n = int(cfg.slots)
    pts = np.unique(np.round(np.logspace(1, np.log10(n), 60)).astype(int))
    run_avg = {}            # policy -> metric -> list over seeds of running averages at pts
    finals = {}
    qpath, zpath, qlin, zlin = [], [], [], []
    lin_step = 20
    n_lv, n_pw = len(m.levels), len(m.powers_dbm)
    rs_counts = np.zeros((3, n_lv, 3))          # threat level x rate level x tier (DPP)
    gain_edges = np.array([-np.inf, -10, -5, 0, 3, np.inf])
    pg_counts = np.zeros((len(gain_edges) - 1, n_pw))   # channel-gain bin x power (DPP)
    for seed in cfg.seeds:
        tr = traces(cfg, np.random.default_rng(seed))
        tab = m.tables_all(tr["gain"], tr["outage"])
        for p in pol.make_policies(cfg, n_lv, n_pw):
            if p.name not in SHOW:
                continue
            r = run(p, m, tr, tab, record=True)
            rec = r["trace"]
            cnt = np.arange(1, n + 1)
            for key, arr in (("utility", rec["u"]), ("energy_uj", rec["e"] * 1e6), ("delay_ms", rec["d"] * 1e3)):
                ra = np.cumsum(arr) / cnt
                run_avg.setdefault(p.name, {}).setdefault(key, []).append(ra[pts - 1].tolist())
            finals.setdefault(p.name, []).append({"utility": r["utility"], "energy_per_slot_j": r["energy_per_slot_j"]})
            if p.name == "DPP (proposed)":
                qpath.append(rec["Q"][pts - 1].tolist())
                zpath.append(rec["Z"][pts - 1].tolist())
                qlin.append(rec["Q"][::lin_step].tolist())
                zlin.append(rec["Z"][::lin_step].tolist())
                th, rr, ss = rec["theta"].astype(int), rec["r"].astype(int), rec["s"].astype(int)
                np.add.at(rs_counts, (th, rr, ss), 1)
                gb = np.digitize(rec["gain_db"], gain_edges[1:-1])
                np.add.at(pg_counts, (gb, rec["p"].astype(int)), 1)
    # feasibility certificate (Lemma 3): Phi = max over lam in [0,1] of the trace average of
    # min_x [lam (e~ - 1) + (1 - lam)(d~ - 1)] over floor-admissible x of the relaxed system (no handshake
    # charged, re-key charged when the floor forces tier 2). Phi > 0: no policy meets both budgets on the
    # trace; Phi < 0: -Phi is the empirical Slater slack of Assumption 2.
    lams = np.linspace(0.0, 1.0, 201)
    feas = []
    for g0 in cfg.gamma0_sweep:
        mg = SystemModel.build(OmegaConf.merge(cfg, {"gamma0_db": g0}), ber)
        for seed in cfg.seeds:
            tr = traces(cfg, np.random.default_rng(seed))
            _, _, EA, DA = mg.tables_all(tr["gain"], tr["outage"])
            th = np.asarray(tr["theta"], int)
            rk = np.where(np.array([mg.floor[int(x)] for x in th]) == 2, mg.rekey_energy_j / mg.rekey_every, 0.0)
            et = (EA + rk[:, None, None]) / cfg.E_bar - 1.0          # (T, R, P)
            dt = DA / cfg.D_bar - 1.0
            phi = [float(np.mean(np.min(l * et + (1 - l) * dt, axis=(1, 2)))) for l in lams]
            k = int(np.argmax(phi))
            feas.append({"gamma0_db": float(g0), "seed": int(seed), "phi_max": phi[k], "lam_argmax": float(lams[k]),
                         "min_avg_energy_ratio": float(np.mean(np.min(et, axis=(1, 2))) + 1),
                         "min_avg_delay_ratio": float(np.mean(np.min(dt, axis=(1, 2))) + 1)})
    # mixing term delta_W of Theorem 2 for the Markov part of the state (threat chain x Gilbert-Elliott
    # outage chain; the fading is independent per slot), and the constants Lambda, e_max, d_max
    from src.channel.gilbert_elliott import GilbertElliott
    Pth = np.asarray(cfg.threat.P, float)
    ge = GilbertElliott.from_outage(cfg.outage.mean_slots, cfg.outage.fraction)
    Pge = np.array([[1 - ge.p_gb, ge.p_gb], [ge.p_bg, 1 - ge.p_bg]])
    Px = np.kron(Pth, Pge)
    w, v = np.linalg.eig(Px.T)
    pi = np.real(v[:, np.argmin(abs(w - 1))]); pi = pi / pi.sum()
    W_grid = [1, 10, 100, 1000, 10000, n]
    dist, Pk = [], np.eye(len(Px))
    for k in range(1, max(W_grid) + 1):
        Pk = Pk @ Px
        dist.append(np.abs(Pk - pi[None, :]).sum(axis=1).max())
    cs = np.cumsum(dist)
    delta = {int(W): float(cs[W - 1] / W) for W in W_grid}
    e_tab = np.max(m.tables_all(np.ones(1), np.zeros(1, bool))[2])
    e_max = (e_tab + max(m.tiers[s]["e_hs"] for s in m.tiers) + m.rekey_energy_j / m.rekey_every) / cfg.E_bar
    d_max = (cfg.slot_s + max(m.tiers[s]["t_hs"] for s in m.tiers)) / cfg.D_bar
    Lam = max(1.0, e_max - 1, d_max - 1)
    res = REPO / "results" / "wp3"
    check = None
    if (res / "main.json").exists():
        mj = json.loads((res / "main.json").read_text())
        check = {}
        for name in finals:
            ref = [row for row in mj["rows"] if row["policy"] == name]
            if ref:
                check[name] = max(abs(a["utility"] - b["utility"]) for a, b in zip(finals[name], sorted(ref, key=lambda x: x["seed"])))
    out = {"provenance": provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed), "slots_at": pts.tolist(),
           "E_bar_uj": cfg.E_bar * 1e6, "D_bar_ms": cfg.D_bar * 1e3,
           "running_mean": {k: {mk: np.mean(v, axis=0).tolist() for mk, v in d.items()} for k, d in run_avg.items()},
           "dpp_Q": qpath, "dpp_Z": zpath,
           "dpp_counts_threat_rate_tier": rs_counts.tolist(), "dpp_counts_gainbin_power": pg_counts.tolist(),
           "gain_bin_edges_db": [-10, -5, 0, 3], "levels_bytes": [lv["payload_bytes"] for lv in m.levels],
           "powers_dbm": list(m.powers_dbm), "check_vs_main_max_abs_utility_diff": check, "feasibility": feas,
           "mixing": {"delta_W": delta, "e_max": e_max, "d_max": d_max, "Lambda": Lam},
           "dpp_Q_lin": qlin, "dpp_Z_lin": zlin, "lin_step": lin_step}
    res.mkdir(parents=True, exist_ok=True)
    (res / "traces.json").write_text(json.dumps(out, indent=1, default=float) + "\n")

    plotting.apply_style()
    fig, ax = plt.subplots(1, 4, figsize=(plotting.IEEE_PAGE_IN, 2.0))
    for i, (name, lab) in enumerate(SHOW.items()):
        if name not in out["running_mean"]:
            continue
        rm = out["running_mean"][name]
        ls = "--" if "tier fixed" in name else "-"
        lw = 1.3 if name == "DPP (proposed)" else 0.8
        ax[0].plot(pts, rm["utility"], ls, lw=lw, label=lab, color=plotting.PALETTE[i % 7])
        ax[1].plot(pts, rm["energy_uj"], ls, lw=lw, color=plotting.PALETTE[i % 7])
    ax[1].axhline(out["E_bar_uj"], color="k", lw=0.6, ls=":", label=r"$\bar E$")
    xs = np.arange(0, n, lin_step)
    shades = ["#1b2a49", "#2F4B7C", "#5578a8", "#86a3c9", "#b8c9e0"]
    for j, (qp, zp) in enumerate(zip(qlin, zlin)):
        ax[2].plot(xs, qp, lw=0.6, color=shades[j % 5], label=f"seed {j}")
        ax[3].plot(xs, zp, lw=0.6, color=shades[j % 5])
    for a in ax[:2]:
        a.set_xscale("log")
        a.set_xlabel("Slot $t$")
    for a in ax[2:]:
        a.set_xlabel("Slot $t$ ($\\times 10^3$)")
        a.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5", "10", "15", "20"])
    ax[0].set_ylabel("Running-average utility")
    ax[1].set_ylabel(r"Running-average energy ($\mu$J)")
    ax[2].set_ylabel("Energy queue $Q(t)$")
    ax[3].set_ylabel("Delay queue $Z(t)$")
    ax[1].set_ylim(0, 120)
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, ncol=7, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    plotting.save(fig, REPO / "figures" / "wp3_convergence")

    fig, ax = plt.subplots(1, 2, figsize=(plotting.IEEE_PAGE_IN, 1.9), gridspec_kw={"width_ratios": [1.6, 1]})
    # left: share of slots per (threat level, tier) and rate level -> show rate x tier per threat level
    rs = rs_counts / np.maximum(rs_counts.sum(axis=(1, 2), keepdims=True), 1)
    mat = np.concatenate([rs[t] for t in range(3)], axis=1)   # levels x (3 threat * 3 tiers)
    im = ax[0].imshow(mat, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    for (i_, j_), v_ in np.ndenumerate(mat):
        if v_ >= 0.005:
            ax[0].text(j_, i_, f"{v_:.2f}", ha="center", va="center", fontsize=5, color="white" if v_ > 0.55 else "black")
    ax[0].set_yticks(range(n_lv), [str(b) for b in out["levels_bytes"]])
    ax[0].set_ylabel("Payload (bytes)")
    ax[0].set_xticks(range(9), [f"{s}" for _ in range(3) for s in range(3)])
    ax[0].set_xlabel(r"Tier $s$ within threat level $\Theta=0,1,2$")
    for x in (2.5, 5.5):
        ax[0].axvline(x, color="k", lw=0.6)
    fig.colorbar(im, ax=ax[0], fraction=0.04, pad=0.02, label="Share of slots")
    pg = pg_counts / np.maximum(pg_counts.sum(axis=1, keepdims=True), 1)
    im2 = ax[1].imshow(pg, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    for (i_, j_), v_ in np.ndenumerate(pg):
        if v_ >= 0.005:
            ax[1].text(j_, i_, f"{v_:.2f}", ha="center", va="center", fontsize=5, color="white" if v_ > 0.55 else "black")
    ax[1].set_xticks(range(n_pw), [f"${p:g}$" for p in m.powers_dbm])
    ax[1].set_xlabel("Transmit power (dBm)")
    ax[1].set_yticks(range(len(gain_edges) - 1), [r"$<-10$", r"$[-10,-5)$", r"$[-5,0)$", r"$[0,3)$", r"$\geq 3$"])
    ax[1].set_ylabel("Channel gain (dB)")
    fig.colorbar(im2, ax=ax[1], fraction=0.05, pad=0.02, label="Share within row")
    for a in ax:
        a.grid(False)
    fig.tight_layout()
    plotting.save(fig, REPO / "figures" / "wp3_policy_heatmap")
    print(json.dumps(check, indent=1))
    for f in feas:
        print(f)


if __name__ == "__main__":
    main()
