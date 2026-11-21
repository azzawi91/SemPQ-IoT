"""WP2: PQ-Lite and comparators - message sizes, node cost model, per-record overhead, and
handshake completion over an intermittent BLE link.

    python experiments/wp2_pqlite.py                      # full run
    python experiments/wp2_pqlite.py sim.trials=100       # quicker

Outputs (results/wp2/): handshakes.json, records.json, linksim.json, plus tables/wp2_*.tex and
figures/wp2_*.{pdf,png}. The node cost is a model from configs/constants.yaml, not a measurement.
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


def run_handshakes(cm):
    from src.l2_pqlite import comparators as C
    from src.l2_pqlite import primitives as P
    from src.l2_pqlite import protocol as pr

    ca = pr.CA.create()
    gid = pr.GatewayIdentity.create(ca)
    psk = b"\x11" * 32
    rows, steps = [], {}

    def pq(name, tier, cached, rekey):
        node = pr.Node(b"node-7", psk, ca.vk)
        if cached:
            node.cache[b"gateway-1"] = gid.vk
        gw = pr.Gateway(gid, {b"node-7": psk})
        m1 = node.start(tier, 0)
        c1 = cm.compute(node.led)["time_s"]
        m2 = gw.respond(m1)
        before = cm.compute(node.led)["cycles"]
        m3, sn = node.finish(m2)
        c3 = (cm.compute(node.led)["cycles"] - before) / cm.f_hz
        sg = gw.complete(m3)
        sent, recv = [len(m1), len(m3)], [len(m2)]
        st = [("node", len(m1), c1), ("gw", len(m2), 0.0), ("node", len(m3), c3)]
        if rekey:
            b0 = cm.compute(node.led)["cycles"]
            req = pr.rekey_request(node.led, sn)
            c_req = (cm.compute(node.led)["cycles"] - b0) / cm.f_hz
            resp = pr.rekey_respond(gw.led, sg, req)
            pr.rekey_finish(node.led, sn, resp)
            sent.append(len(req))
            recv.append(len(resp))
            st += [("node", len(req), c_req), ("gw", len(resp), 0.0)]
        tot = cm.total(node.led, sent, recv)
        rows.append({"protocol": name, **{k: v for k, v in tot.items() if k != "per_op_cycles"},
                     "messages": len(sent) + len(recv), "node_ops": dict(node.led.ops)})
        steps[name] = st

    pq("PQ-Lite tier 0 (classical, PSK)", 0, True, False)
    pq("PQ-Lite tier 1 (hybrid, cached cert.)", 1, True, False)
    pq("PQ-Lite tier 1 (hybrid, first contact)", 1, False, False)
    pq("PQ-Lite tier 2 (hybrid + immediate re-key)", 2, True, True)

    ns, nvk = P.sig_keygen(P.Ledger())
    ln, lg, sz, ok = C.tls13_hybrid_mtls(ca.signer, ca.vk, gid.signer, gid.vk, ns, nvk)
    assert ok
    tot = cm.total(ln, sz["node_to_gw"], sz["gw_to_node"])
    rows.append({"protocol": "Hybrid TLS 1.3-style (mTLS)", **{k: v for k, v in tot.items() if k != "per_op_cycles"},
                 "messages": len(sz["node_to_gw"]) + len(sz["gw_to_node"]), "node_ops": dict(ln.ops)})
    t = [cm.compute(m)["time_s"] for m in sz["node_marks"]]  # cumulative node compute at each send
    steps["Hybrid TLS 1.3-style (mTLS)"] = [("node", sz["node_to_gw"][0], t[0]), ("gw", sz["gw_to_node"][0], 0.0),
                                            ("node", sz["node_to_gw"][1], t[1] - t[0])]
    ln, lg, sz, ok = C.kemtls_mutual(ca.signer, ca.vk)
    assert ok
    tot = cm.total(ln, sz["node_to_gw"], sz["gw_to_node"])
    rows.append({"protocol": "KEMTLS-style (mutual, signature-free)", **{k: v for k, v in tot.items() if k != "per_op_cycles"},
                 "messages": len(sz["node_to_gw"]) + len(sz["gw_to_node"]), "node_ops": dict(ln.ops)})
    t = [cm.compute(m)["time_s"] for m in sz["node_marks"]]
    steps["KEMTLS-style (mutual, signature-free)"] = [
        ("node", sz["node_to_gw"][0], t[0]), ("gw", sz["gw_to_node"][0], 0.0),
        ("node", sz["node_to_gw"][1], t[1] - t[0]), ("gw", sz["gw_to_node"][1], 0.0),
        ("node", sz["node_to_gw"][2], t[2] - t[1])]
    for name, fn in (("Hybrid TLS 1.3 PSK-(EC)DHE (external or resumption PSK)", lambda: C.tls13_hybrid_psk_dhe()),
                     ("KEMTLS-PDK (mutual, pre-distributed keys)", lambda: C.kemtls_pdk(ca.signer)),
                     # equivalent trust to PQ-Lite tier 1 (PSK + gateway certificate; decision 2026-10-13)
                     ("Hybrid TLS 1.3 cert. + external PSK (RFC 8773, cached cert.)",
                      lambda: C.tls13_cert_extern_psk(ca.signer, ca.vk, gid.signer, gid.vk, True)),
                     ("Hybrid TLS 1.3 cert. + external PSK (RFC 8773, first contact)",
                      lambda: C.tls13_cert_extern_psk(ca.signer, ca.vk, gid.signer, gid.vk, False))):
        ln, lg, sz, ok = fn()
        assert ok
        tot = cm.total(ln, sz["node_to_gw"], sz["gw_to_node"])
        rows.append({"protocol": name, **{k: v for k, v in tot.items() if k != "per_op_cycles"},
                     "messages": len(sz["node_to_gw"]) + len(sz["gw_to_node"]), "node_ops": dict(ln.ops)})
        t = [cm.compute(m)["time_s"] for m in sz["node_marks"]]
        steps[name] = [("node", sz["node_to_gw"][0], t[0]), ("gw", sz["gw_to_node"][0], 0.0),
                       ("node", sz["node_to_gw"][1], t[1] - t[0])]
    return rows, steps


def run_records(cm, payloads):
    from src.l2_pqlite import primitives as P
    from src.l2_pqlite import protocol as pr

    ca = pr.CA.create()
    gid = pr.GatewayIdentity.create(ca)
    node = pr.Node(b"node-7", b"\x11" * 32, ca.vk)
    node.cache[b"gateway-1"] = gid.vk
    gw = pr.Gateway(gid, {b"node-7": b"\x11" * 32})
    sn, sg, _ = pr.handshake(node, gw, 1, 0)
    out = []
    for p in payloads:
        led = P.Ledger()
        rec = pr.seal(led, sn, b"\x00" * p)
        pr.open_record(gw.led, sg, rec)
        tot = cm.total(led, [len(rec)], [])
        out.append({"payload_bytes": p, "record_bytes": len(rec), "overhead_bytes": len(rec) - p,
                    **{k: v for k, v in tot.items() if k != "per_op_cycles"}})
    return out


@hydra.main(config_path="../configs", config_name="wp2", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.channel.ble_phy import BerModel, gfsk_ber_curve
    from src.channel.gilbert_elliott import GilbertElliott
    from src.l2_pqlite.cost import CostModel
    from src.l2_pqlite.linksim import LinkParams, Step, simulate
    from src.utils import plotting
    from src.utils.provenance import provenance
    from src.utils.seed import pin_threads, set_seed

    set_seed(cfg.seed)
    pin_threads(1)
    cm = CostModel.from_yaml()
    rows, steps = run_handshakes(cm)
    recs = run_records(cm, list(cfg.record_payloads))

    g = cfg.ber
    grid = np.asarray(g.ebn0_grid_db, float)
    ber = BerModel("gfsk_mc", grid, gfsk_ber_curve(grid, np.random.default_rng(g.seed), n_bits=g.n_bits))
    sim = []
    for proto, st in steps.items():
        if proto not in cfg.sim.protocols:
            continue
        ss = [Step(*s) for s in st]
        for ebn0 in cfg.sim.ebn0_db:
            for mo in cfg.sim.mean_outage_slots:
                for frac in cfg.sim.outage_fraction:
                    if frac == 0 and mo != cfg.sim.mean_outage_slots[0]:
                        continue  # no outages: the outage duration is irrelevant
                    ge = GilbertElliott.from_outage(mo, frac) if frac > 0 else None
                    lp = LinkParams(ebn0, cfg.sim.k_factor, cfg.sim.block_len, ge, cfg.sim.slot_s, cfg.sim.deadline_s)
                    for seed in cfg.sim.seeds:
                        r = simulate(ss, lp, ber, np.random.default_rng(seed), cfg.sim.trials)
                        sim.append({"protocol": proto, "ebn0_db": ebn0, "mean_outage_slots": mo,
                                    "outage_fraction": frac, "seed": seed, **r})
    res = REPO / "results" / "wp2"
    res.mkdir(parents=True, exist_ok=True)
    prov = provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed)
    (res / "handshakes.json").write_text(json.dumps({"provenance": prov, "rows": rows, "steps": steps}, indent=1, default=float))
    (res / "records.json").write_text(json.dumps({"provenance": prov, "rows": recs}, indent=1, default=float))
    (res / "linksim.json").write_text(json.dumps({"provenance": prov, "rows": sim}, indent=1, default=float))

    # tables
    df = pd.DataFrame(rows)
    tab = pd.DataFrame({
        "Protocol": df.protocol,
        "Node TX (B)": df.bytes_tx, "Node RX (B)": df.bytes_rx,
        "Packets": df.packets_tx + df.packets_rx,
        "Node compute (Mcycles)": (df.cycles / 1e6).map("{:.2f}".format),
        "Compute time (ms)": (df.time_s * 1e3).map("{:.1f}".format),
        "Compute energy (mJ)": (df.energy_j * 1e3).map("{:.3f}".format),
        "Radio energy (mJ)": ((df.energy_tx_j + df.energy_rx_j) * 1e3).map("{:.3f}".format),
        "Total energy (mJ)": (df.energy_total_j * 1e3).map("{:.3f}".format)})
    tdir = REPO / "tables"
    tdir.mkdir(exist_ok=True)
    (tdir / "wp2_handshakes.tex").write_text(tab.to_latex(index=False, escape=True))
    print(tab.to_string(index=False))
    rt = pd.DataFrame(recs)
    print(rt[["payload_bytes", "record_bytes", "overhead_bytes", "cycles", "energy_total_j"]].to_string(index=False))

    # figure: completion probability vs mean outage duration at the middle Eb/N0 and outage fraction
    sd = pd.DataFrame(sim)
    if not sd.empty:
        plotting.apply_style()
        import matplotlib.pyplot as plt

        e_mid = sorted(cfg.sim.ebn0_db)[len(cfg.sim.ebn0_db) // 2]
        f_sel = cfg.sim.figure_outage_fraction
        sub = sd[(sd.ebn0_db == e_mid) & (sd.outage_fraction == f_sel)]
        fig, ax = plt.subplots(figsize=(plotting.IEEE_COL_IN, 2.5))
        for i, (proto, gp) in enumerate(sub.groupby("protocol", sort=False)):
            a = gp.groupby("mean_outage_slots").completion_prob.agg(["mean", "std"]).reset_index()
            ax.errorbar(a.mean_outage_slots * cfg.sim.slot_s * 1e3, a["mean"], yerr=a["std"], capsize=2,
                        marker=plotting.MARKERS[i % 7], color=plotting.PALETTE[i % 7], label=proto)
        ax.set_xscale("log")
        ax.set_xlabel("Mean outage duration (ms)")
        ax.set_ylabel(f"Completion within {cfg.sim.deadline_s:g} s")
        ax.legend(fontsize=5.5, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.24))
        plotting.save(fig, REPO / "figures" / "wp2_completion_vs_outage")
        fig, ax = plt.subplots(figsize=(plotting.IEEE_COL_IN, 2.5))
        for i, (proto, gp) in enumerate(sub.groupby("protocol", sort=False)):
            a = gp.groupby("mean_outage_slots").latency_mean_s.agg(["mean", "std"]).reset_index()
            ax.errorbar(a.mean_outage_slots * cfg.sim.slot_s * 1e3, a["mean"] * 1e3, yerr=a["std"] * 1e3, capsize=2,
                        marker=plotting.MARKERS[i % 7], color=plotting.PALETTE[i % 7], label=proto)
        ax.set_xscale("log")
        ax.set_xlabel("Mean outage duration (ms)")
        ax.set_ylabel("Mean completed-handshake latency (ms)")
        ax.legend(fontsize=5.5, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.24))
        plotting.save(fig, REPO / "figures" / "wp2_latency_vs_outage")


if __name__ == "__main__":
    main()
