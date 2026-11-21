import numpy as np

from src.channel.ble_phy import BerModel
from src.channel.gilbert_elliott import GilbertElliott
from src.l2_pqlite.linksim import LinkParams, Step, simulate

BER = BerModel("noncoherent_fsk")


def test_clean_link_latency_is_packet_count_plus_compute():
    steps = [Step("node", 900, 0.05), Step("gw", 3300, 0.01), Step("node", 40, 0.02)]
    lp = LinkParams(avg_ebn0_db=40.0, k_factor=100.0, block_len=1, outage=None, slot_s=0.01, deadline_s=5)
    r = simulate(steps, lp, BER, np.random.default_rng(0), 50)
    # packets: 900 -> 4, 3300 -> 14, 40 -> 1; compute slots 5 + 1 + 2
    assert r["completion_prob"] == 1.0
    assert abs(r["latency_mean_s"] - (4 + 14 + 1 + 5 + 1 + 2) * 0.01) < 1e-9


def test_outages_reduce_completion_and_raise_latency():
    steps = [Step("node", 900, 0.0), Step("gw", 3300, 0.0), Step("node", 40, 0.0)]
    ge = GilbertElliott.from_outage(mean_outage_slots=200, outage_fraction=0.5)
    good = LinkParams(20.0, 3.0, 1, None, deadline_s=1.0)
    badl = LinkParams(20.0, 3.0, 1, ge, deadline_s=1.0)
    a = simulate(steps, good, BER, np.random.default_rng(1), 200)
    b = simulate(steps, badl, BER, np.random.default_rng(1), 200)
    assert a["completion_prob"] >= b["completion_prob"]
    assert b["completion_prob"] < 1.0
    assert a["latency_mean_s"] < b["latency_mean_s"]


def test_determinism():
    steps = [Step("node", 500, 0.0)]
    lp = LinkParams(12.0, 3.0, 1, GilbertElliott.from_outage(10, 0.1))
    r1 = simulate(steps, lp, BER, np.random.default_rng(3), 30)
    r2 = simulate(steps, lp, BER, np.random.default_rng(3), 30)
    assert r1 == r2
