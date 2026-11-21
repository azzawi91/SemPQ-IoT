import pytest

pytest.importorskip("oqs")

from src.l2_pqlite import protocol as pr  # noqa: E402
from src.l2_pqlite.cost import CostModel, MissingConstant  # noqa: E402
from src.l2_pqlite.primitives import Ledger  # noqa: E402

FAKE = {"sha256_compression": 1000, "ghash_per_block": 200}  # test-only stand-ins, not sourced


def test_constants_file_values_and_missing_guard():
    m = CostModel.from_yaml()
    assert m.f_hz == 64e6 and m.i_run_a == pytest.approx(3.3e-3) and m.v == 3.0
    led = Ledger()
    led.ops["sha256_block"] = 1
    if m.cycles.get("sha256_compression") is None:
        with pytest.raises(MissingConstant):
            m.compute(led)


def test_compute_arithmetic():
    m = CostModel.from_yaml(overrides=FAKE)
    led = Ledger()
    led.ops.update({"mlkem512_keygen": 1, "x25519_shared": 2, "sha256_block": 10})
    c = m.compute(led)
    expect = 392423 + 2 * 907240 + 10 * 1000
    assert c["cycles"] == expect
    assert c["energy_j"] == pytest.approx(expect / 64e6 * 3.3e-3 * 3.0)


def test_radio_airtime():
    m = CostModel.from_yaml(overrides=FAKE)
    r = m.radio([240], [])
    # one packet: 240 data + 4 fragment + 7 L2CAP/ATT = 251 bytes LL payload
    assert r["packets_tx"] == 1
    assert r["airtime_tx_s"] == pytest.approx((8 + 32 + 16 + 8 * 251 + 24) / 1e6)
    assert m.radio([241], [])["packets_tx"] == 2


def test_handshake_cost_ordering():
    m = CostModel.from_yaml(overrides=FAKE)
    ca = pr.CA.create()
    gid = pr.GatewayIdentity.create(ca)
    out = {}
    for tier in (0, 1):
        node = pr.Node(b"n", b"k" * 32, ca.vk)
        node.cache[b"gateway-1"] = gid.vk
        gw = pr.Gateway(gid, {b"n": b"k" * 32})
        _, _, sz = pr.handshake(node, gw, tier, 0)
        out[tier] = m.total(node.led, [sz["M1"], sz["M3"]], [sz["M2"]])
    assert out[1]["cycles"] > out[0]["cycles"] and out[1]["energy_total_j"] > out[0]["energy_total_j"]
