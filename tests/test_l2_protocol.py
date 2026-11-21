import pytest

pytest.importorskip("oqs")

from src.l2_pqlite import primitives as P  # noqa: E402
from src.l2_pqlite import protocol as pr  # noqa: E402
from src.l2_pqlite.fragment import APP_PAYLOAD, Reassembler, fragment, ll_payload_sizes  # noqa: E402


@pytest.fixture(scope="module")
def world():
    ca = pr.CA.create()
    gid = pr.GatewayIdentity.create(ca)
    return ca, gid


def _pair(world, theta_g=0):
    ca, gid = world
    psk = b"k" * 32
    node = pr.Node(b"node-7", psk, ca.vk)
    gw = pr.Gateway(gid, {b"node-7": psk}, theta=theta_g)
    return node, gw


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_handshake_and_records(world, tier):
    node, gw = _pair(world)
    sn, sg, sizes = pr.handshake(node, gw, tier, theta=0)
    assert sn.sid == sg.sid and sn.ck_send == sg.ck_recv and sn.ck_recv == sg.ck_send
    for i in range(5):
        rt, pt = pr.open_record(gw.led, sg, pr.seal(node.led, sn, b"beat%d" % i))
        assert pt == b"beat%d" % i
    rt, pt = pr.open_record(node.led, sn, pr.seal(gw.led, sg, b"ack"))
    assert pt == b"ack"
    if tier >= 1:
        assert sizes["M1"] > 800 and sizes["M2"] > 2420  # ML-KEM key, ML-DSA signature
    else:
        assert sizes["M1"] < 100


def test_certificate_verified_once_then_cached(world):
    node, gw = _pair(world)
    pr.handshake(node, gw, 1, 0)
    first = node.led.ops["mldsa44_verify"]
    _, _, sizes2 = pr.handshake(node, gw, 1, 0)
    assert first == 2  # certificate (CA signature) + transcript signature
    assert node.led.ops["mldsa44_verify"] == 3  # only the transcript signature the second time
    assert sizes2["M2"] < 2420 + 1312 + 800  # no certificate in the second M2


def test_floor_refuses_low_tier(world):
    node, gw = _pair(world, theta_g=2)
    with pytest.raises(pr.DowngradeError):
        node.start(0, theta=1)  # node's own floor at threat 1
    m1 = node.start(0, theta=0)
    with pytest.raises(pr.DowngradeError):
        gw.respond(m1)  # gateway at threat 2 requires tier 2


def test_tier_tampering_in_m1_detected(world):
    node, gw = _pair(world)
    m1 = node.start(1, 0)
    t, f = pr.dec(m1)
    f[2] = bytes([0, 0])  # attacker rewrites tier 1 -> 0
    m2 = gw.respond(pr.enc(t, *f))
    with pytest.raises(pr.ProtocolError):
        node.finish(m2)


def test_wrong_psk_fails_key_confirmation(world):
    ca, gid = world
    node = pr.Node(b"node-7", b"a" * 32, ca.vk)
    gw = pr.Gateway(gid, {b"node-7": b"b" * 32})
    with pytest.raises(pr.ProtocolError, match="confirmation"):
        pr.handshake(node, gw, 1, 0)


def test_record_tamper_replay_and_header_downgrade(world):
    node, gw = _pair(world)
    sn, sg, _ = pr.handshake(node, gw, 1, 0)
    rec = pr.seal(node.led, sn, b"x" * 20)
    bad = bytearray(rec)
    bad[-1] ^= 1
    with pytest.raises(Exception):
        pr.open_record(gw.led, sg, bytes(bad))
    hdr = bytearray(rec)
    hdr[1] = hdr[1] & 0x0F  # rewrite tier bits 1 -> 0 in the header
    with pytest.raises(Exception):
        pr.open_record(gw.led, sg, bytes(hdr))
    pr.open_record(gw.led, sg, rec)
    with pytest.raises(pr.ProtocolError, match="replayed"):
        pr.open_record(gw.led, sg, rec)


def test_lost_records_are_skipped(world):
    node, gw = _pair(world)
    sn, sg, _ = pr.handshake(node, gw, 1, 0)
    recs = [pr.seal(node.led, sn, b"%d" % i) for i in range(6)]
    assert pr.open_record(gw.led, sg, recs[0])[1] == b"0"
    assert pr.open_record(gw.led, sg, recs[4])[1] == b"4"  # 1-3 lost
    with pytest.raises(pr.ProtocolError):
        pr.open_record(gw.led, sg, recs[2])  # late record after the receiver moved on


def test_rekey_changes_keys_and_continues(world):
    node, gw = _pair(world)
    sn, sg, _ = pr.handshake(node, gw, 2, 0)
    old_root = sn.root
    req = pr.rekey_request(node.led, sn)
    resp = pr.rekey_respond(gw.led, sg, req)
    pr.rekey_finish(node.led, sn, resp)
    assert sn.root == sg.root != old_root and sn.epoch == sg.epoch == 1
    assert pr.open_record(gw.led, sg, pr.seal(node.led, sn, b"after"))[1] == b"after"


def test_ledger_counts_node_operations(world):
    node, gw = _pair(world)
    node.cache[b"gateway-1"] = world[1].vk
    pr.handshake(node, gw, 1, 0)
    ops = node.led.ops
    assert ops["mlkem512_keygen"] == 1 and ops["mlkem512_decaps"] == 1
    assert ops["x25519_keygen"] == 1 and ops["x25519_shared"] == 1
    assert ops["mldsa44_verify"] == 1 and ops.get("mldsa44_sign", 0) == 0
    assert gw.led.ops["mldsa44_sign"] == 1 and gw.led.ops["mlkem512_encaps"] == 1


def test_fragmentation_roundtrip_and_timeout():
    msg = bytes(range(256)) * 13  # 3328 bytes
    frags = fragment(msg, 5)
    assert all(len(f) <= APP_PAYLOAD for f in frags)
    assert sum(ll_payload_sizes(len(msg))) == len(msg) + len(frags) * (4 + 7)
    r = Reassembler(timeout=len(frags) + 1)
    out = None
    for i, f in enumerate(reversed(frags)):
        out = r.push(f, slot=i) or out
    assert out == msg
    r2 = Reassembler(timeout=2)
    assert r2.push(frags[0], 0) is None
    r2.push(frags[1], 5)  # first fragment expired, new partial message started
    assert 5 in r2._buf and len(r2._buf[5][2]) == 1


def test_comparators_agree_on_keys_and_count_ops(world):
    from src.l2_pqlite import comparators as C

    ca, gid = world
    ns, nvk = P.sig_keygen(P.Ledger())
    ln, lg, sizes, ok = C.tls13_hybrid_mtls(ca.signer, ca.vk, gid.signer, gid.vk, ns, nvk)
    assert ok and ln.ops["mldsa44_verify"] == 2 and ln.ops["mldsa44_sign"] == 1
    assert sum(sizes["gw_to_node"]) > 2 * 2420 + 1312
    ln, lg, sizes, ok = C.kemtls_mutual(ca.signer, ca.vk)
    assert ok and ln.ops.get("mldsa44_sign", 0) == 0 and ln.ops["mldsa44_verify"] == 1
    assert ln.ops["mlkem512_decaps"] == 2 and ln.ops["mlkem512_encaps"] == 1
