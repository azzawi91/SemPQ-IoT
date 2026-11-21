"""PQ-Lite: hybrid post-quantum handshake, symmetric ratchet and record layer (brief WP2).

Parties: wearable node N and gateway G.
Provisioning (once): N holds the CA verification key and a pairing secret PSK shared with G.
G holds an ML-DSA-44 key pair and a certificate cert_G = (G_id, vk_G, Sig_CA(G_id || vk_G)).
N verifies cert_G the first time it meets G and caches vk_G; later handshakes skip it.

Security tiers s: 0 = classical (X25519 + PSK), 1 = hybrid-PQ (X25519 + ML-KEM-512 + PSK,
gateway signs the transcript with ML-DSA-44), 2 = hybrid-PQ with immediate re-key (tier 1
handshake, then a KEM re-key at once and after every ``rekey_interval`` records).
Threat level theta in {0, 1, 2}; the floor s >= s_min(theta) is enforced by both parties.

Handshake (1.5 round trips):
  M1  N -> G : N_id, nonce_N, tier, theta_N, epk_N [, ek_N (tier >= 1)], cached-cert flag
  M2  G -> N : nonce_G, tier, theta_G, epk_G [, ct_G], [cert_G if not cached],
               [Sig_G(transcript) (tier >= 1)], MAC_G
  M3  N -> G : MAC_N
Key schedule: IKM = ss_dh || ss_kem || PSK, salt = H(M1 || M2-without-auth), HKDF-SHA-256.
Tier and theta are part of the transcript (signed and MACed by both sides) and of the AEAD
associated data of every record, so a modified tier is detected (downgrade resistance).

Records: per-message symmetric ratchet (HKDF chain), AES-GCM-128, AAD = sid || epoch || seq ||
tier || theta || direction. Message keys are deleted after use (forward secrecy within a session);
a KEM re-key mixes a fresh ML-KEM shared secret into the root key (post-compromise recovery).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

from . import primitives as P

TIERS = (0, 1, 2)
DEFAULT_FLOOR = {0: 0, 1: 1, 2: 2}  # s_min(theta); configurable


class ProtocolError(Exception):
    pass


class DowngradeError(ProtocolError):
    pass


# ------------------------------------------------------------------ encoding
def enc(msg_type: int, *fields: bytes) -> bytes:
    out = bytes([msg_type])
    for f in fields:
        out += struct.pack(">H", len(f)) + f
    return out


def dec(data: bytes) -> tuple[int, list[bytes]]:
    t, i, fields = data[0], 1, []
    while i < len(data):
        (n,) = struct.unpack(">H", data[i:i + 2])
        fields.append(data[i + 2:i + 2 + n])
        i += 2 + n
    if i != len(data):
        raise ProtocolError("malformed message")
    return t, fields


M1, M2, M3, REC = 1, 2, 3, 4
REC_DATA, REC_REKEY_REQ, REC_REKEY_RESP = 0, 1, 2


# ------------------------------------------------------------------ provisioning
@dataclass
class CA:
    signer: object
    vk: bytes

    @classmethod
    def create(cls):
        led = P.Ledger()
        s, vk = P.sig_keygen(led)
        return cls(s, vk)

    def certify(self, g_id: bytes, vk_g: bytes) -> bytes:
        sig = P.sig_sign(P.Ledger(), self.signer, g_id + vk_g)
        return enc(9, g_id, vk_g, sig)


@dataclass
class GatewayIdentity:
    g_id: bytes
    signer: object
    vk: bytes
    cert: bytes

    @classmethod
    def create(cls, ca: CA, g_id: bytes = b"gateway-1"):
        s, vk = P.sig_keygen(P.Ledger())
        return cls(g_id, s, vk, ca.certify(g_id, vk))


# ------------------------------------------------------------------ session state
@dataclass
class Session:
    sid: bytes
    tier: int
    theta: int
    root: bytes
    ck_send: bytes
    ck_recv: bytes
    is_node: bool
    epoch: int = 0
    seq_send: int = 0
    seq_recv: int = -1
    sent_since_rekey: int = 0
    pending_kem: object = None

    def _dir(self, sending: bool) -> int:
        return (1 if self.is_node else 2) if sending else (2 if self.is_node else 1)


def _derive(led, ikm: bytes, salt: bytes, info: bytes) -> dict:
    okm = P.hkdf(led, ikm, salt, b"pqlite v1 " + info, 32 * 5 + 8)
    k = [okm[i * 32:(i + 1) * 32] for i in range(5)]
    return {"conf_n": k[0], "conf_g": k[1], "ck_up": k[2], "ck_down": k[3], "root": k[4],
            "sid": okm[160:168]}


def _ratchet(led, ck: bytes, seq: int) -> tuple[bytes, bytes]:
    out = P.hkdf_expand(led, ck, b"msg" + struct.pack(">Q", seq), 48)
    return out[:16], out[16:]  # message key (AES-128), next chain key


def _aad(sess: Session, seq: int, direction: int, rtype: int) -> bytes:
    return sess.sid + struct.pack(">HQBBBB", sess.epoch, seq, sess.tier, sess.theta, direction, rtype)


# ------------------------------------------------------------------ node
@dataclass
class Node:
    n_id: bytes
    psk: bytes
    ca_vk: bytes
    floor: dict = field(default_factory=lambda: dict(DEFAULT_FLOOR))
    cache: dict = field(default_factory=dict)  # g_id -> vk_G (verified certificate)
    led: P.Ledger = field(default_factory=P.Ledger)
    g_id: bytes = b"gateway-1"

    def start(self, tier: int, theta: int) -> bytes:
        if tier < self.floor[theta]:
            raise DowngradeError(f"node refuses tier {tier} at threat {theta}")
        self._tier, self._theta = tier, theta
        self._nonce = os.urandom(16)
        self._x_sk, epk = P.x25519_keygen(self.led)
        ek = b""
        if tier >= 1:
            self._kem, ek = P.kem_keygen(self.led)
        cached = b"\x01" if self.g_id in self.cache else b"\x00"
        self._m1 = enc(M1, self.n_id, self._nonce, bytes([tier, theta]), epk, ek, cached)
        return self._m1

    def finish(self, m2: bytes) -> tuple[bytes, Session]:
        t, f = dec(m2)
        if t != M2:
            raise ProtocolError("expected M2")
        nonce_g, tt, epk_g, ct, cert, sig, mac_g = f
        tier, theta_g = tt[0], tt[1]
        if tier != self._tier:
            raise DowngradeError("gateway changed the tier")
        theta = max(self._theta, theta_g)
        if tier < self.floor[theta]:
            raise DowngradeError(f"tier {tier} below floor at threat {theta}")
        body = enc(M2, nonce_g, tt, epk_g, ct, cert)
        if tier >= 1:
            if self.g_id not in self.cache:
                if not cert:
                    raise ProtocolError("certificate required")
                _, (g_id, vk_g, ca_sig) = dec(cert)
                if not P.sig_verify(self.led, g_id + vk_g, ca_sig, self.ca_vk):
                    raise ProtocolError("bad certificate")
                self.cache[g_id] = vk_g
            vk_g = self.cache[self.g_id]
            if not P.sig_verify(self.led, P.sha256(self.led, self._m1 + body), sig, vk_g):
                raise ProtocolError("bad gateway signature")
        ss_dh = P.x25519_shared(self.led, self._x_sk, epk_g)
        ss_kem = P.kem_decaps(self.led, self._kem, ct) if tier >= 1 else b""
        th = P.sha256(self.led, self._m1 + body + sig)
        k = _derive(self.led, ss_dh + ss_kem + self.psk, th, bytes([tier, theta]))
        if not P.ct_equal(P.hmac256(self.led, k["conf_g"], th), mac_g):
            raise ProtocolError("gateway key confirmation failed")
        mac_n = P.hmac256(self.led, k["conf_n"], th + mac_g)
        sess = Session(k["sid"], tier, theta, k["root"], k["ck_up"], k["ck_down"], is_node=True)
        return enc(M3, mac_n), sess


# ------------------------------------------------------------------ gateway
@dataclass
class Gateway:
    ident: GatewayIdentity
    psks: dict  # n_id -> PSK
    theta: int = 0
    floor: dict = field(default_factory=lambda: dict(DEFAULT_FLOOR))
    led: P.Ledger = field(default_factory=P.Ledger)

    def respond(self, m1: bytes) -> bytes:
        t, f = dec(m1)
        if t != M1:
            raise ProtocolError("expected M1")
        n_id, nonce_n, tt, epk_n, ek, cached = f
        tier, theta_n = tt[0], tt[1]
        theta = max(self.theta, theta_n)
        if tier < self.floor[theta]:
            raise DowngradeError(f"gateway refuses tier {tier} at threat {theta}")
        if n_id not in self.psks:
            raise ProtocolError("unknown node")
        self._n_id, self._tier, self._theta, self._m1 = n_id, tier, theta, m1
        x_sk, epk_g = P.x25519_keygen(self.led)
        ss_dh = P.x25519_shared(self.led, x_sk, epk_n)
        ct, ss_kem = (P.kem_encaps(self.led, ek) if tier >= 1 else (b"", b""))
        cert = self.ident.cert if (tier >= 1 and cached == b"\x00") else b""
        tt_g = bytes([tier, self.theta])
        body = enc(M2, os.urandom(16), tt_g, epk_g, ct, cert)
        sig = P.sig_sign(self.led, self.ident.signer, P.sha256(self.led, m1 + body)) if tier >= 1 else b""
        th = P.sha256(self.led, m1 + body + sig)
        k = _derive(self.led, ss_dh + ss_kem + self.psks[n_id], th, bytes([tier, theta]))
        mac_g = P.hmac256(self.led, k["conf_g"], th)
        self._th, self._k, self._mac_g = th, k, mac_g
        _, fb = dec(body)
        return enc(M2, *fb, sig, mac_g)

    def complete(self, m3: bytes) -> Session:
        t, (mac_n,) = dec(m3)
        if t != M3:
            raise ProtocolError("expected M3")
        if not P.ct_equal(P.hmac256(self.led, self._k["conf_n"], self._th + self._mac_g), mac_n):
            raise ProtocolError("node key confirmation failed")
        k = self._k
        return Session(k["sid"], self._tier, self._theta, k["root"], k["ck_down"], k["ck_up"], is_node=False)


# ------------------------------------------------------------------ record layer
def seal(led: P.Ledger, sess: Session, payload: bytes, rtype: int = REC_DATA) -> bytes:
    seq = sess.seq_send
    mk, sess.ck_send = _ratchet(led, sess.ck_send, seq)
    d = sess._dir(True)
    nonce = struct.pack(">IQ", (d << 16) | sess.epoch, seq)
    ct = P.aead_encrypt(led, mk, nonce, payload, _aad(sess, seq, d, rtype))
    sess.seq_send += 1
    sess.sent_since_rekey += 1
    header = struct.pack(">BBHI", REC, (sess.tier << 4) | (sess.theta << 2) | rtype, sess.epoch, seq)
    return header + ct


RECORD_HEADER = 8  # bytes: type, tier/theta/rtype, epoch, 32-bit sequence number
AEAD_TAG = 16


def open_record(led: P.Ledger, sess: Session, rec: bytes, floor: dict = DEFAULT_FLOOR) -> tuple[int, bytes]:
    t, flags, epoch, seq = struct.unpack(">BBHI", rec[:RECORD_HEADER])
    if t != REC:
        raise ProtocolError("not a record")
    tier, theta, rtype = flags >> 4, (flags >> 2) & 3, flags & 3
    if tier < floor[max(theta, sess.theta)]:
        raise DowngradeError("record below the security floor")
    if epoch != sess.epoch or seq <= sess.seq_recv:
        raise ProtocolError("stale or replayed record")
    # chain forward over skipped sequence numbers (lost records); skipped keys are discarded
    ck = sess.ck_recv
    for s in range(sess.seq_recv + 1, seq):
        _, ck = _ratchet(led, ck, s)
    mk, ck = _ratchet(led, ck, seq)
    d = sess._dir(False)
    nonce = struct.pack(">IQ", (d << 16) | epoch, seq)
    probe = Session(sess.sid, tier, theta, sess.root, b"", b"", sess.is_node, epoch=epoch)
    pt = P.aead_decrypt(led, mk, nonce, rec[RECORD_HEADER:], _aad(probe, seq, d, rtype))  # raises on tamper
    if tier != sess.tier or theta != sess.theta:
        raise DowngradeError("record tier/threat differ from the session")
    sess.ck_recv, sess.seq_recv = ck, seq
    return rtype, pt


# ------------------------------------------------------------------ KEM re-key (post-compromise recovery)
def rekey_request(led: P.Ledger, sess: Session) -> bytes:
    kem, ek = P.kem_keygen(led)
    sess.pending_kem = kem
    return seal(led, sess, ek, REC_REKEY_REQ)


def rekey_respond(led: P.Ledger, sess: Session, rec: bytes, floor=DEFAULT_FLOOR) -> bytes:
    rtype, ek = open_record(led, sess, rec, floor)
    if rtype != REC_REKEY_REQ:
        raise ProtocolError("expected re-key request")
    ct, ss = P.kem_encaps(led, ek)
    out = seal(led, sess, ct, REC_REKEY_RESP)
    _advance(led, sess, ss)
    return out


def rekey_finish(led: P.Ledger, sess: Session, rec: bytes, floor=DEFAULT_FLOOR):
    rtype, ct = open_record(led, sess, rec, floor)
    if rtype != REC_REKEY_RESP:
        raise ProtocolError("expected re-key response")
    ss = P.kem_decaps(led, sess.pending_kem, ct)
    sess.pending_kem = None
    _advance(led, sess, ss)


def _advance(led, sess: Session, ss: bytes):
    k = _derive(led, ss, sess.root, b"rekey" + struct.pack(">H", sess.epoch + 1))
    sess.root = k["root"]
    sess.ck_send, sess.ck_recv = (k["ck_up"], k["ck_down"]) if sess.is_node else (k["ck_down"], k["ck_up"])
    sess.epoch += 1
    sess.seq_send, sess.seq_recv, sess.sent_since_rekey = 0, -1, 0


def handshake(node: Node, gw: Gateway, tier: int, theta: int):
    """Run M1-M3 in memory; returns (node_session, gateway_session, message sizes)."""
    m1 = node.start(tier, theta)
    m2 = gw.respond(m1)
    m3, sn = node.finish(m2)
    sg = gw.complete(m3)
    return sn, sg, {"M1": len(m1), "M2": len(m2), "M3": len(m3)}
