"""Cryptographic primitives with an operation ledger (L2 PQ-Lite).

Every protocol party holds a ``Ledger``; each primitive call records the operation and the
number of bytes it processed. The cost model (``cost.py``) multiplies these counts by the
pinned per-operation cycle counts, so the on-device cost is computed from the protocol's actual
operations rather than from a hand-written tally.

Implementations: ML-KEM-512 and ML-DSA-44 from liboqs (liboqs-python, liboqs 0.16.0), X25519,
HKDF-SHA-256, HMAC-SHA-256 and AES-GCM-128 from ``cryptography``.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
from collections import Counter
from dataclasses import dataclass, field

import oqs
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand

KEM_ALG = "ML-KEM-512"
SIG_ALG = "ML-DSA-44"


@dataclass
class Ledger:
    ops: Counter = field(default_factory=Counter)        # operation -> count
    bytes_in: Counter = field(default_factory=Counter)   # byte-proportional ops -> bytes

    def add(self, op: str, nbytes: int = 0):
        self.ops[op] += 1
        if nbytes:
            self.bytes_in[op] += nbytes

    def blocks(self, op: str, n: int):
        """Count primitive blocks (SHA-256 compressions, AES blocks, GHASH blocks)."""
        self.ops[op] += n

    def merge(self, other: "Ledger"):
        self.ops.update(other.ops)
        self.bytes_in.update(other.bytes_in)

    def as_dict(self) -> dict:
        return {"ops": dict(self.ops), "bytes": dict(self.bytes_in)}


# ------------------------------------------------------------------ X25519
def x25519_keygen(led: Ledger):
    led.add("x25519_keygen")
    sk = X25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sk, pk


def x25519_shared(led: Ledger, sk: X25519PrivateKey, peer_pk: bytes) -> bytes:
    led.add("x25519_shared")
    return sk.exchange(X25519PublicKey.from_public_bytes(peer_pk))


# ------------------------------------------------------------------ ML-KEM-512
def kem_keygen(led: Ledger):
    led.add("mlkem512_keygen")
    k = oqs.KeyEncapsulation(KEM_ALG)
    pk = k.generate_keypair()
    return k, pk


def kem_encaps(led: Ledger, pk: bytes):
    led.add("mlkem512_encaps")
    with oqs.KeyEncapsulation(KEM_ALG) as k:
        ct, ss = k.encap_secret(pk)
    return ct, ss


def kem_decaps(led: Ledger, k: "oqs.KeyEncapsulation", ct: bytes) -> bytes:
    led.add("mlkem512_decaps")
    return k.decap_secret(ct)


# ------------------------------------------------------------------ ML-DSA-44
def sig_keygen(led: Ledger):
    led.add("mldsa44_keygen")
    s = oqs.Signature(SIG_ALG)
    vk = s.generate_keypair()
    return s, vk


def sig_sign(led: Ledger, signer: "oqs.Signature", msg: bytes) -> bytes:
    led.add("mldsa44_sign")
    return signer.sign(msg)


def sig_verify(led: Ledger, msg: bytes, sig: bytes, vk: bytes) -> bool:
    led.add("mldsa44_verify")
    with oqs.Signature(SIG_ALG) as v:
        return v.verify(msg, sig, vk)


# ------------------------------------------------------------------ hashing / KDF / MAC / AEAD
# Block accounting (FIPS 180-4 padding: 9 bytes minimum; HMAC per RFC 2104 with 64-byte blocks)
def sha256_blocks(n: int) -> int:
    return (n + 9 + 63) // 64


def hmac_blocks(msg_len: int) -> int:
    return sha256_blocks(64 + msg_len) + sha256_blocks(64 + 32)


def sha256(led: Ledger, data: bytes) -> bytes:
    led.add("sha256", len(data))
    led.blocks("sha256_block", sha256_blocks(len(data)))
    return hashlib.sha256(data).digest()


def _expand_blocks(info_len: int, length: int) -> int:
    n = -(-length // 32)
    return sum(hmac_blocks((32 if i else 0) + info_len + 1) for i in range(n))


def hkdf(led: Ledger, ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    led.add("hkdf", len(ikm) + len(salt) + len(info))
    led.blocks("sha256_block", hmac_blocks(len(ikm)) + _expand_blocks(len(info), length))
    return HKDF(hashes.SHA256(), length, salt, info).derive(ikm)


def hkdf_expand(led: Ledger, prk: bytes, info: bytes, length: int) -> bytes:
    led.add("hkdf_expand", len(info))
    led.blocks("sha256_block", _expand_blocks(len(info), length))
    return HKDFExpand(hashes.SHA256(), length, info).derive(prk)


def hmac256(led: Ledger, key: bytes, msg: bytes) -> bytes:
    led.add("hmac_sha256", len(msg))
    led.blocks("sha256_block", hmac_blocks(len(msg)))
    return _hmac.new(key, msg, hashlib.sha256).digest()


def ct_equal(a: bytes, b: bytes) -> bool:
    return _hmac.compare_digest(a, b)


def _gcm_blocks(led: Ledger, pt_len: int, aad_len: int):
    # key schedule + H = E(0) + E(J0) for the tag + CTR blocks; GHASH over AAD, data, lengths
    led.blocks("aes128_keyschedule", 1)
    led.blocks("aes128_block", 2 + -(-pt_len // 16))
    led.blocks("ghash_block", -(-aad_len // 16) + -(-pt_len // 16) + 1)


def aead_encrypt(led: Ledger, key: bytes, nonce: bytes, pt: bytes, aad: bytes) -> bytes:
    led.add("aes128gcm", len(pt) + len(aad))
    _gcm_blocks(led, len(pt), len(aad))
    return AESGCM(key).encrypt(nonce, pt, aad)


def aead_decrypt(led: Ledger, key: bytes, nonce: bytes, ct: bytes, aad: bytes) -> bytes:
    led.add("aes128gcm", len(ct) - 16 + len(aad))
    _gcm_blocks(led, len(ct) - 16, len(aad))
    return AESGCM(key).decrypt(nonce, ct, aad)
