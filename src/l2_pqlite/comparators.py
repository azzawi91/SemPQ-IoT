"""Comparator handshakes (brief WP2), implemented with the same primitives and ledger so that
message sizes and per-party operation counts come from executed code.

Framing is minimal and identical across protocols (length-prefixed fields, one 16-byte AEAD tag
per encrypted flight); the comparators are TLS-like and KEMTLS-like models, not byte-exact
implementations of RFC 8446 or of the KEMTLS papers.

* ``tls13_hybrid_mtls`` - TLS 1.3-style, X25519 + ML-KEM-512 key share, mutual authentication
  with ML-DSA-44 certificates and CertificateVerify signatures on both sides; the node verifies
  the gateway certificate on every handshake (no caching) and signs once.
* ``kemtls_mutual`` - KEMTLS-style signature-free handshake: ephemeral ML-KEM-512 for the
  key exchange, static ML-KEM-512 keys in certificates for implicit mutual authentication;
  the node verifies the gateway certificate's CA signature on every handshake.
* ``tls13_hybrid_psk_dhe`` - TLS 1.3 psk_dhe_ke mode with an X25519 + ML-KEM-512 hybrid key
  share and a PSK (an external pairing PSK or a resumption PSK from an earlier session; the
  handshake is the same): no certificates and no signatures; ClientHello carries a PSK binder,
  the server answers with ServerHello and an encrypted {EncryptedExtensions, Finished} flight,
  and the client sends Finished. Authentication of both sides rests on the PSK only.
* ``kemtls_pdk`` - KEMTLS with pre-distributed keys, mutual: the node holds the gateway's
  static ML-KEM-512 key (cached certificate) and its own static ML-KEM-512 key pair. The node
  sends an ephemeral encapsulation key and a ciphertext to the gateway's static key in the first
  flight (with its certificate encrypted); the gateway answers with the ephemeral ciphertext, a
  ciphertext to the node's static key and Finished; the node sends Finished. Signature-free.
* ``tls13_cert_extern_psk`` - equivalent-trust comparator (external review, decision 2026-10-13):
  TLS 1.3 with certificate-based server authentication combined with an external PSK (RFC 8773,
  tls_cert_with_extern_psk, psk_dhe_ke mode) and an X25519 + ML-KEM-512 hybrid key share. The node
  is authenticated by the PSK, the gateway by an ML-DSA-44 certificate and CertificateVerify, as in
  PQ-Lite tier 1. ``cached=True`` models the RFC 7924 cached-information extension: the node already
  holds the gateway certificate, the server sends its 32-byte hash instead, and the CA signature is not
  re-verified (PQ-Lite tier 1 with a cached certificate); ``cached=False`` sends and verifies it.
* ``ecdhe_psk`` - classical: X25519 ephemeral Diffie-Hellman authenticated by the pairing PSK
  (TLS 1.3 psk_dhe_ke style); identical to PQ-Lite tier 0.
"""
from __future__ import annotations

import os

from . import primitives as P
from .protocol import dec, enc

TAG = 16


def _cert(ca_signer, subject: bytes, key: bytes) -> bytes:
    return enc(9, subject, key, P.sig_sign(P.Ledger(), ca_signer, subject + key))


def tls13_hybrid_mtls(ca_signer, ca_vk, gw_signer, gw_vk, node_signer, node_vk):
    ln, lg = P.Ledger(), P.Ledger()
    cert_g = _cert(ca_signer, b"gateway-1", gw_vk)
    cert_n = _cert(ca_signer, b"node-7", node_vk)
    # ClientHello
    xn, epk_n = P.x25519_keygen(ln)
    kem, ek = P.kem_keygen(ln)
    ch = enc(1, os.urandom(32), epk_n, ek)
    marks = [P.Ledger()]
    marks[0].merge(ln)
    # ServerHello + {EncryptedExtensions, CertificateRequest, Certificate, CertificateVerify, Finished}
    xg, epk_g = P.x25519_keygen(lg)
    ct, ss_g = P.kem_encaps(lg, ek)
    sh = enc(2, os.urandom(32), epk_g, ct)
    ss_dh_g = P.x25519_shared(lg, xg, epk_n)
    th = P.sha256(lg, ch + sh)
    hs_g = P.hkdf(lg, ss_dh_g + ss_g, th, b"tls13 hs", 64)
    cv_g = P.sig_sign(lg, gw_signer, P.sha256(lg, ch + sh + cert_g))
    fin_g = P.hmac256(lg, hs_g[:32], P.sha256(lg, ch + sh + cert_g + cv_g))
    flight_g = enc(3, b"\x00" * 2, b"\x00" * 2, cert_g, cv_g, fin_g)
    enc_g = P.aead_encrypt(lg, hs_g[32:48], b"\x00" * 12, flight_g, b"")
    # client side
    ss_dh_n = P.x25519_shared(ln, xn, epk_g)
    ss_n = P.kem_decaps(ln, kem, ct)
    P.sha256(ln, ch + sh)
    hs_n = P.hkdf(ln, ss_dh_n + ss_n, th, b"tls13 hs", 64)
    flight = P.aead_decrypt(ln, hs_n[32:48], b"\x00" * 12, enc_g, b"")
    assert flight == flight_g
    _, (g_id, vk, ca_sig) = dec(cert_g)
    assert P.sig_verify(ln, g_id + vk, ca_sig, ca_vk)
    assert P.sig_verify(ln, P.sha256(ln, ch + sh + cert_g), cv_g, vk)
    assert P.ct_equal(P.hmac256(ln, hs_n[:32], P.sha256(ln, ch + sh + cert_g + cv_g)), fin_g)
    cv_n = P.sig_sign(ln, node_signer, P.sha256(ln, ch + sh + flight_g + cert_n))
    fin_n = P.hmac256(ln, hs_n[:32], P.sha256(ln, ch + sh + flight_g + cert_n + cv_n))
    enc_n = P.aead_encrypt(ln, hs_n[32:48], b"\x00" * 11 + b"\x01", enc(4, cert_n, cv_n, fin_n), b"")
    marks.append(P.Ledger())
    marks[1].merge(ln)
    # server verifies the client
    P.aead_decrypt(lg, hs_g[32:48], b"\x00" * 11 + b"\x01", enc_n, b"")
    P.sig_verify(lg, b"node-7" + node_vk, cert_n[-2420:], ca_vk)
    P.sig_verify(lg, P.sha256(lg, ch + sh + flight_g + cert_n), cv_n, node_vk)
    sizes = {"node_to_gw": [len(ch), len(enc_n)], "gw_to_node": [len(sh) + len(enc_g)], "node_marks": marks}
    return ln, lg, sizes, hs_n == hs_g


def kemtls_mutual(ca_signer, ca_vk):
    ln, lg = P.Ledger(), P.Ledger()
    # long-term static KEM keys certified by the CA (generation is provisioning, not counted)
    g_static, g_pk = P.kem_keygen(P.Ledger())
    n_static, n_pk = P.kem_keygen(P.Ledger())
    cert_g = _cert(ca_signer, b"gateway-1", g_pk)
    cert_n = _cert(ca_signer, b"node-7", n_pk)
    kem_e, ek = P.kem_keygen(ln)
    ch = enc(1, os.urandom(32), ek)
    marks = [P.Ledger()]
    marks[0].merge(ln)
    ct_e, ss_e_g = P.kem_encaps(lg, ek)
    sh = enc(2, os.urandom(32), ct_e)
    hs_g = P.hkdf(lg, ss_e_g, P.sha256(lg, ch + sh), b"kemtls hs", 48)
    enc_cert_g = P.aead_encrypt(lg, hs_g[32:48], b"\x00" * 12, cert_g, b"")
    # client: decaps ephemeral, verify certificate, encapsulate to the gateway's static key
    ss_e_n = P.kem_decaps(ln, kem_e, ct_e)
    hs_n = P.hkdf(ln, ss_e_n, P.sha256(ln, ch + sh), b"kemtls hs", 48)
    cg = P.aead_decrypt(ln, hs_n[32:48], b"\x00" * 12, enc_cert_g, b"")
    assert P.sig_verify(ln, b"gateway-1" + g_pk, cg[-2420:], ca_vk)
    ct_s, ss_s_n = P.kem_encaps(ln, g_pk)
    ahs_n = P.hkdf(ln, ss_e_n + ss_s_n, hs_n, b"kemtls ahs", 48)
    c_flight = enc(3, ct_s, P.aead_encrypt(ln, ahs_n[32:48], b"\x00" * 12, cert_n, b""))
    marks.append(P.Ledger())
    marks[1].merge(ln)
    # server: decapsulate, verify the client certificate, encapsulate to the client's static key
    ss_s_g = P.kem_decaps(lg, g_static, ct_s)
    ahs_g = P.hkdf(lg, ss_e_g + ss_s_g, hs_g, b"kemtls ahs", 48)
    P.sig_verify(lg, b"node-7" + n_pk, cert_n[-2420:], ca_vk)
    ct_c, ss_c_g = P.kem_encaps(lg, n_pk)
    ms_g = P.hkdf(lg, ss_c_g, ahs_g, b"kemtls ms", 64)
    fin_g = P.hmac256(lg, ms_g[:32], b"server finished")
    s_flight = enc(4, ct_c, fin_g)
    ss_c_n = P.kem_decaps(ln, n_static, ct_c)
    ms_n = P.hkdf(ln, ss_c_n, ahs_n, b"kemtls ms", 64)
    assert P.ct_equal(P.hmac256(ln, ms_n[:32], b"server finished"), fin_g)
    fin_n = P.hmac256(ln, ms_n[:32], b"client finished")
    marks.append(P.Ledger())
    marks[2].merge(ln)
    sizes = {"node_to_gw": [len(ch), len(c_flight), len(fin_n) + 3],
             "gw_to_node": [len(sh) + len(enc_cert_g), len(s_flight)], "node_marks": marks}
    return ln, lg, sizes, ms_n == ms_g


def tls13_hybrid_psk_dhe():
    ln, lg = P.Ledger(), P.Ledger()
    psk = b"\x22" * 32
    # ClientHello: random, hybrid key shares, PSK identity and binder (HMAC over the partial hello)
    xn, epk_n = P.x25519_keygen(ln)
    kem, ek = P.kem_keygen(ln)
    early = P.hkdf(ln, psk, b"", b"tls13 early", 32)
    partial = enc(1, os.urandom(32), epk_n, ek, b"psk-id-0001")
    binder = P.hmac256(ln, early, P.sha256(ln, partial))
    ch = enc(1, partial, binder)
    marks = [P.Ledger()]
    marks[0].merge(ln)
    # server: verify binder, hybrid shares, {EncryptedExtensions, Finished}
    early_g = P.hkdf(lg, psk, b"", b"tls13 early", 32)
    assert P.ct_equal(P.hmac256(lg, early_g, P.sha256(lg, partial)), binder)
    xg, epk_g = P.x25519_keygen(lg)
    ct, ss_g = P.kem_encaps(lg, ek)
    sh = enc(2, os.urandom(32), epk_g, ct)
    ss_dh_g = P.x25519_shared(lg, xg, epk_n)
    th = P.sha256(lg, ch + sh)
    hs_g = P.hkdf(lg, early_g + ss_dh_g + ss_g, th, b"tls13 hs", 64)
    fin_g = P.hmac256(lg, hs_g[:32], th)
    enc_g = P.aead_encrypt(lg, hs_g[32:48], b"\x00" * 12, enc(3, b"\x00" * 2, fin_g), b"")
    # client
    ss_dh_n = P.x25519_shared(ln, xn, epk_g)
    ss_n = P.kem_decaps(ln, kem, ct)
    th_n = P.sha256(ln, ch + sh)
    hs_n = P.hkdf(ln, early + ss_dh_n + ss_n, th_n, b"tls13 hs", 64)
    _, (_, fin) = dec(P.aead_decrypt(ln, hs_n[32:48], b"\x00" * 12, enc_g, b""))
    assert P.ct_equal(P.hmac256(ln, hs_n[:32], th_n), fin)
    fin_n = P.hmac256(ln, hs_n[:32], th_n + fin)
    enc_n = P.aead_encrypt(ln, hs_n[32:48], b"\x00" * 11 + b"\x01", enc(4, fin_n), b"")
    marks.append(P.Ledger())
    marks[1].merge(ln)
    P.aead_decrypt(lg, hs_g[32:48], b"\x00" * 11 + b"\x01", enc_n, b"")
    sizes = {"node_to_gw": [len(ch), len(enc_n)], "gw_to_node": [len(sh) + len(enc_g)], "node_marks": marks}
    return ln, lg, sizes, hs_n == hs_g


def kemtls_pdk(ca_signer):
    ln, lg = P.Ledger(), P.Ledger()
    # provisioning (not counted): static KEM key pairs; the node caches the gateway's static key
    g_static, g_pk = P.kem_keygen(P.Ledger())
    n_static, n_pk = P.kem_keygen(P.Ledger())
    cert_n = _cert(ca_signer, b"node-7", n_pk)
    # node: ephemeral KEM key, encapsulate to the gateway's static key, early-encrypted certificate
    kem_e, ek = P.kem_keygen(ln)
    ct_s, ss_s_n = P.kem_encaps(ln, g_pk)
    ets_n = P.hkdf(ln, ss_s_n, b"", b"kemtls-pdk early", 32)
    hello = enc(1, os.urandom(32), ek, ct_s)
    ch = enc(1, hello, P.aead_encrypt(ln, ets_n[:16], b"\x00" * 12, cert_n, P.sha256(ln, hello)))
    marks = [P.Ledger()]
    marks[0].merge(ln)
    # gateway: decapsulate static, verify the node certificate, ephemeral and client-static ciphertexts
    ss_s_g = P.kem_decaps(lg, g_static, ct_s)
    ets_g = P.hkdf(lg, ss_s_g, b"", b"kemtls-pdk early", 32)
    _, (hello_g, ecert) = dec(ch)
    cn = P.aead_decrypt(lg, ets_g[:16], b"\x00" * 12, ecert, P.sha256(lg, hello_g))
    P.sig_verify(lg, b"node-7" + n_pk, cn[-2420:], P.sig_keygen(P.Ledger())[1])  # cost of the CA check
    ct_e, ss_e_g = P.kem_encaps(lg, ek)
    ct_c, ss_c_g = P.kem_encaps(lg, n_pk)
    th = P.sha256(lg, ch + enc(2, ct_e, ct_c))
    ms_g = P.hkdf(lg, ss_s_g + ss_e_g + ss_c_g, th, b"kemtls-pdk ms", 64)
    fin_g = P.hmac256(lg, ms_g[:32], th)
    sh = enc(2, ct_e, ct_c, fin_g)
    # node
    ss_e_n = P.kem_decaps(ln, kem_e, ct_e)
    ss_c_n = P.kem_decaps(ln, n_static, ct_c)
    th_n = P.sha256(ln, ch + enc(2, ct_e, ct_c))
    ms_n = P.hkdf(ln, ss_s_n + ss_e_n + ss_c_n, th_n, b"kemtls-pdk ms", 64)
    assert P.ct_equal(P.hmac256(ln, ms_n[:32], th_n), fin_g)
    fin_n = P.hmac256(ln, ms_n[:32], th_n + fin_g)
    c_fin = enc(3, fin_n)
    marks.append(P.Ledger())
    marks[1].merge(ln)
    sizes = {"node_to_gw": [len(ch), len(c_fin)], "gw_to_node": [len(sh)], "node_marks": marks}
    return ln, lg, sizes, ms_n == ms_g


def tls13_cert_extern_psk(ca_signer, ca_vk, gw_signer, gw_vk, cached: bool):
    ln, lg = P.Ledger(), P.Ledger()
    psk = b"\x22" * 32
    cert_g = _cert(ca_signer, b"gateway-1", gw_vk)
    # ClientHello: random, hybrid key shares, tls_cert_with_extern_psk flag, PSK identity and binder
    xn, epk_n = P.x25519_keygen(ln)
    kem, ek = P.kem_keygen(ln)
    early = P.hkdf(ln, psk, b"", b"tls13 early", 32)
    partial = enc(1, os.urandom(32), epk_n, ek, b"\x00\x21", b"psk-id-0001",
                  P.sha256(P.Ledger(), cert_g) if cached else b"")
    binder = P.hmac256(ln, early, P.sha256(ln, partial))
    ch = enc(1, partial, binder)
    marks = [P.Ledger()]
    marks[0].merge(ln)
    # server: binder, hybrid shares, {EncryptedExtensions, Certificate, CertificateVerify, Finished}
    early_g = P.hkdf(lg, psk, b"", b"tls13 early", 32)
    assert P.ct_equal(P.hmac256(lg, early_g, P.sha256(lg, partial)), binder)
    xg, epk_g = P.x25519_keygen(lg)
    ct, ss_g = P.kem_encaps(lg, ek)
    sh = enc(2, os.urandom(32), epk_g, ct, b"\x00\x21")
    ss_dh_g = P.x25519_shared(lg, xg, epk_n)
    th = P.sha256(lg, ch + sh)
    hs_g = P.hkdf(lg, early_g + ss_dh_g + ss_g, th, b"tls13 hs", 64)
    cert_field = P.sha256(lg, cert_g) if cached else cert_g
    cv_g = P.sig_sign(lg, gw_signer, P.sha256(lg, ch + sh + cert_field))
    fin_g = P.hmac256(lg, hs_g[:32], P.sha256(lg, ch + sh + cert_field + cv_g))
    flight_g = enc(3, b"\x00" * 2, cert_field, cv_g, fin_g)
    enc_g = P.aead_encrypt(lg, hs_g[32:48], b"\x00" * 12, flight_g, b"")
    # client
    ss_dh_n = P.x25519_shared(ln, xn, epk_g)
    ss_n = P.kem_decaps(ln, kem, ct)
    th_n = P.sha256(ln, ch + sh)
    hs_n = P.hkdf(ln, early + ss_dh_n + ss_n, th_n, b"tls13 hs", 64)
    flight = P.aead_decrypt(ln, hs_n[32:48], b"\x00" * 12, enc_g, b"")
    assert flight == flight_g
    _, (_, cf, cv, fin) = dec(flight)
    vk = gw_vk
    if not cached:
        _, (g_id, vk, ca_sig) = dec(cf)
        assert P.sig_verify(ln, g_id + vk, ca_sig, ca_vk)
    assert P.sig_verify(ln, P.sha256(ln, ch + sh + cf), cv, vk)
    assert P.ct_equal(P.hmac256(ln, hs_n[:32], P.sha256(ln, ch + sh + cf + cv)), fin)
    fin_n = P.hmac256(ln, hs_n[:32], P.sha256(ln, ch + sh + flight))
    enc_n = P.aead_encrypt(ln, hs_n[32:48], b"\x00" * 11 + b"\x01", enc(4, fin_n), b"")
    marks.append(P.Ledger())
    marks[1].merge(ln)
    P.aead_decrypt(lg, hs_g[32:48], b"\x00" * 11 + b"\x01", enc_n, b"")
    sizes = {"node_to_gw": [len(ch), len(enc_n)], "gw_to_node": [len(sh) + len(enc_g)], "node_marks": marks}
    return ln, lg, sizes, hs_n == hs_g
