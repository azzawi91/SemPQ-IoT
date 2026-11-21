

def test_new_comparators_agree_on_keys():
    from src.l2_pqlite import comparators as C
    from src.l2_pqlite import protocol as pr
    ca = pr.CA.create()
    for fn in (C.tls13_hybrid_psk_dhe, lambda: C.kemtls_pdk(ca.signer)):
        ln, lg, sizes, ok = fn()
        assert ok and sizes["node_to_gw"] and sizes["gw_to_node"]
