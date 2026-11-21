"""Environment checks: seeding is deterministic and liboqs exposes the algorithms L2 needs."""
import numpy as np
import pytest

from src.utils.seed import set_seed


def test_set_seed_numpy_and_torch():
    a = set_seed(123).random(5)
    b = set_seed(123).random(5)
    np.testing.assert_array_equal(a, b)
    torch = pytest.importorskip("torch")
    set_seed(7)
    t1 = torch.randn(4)
    set_seed(7)
    assert torch.equal(t1, torch.randn(4))


@pytest.mark.liboqs
def test_liboqs_ml_kem_and_ml_dsa():
    oqs = pytest.importorskip("oqs")
    assert "ML-KEM-512" in oqs.get_enabled_kem_mechanisms()
    assert "ML-DSA-44" in oqs.get_enabled_sig_mechanisms()
    with oqs.KeyEncapsulation("ML-KEM-512") as server:
        pk = server.generate_keypair()
        with oqs.KeyEncapsulation("ML-KEM-512") as client:
            ct, ss = client.encap_secret(pk)
        assert server.decap_secret(ct) == ss
    # FIPS 203 / FIPS 204 sizes
    assert (len(pk), len(ct), len(ss)) == (800, 768, 32)
    with oqs.Signature("ML-DSA-44") as signer:
        vk = signer.generate_keypair()
        sig = signer.sign(b"gateway-cert")
        with oqs.Signature("ML-DSA-44") as v:
            assert v.verify(b"gateway-cert", sig, vk)
    assert (len(vk), len(sig)) == (1312, 2420)
