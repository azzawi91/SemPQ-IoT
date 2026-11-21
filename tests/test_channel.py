import numpy as np
import pytest
from scipy import stats

from src.channel.ble_phy import BerModel, BleFrame, ber_noncoherent_fsk, gfsk_ber_curve, packet_error_rate
from src.channel.fading import block_fading_trace, power_gain_cdf, rician_power_gains
from src.channel.gilbert_elliott import GilbertElliott, run_lengths
from src.channel.link import LinkConfig, LinkSimulator, expected_delivery


# ---------------- Rician block fading ----------------
@pytest.mark.parametrize("k", [0.0, 1.0, 3.0, 10.0])
def test_rician_unit_mean_and_distribution(k):
    g = rician_power_gains(200_000, k, np.random.default_rng(0))
    assert abs(g.mean() - 1.0) < 0.01
    ks = stats.kstest(g[:20_000], lambda x: power_gain_cdf(x, k))
    assert ks.pvalue > 0.01


def test_rician_variance_decreases_with_k():
    rng = np.random.default_rng(1)
    v = [rician_power_gains(100_000, k, rng).var() for k in (0, 3, 10)]
    assert v[0] > v[1] > v[2]
    assert abs(v[0] - 1.0) < 0.03  # Rayleigh: exponential |h|^2 has unit variance


def test_block_structure_and_seed_determinism():
    t1 = block_fading_trace(103, 10, 3.0, np.random.default_rng(5))
    t2 = block_fading_trace(103, 10, 3.0, np.random.default_rng(5))
    np.testing.assert_array_equal(t1, t2)
    assert len(t1) == 103
    for b in range(10):
        assert np.all(t1[b * 10:(b + 1) * 10] == t1[b * 10])
    assert t1[0] != t1[10]


def test_rician_rejects_negative_k():
    with pytest.raises(ValueError):
        rician_power_gains(10, -1.0, np.random.default_rng(0))


# ---------------- BLE PHY ----------------
def test_frame_bits_and_airtime():
    f = BleFrame()
    assert f.protected_bits(0) == 32 + 16 + 24
    assert f.protected_bits(251) == 32 + 16 + 8 * 251 + 24
    assert f.airtime_s(251) == pytest.approx((8 + 32 + 16 + 2008 + 24) / 1e6)
    with pytest.raises(ValueError):
        f.protected_bits(252)


def test_per_monotone_in_snr_and_payload():
    m = BerModel("noncoherent_fsk")
    snr = np.arange(0, 21)
    per = packet_error_rate(snr, 100, m)
    assert np.all(np.diff(per) <= 0) and per[0] > 0.99 and per[-1] < 1e-6
    assert packet_error_rate(10.0, 20, m) < packet_error_rate(10.0, 200, m)


def test_noncoherent_fsk_formula():
    assert ber_noncoherent_fsk(0.0) == pytest.approx(0.5 * np.exp(-0.5))


def test_gfsk_mc_curve_shape():
    db = np.array([0.0, 4.0, 8.0, 12.0])
    ber = gfsk_ber_curve(db, np.random.default_rng(0), n_bits=60_000)
    assert np.all(np.diff(ber) < 0)
    assert 0.1 < ber[0] < 0.5 and ber[-1] < 0.01
    # discriminator GFSK should lie within 0-5 dB of noncoherent orthogonal FSK at BER 1e-2
    ncf_db = 2 * np.log(50) # Eb/N0 (linear) giving 1e-2 for noncoherent FSK
    gfsk_db = np.interp(np.log10(1e-2), np.log10(ber[::-1]), db[::-1])
    assert 0.0 <= gfsk_db - 10 * np.log10(ncf_db) <= 5.0


def test_mc_ber_model_interpolation():
    m = BerModel("gfsk_mc", table_db=[0, 10], table_ber=[1e-1, 1e-3])
    assert m(5.0) == pytest.approx(1e-2, rel=1e-6)
    assert m(20.0) == pytest.approx(1e-5, rel=1e-6)  # log-linear extrapolation
    with pytest.raises(ValueError):
        BerModel("gfsk_mc")


# ---------------- Gilbert-Elliott ----------------
def test_ge_from_outage_parameters():
    ge = GilbertElliott.from_outage(mean_outage_slots=20, outage_fraction=0.1)
    assert ge.p_bg == pytest.approx(0.05)
    assert ge.stationary_bad == pytest.approx(0.1)
    assert ge.mean_loss == pytest.approx(0.1)


def test_ge_empirical_statistics():
    ge = GilbertElliott.from_outage(mean_outage_slots=10, outage_fraction=0.2)
    bad = ge.states(400_000, np.random.default_rng(3))
    assert abs(bad.mean() - 0.2) < 0.01
    assert abs(run_lengths(bad).mean() - 10.0) < 0.3


def test_ge_partial_loss_and_determinism():
    ge = GilbertElliott(p_gb=0.02, p_bg=0.1, e_g=0.01, e_b=0.5)
    b1, l1 = ge.losses(200_000, np.random.default_rng(7))
    b2, l2 = ge.losses(200_000, np.random.default_rng(7))
    np.testing.assert_array_equal(l1, l2)
    assert abs(l1.mean() - ge.mean_loss) < 0.005


def test_ge_validation():
    with pytest.raises(ValueError):
        GilbertElliott(p_gb=1.5, p_bg=0.1)
    with pytest.raises(ValueError):
        GilbertElliott.from_outage(0.5, 0.1)


def test_run_lengths():
    assert run_lengths(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)).tolist() == [2, 1, 3]


# ---------------- Link ----------------
def test_link_delivery_matches_expectation():
    m = BerModel("noncoherent_fsk")
    ge = GilbertElliott.from_outage(mean_outage_slots=5, outage_fraction=0.1)
    cfg = LinkConfig(avg_ebn0_db=12.0, k_factor=3.0, block_len=1, outage=ge)
    out = LinkSimulator(cfg, m, np.random.default_rng(11)).run(np.full(200_000, 60))
    target = 0.9 * expected_delivery(12.0, 60, 3.0, m)
    assert abs(out["delivered"].mean() - target) < 0.01
    assert not np.any(out["delivered"] & out["outage_lost"])


def test_link_seed_determinism():
    m = BerModel("noncoherent_fsk")
    cfg = LinkConfig(avg_ebn0_db=8.0, outage=GilbertElliott.from_outage(3, 0.05))
    a = LinkSimulator(cfg, m, np.random.default_rng(2)).run(np.full(5000, 100))
    b = LinkSimulator(cfg, m, np.random.default_rng(2)).run(np.full(5000, 100))
    np.testing.assert_array_equal(a["delivered"], b["delivered"])


def test_ge_fast_matches_statistics():
    ge = GilbertElliott.from_outage(mean_outage_slots=10, outage_fraction=0.2)
    bad = ge.states_fast(400_000, np.random.default_rng(4))
    assert len(bad) == 400_000
    assert abs(bad.mean() - 0.2) < 0.01
    assert abs(run_lengths(bad).mean() - 10.0) < 0.3
