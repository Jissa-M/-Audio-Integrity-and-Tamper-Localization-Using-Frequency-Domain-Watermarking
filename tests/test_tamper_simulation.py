import numpy as np
import pytest
from tamper_simulation import (SR, HOP, sec_to_samples, splice, verify_splice)


def make_signal(seed, seconds=20, amp=0.1):
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * SR) / SR
    return (amp * rng.standard_normal(t.size)
            + 0.2 * np.sin(2 * np.pi * (200 + 50 * seed) * t)).astype(np.float32)


@pytest.fixture
def host():
    return make_signal(1)


@pytest.fixture
def donor():
    return make_signal(2)


def test_sec_to_samples():
    assert sec_to_samples(1.0) == 16000
    assert sec_to_samples(0.5) == 8000


def test_same_length_and_verification(host, donor):
    out, info = splice(host, donor, 5.0, 8.0, donor_start_s=2.0)
    assert len(out) == len(host)
    assert verify_splice(host, out, donor, info)


def test_outside_unchanged_inside_changed(host, donor):
    out, info = splice(host, donor, 5.0, 6.0)
    assert np.array_equal(out[:info["start_sample"]], host[:info["start_sample"]])
    assert not np.array_equal(out[info["start_sample"]:info["end_sample"]],
                              host[info["start_sample"]:info["end_sample"]])


def test_crossfade_shorter_than_hop(host, donor):
    _, info = splice(host, donor, 5.0, 6.0)
    assert info["crossfade_samples"] < HOP


def test_rejects_splice_near_edges(host, donor):
    with pytest.raises(ValueError):
        splice(host, donor, 0.5, 1.5)
    with pytest.raises(ValueError):
        splice(host, donor, 18.5, 19.5)


def test_rejects_short_donor(host):
    short = make_signal(3, seconds=2)
    with pytest.raises(ValueError):
        splice(host, short, 5.0, 8.0)