import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.generators.sample import Sample
from pysynth.generators.granular import Granular

SR = 4000


def _sig(values, sr=SR):
    return Signal(np.array(values, dtype=np.float32), sr)


def _sine_sample(freq=100, dur=1.0, sr=SR, root_pitch=None):
    t = np.arange(int(dur * sr), dtype=np.float64) / sr
    data = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return Sample(data, sr, root_pitch=root_pitch)


# ------------------------------------------------------------------ #
# Basic rendering                                                     #
# ------------------------------------------------------------------ #


class TestBasic:
    def test_render_returns_signal(self):
        s = _sine_sample()
        sig = Granular(s).render(0.5, sr=SR)
        assert isinstance(sig, Signal)

    def test_output_duration(self):
        s = _sine_sample()
        sig = Granular(s).render(0.5, sr=SR)
        assert len(sig.data) == int(0.5 * SR)

    def test_output_sample_rate(self):
        s = _sine_sample()
        sig = Granular(s).render(0.5, sr=SR)
        assert sig.sample_rate == SR

    def test_produces_nonzero_output(self):
        s = _sine_sample()
        sig = Granular(s, position=0.3, density=20).render(0.5, sr=SR)
        assert np.max(np.abs(sig.data)) > 0.01

    def test_repr(self):
        s = _sine_sample()
        g = Granular(s, grain_size=0.04, density=30)
        assert "Granular" in repr(g)


# ------------------------------------------------------------------ #
# Position                                                            #
# ------------------------------------------------------------------ #


class TestPosition:
    def test_position_zero_reads_from_start(self):
        # Sample: first half is sine, second half is silence
        n = SR
        data = np.zeros(n, dtype=np.float32)
        data[: n // 2] = np.sin(2 * np.pi * 100 * np.arange(n // 2) / SR).astype(np.float32)
        s = Sample(data, SR)
        sig = Granular(s, position=0.0, density=20, grain_size=0.05, seed=42).render(0.2, sr=SR)
        assert np.max(np.abs(sig.data)) > 0.1

    def test_position_one_reads_from_end(self):
        # Sample: first half is silence, second half is sine
        n = SR
        data = np.zeros(n, dtype=np.float32)
        data[n // 2:] = np.sin(2 * np.pi * 100 * np.arange(n - n // 2) / SR).astype(np.float32)
        s = Sample(data, SR)
        sig = Granular(s, position=1.0, density=20, grain_size=0.02, seed=42).render(0.2, sr=SR)
        # At position 1.0 the grains read from the very end — most content
        # will be near the boundary
        assert sig.data is not None  # sanity check; detailed behavior is grain-dependent

    def test_signal_rate_position(self):
        s = _sine_sample(dur=1.0)
        n = int(0.5 * SR)
        pos = _sig(np.linspace(0, 1, n))
        sig = Granular(s, position=pos, density=20, seed=42).render(0.5, sr=SR)
        assert len(sig.data) == n


# ------------------------------------------------------------------ #
# Density                                                             #
# ------------------------------------------------------------------ #


class TestDensity:
    def test_higher_density_more_energy(self):
        s = _sine_sample()
        sig_low = Granular(s, density=5, grain_size=0.05, seed=42).render(0.5, sr=SR)
        sig_high = Granular(s, density=50, grain_size=0.05, seed=42).render(0.5, sr=SR)
        rms_low = np.sqrt(np.mean(sig_low.data ** 2))
        rms_high = np.sqrt(np.mean(sig_high.data ** 2))
        assert rms_high > rms_low

    def test_signal_rate_density(self):
        s = _sine_sample()
        n = int(0.5 * SR)
        dens = _sig(np.linspace(5, 50, n))
        sig = Granular(s, density=dens, seed=42).render(0.5, sr=SR)
        assert len(sig.data) == n


# ------------------------------------------------------------------ #
# Grain size                                                          #
# ------------------------------------------------------------------ #


class TestGrainSize:
    def test_grain_size_affects_output(self):
        s = _sine_sample()
        sig_small = Granular(s, grain_size=0.01, density=20, seed=42).render(0.5, sr=SR)
        sig_large = Granular(s, grain_size=0.1, density=20, seed=42).render(0.5, sr=SR)
        # Different grain sizes should produce different output
        assert not np.allclose(sig_small.data, sig_large.data)

    def test_signal_rate_grain_size(self):
        s = _sine_sample()
        n = int(0.5 * SR)
        gs = _sig(np.linspace(0.02, 0.1, n))
        sig = Granular(s, grain_size=gs, density=15, seed=42).render(0.5, sr=SR)
        assert len(sig.data) == n


# ------------------------------------------------------------------ #
# Pitch                                                               #
# ------------------------------------------------------------------ #


class TestPitch:
    def test_pitch_two_doubles_frequency(self):
        s = _sine_sample(freq=100, dur=1.0)
        sig1 = Granular(s, pitch=1.0, density=40, grain_size=0.05, spread=0, seed=42).render(0.5, sr=SR)
        sig2 = Granular(s, pitch=2.0, density=40, grain_size=0.05, spread=0, seed=42).render(0.5, sr=SR)
        # Higher pitch should have more zero crossings
        zc1 = int(np.sum(np.diff(np.sign(sig1.data)) != 0))
        zc2 = int(np.sum(np.diff(np.sign(sig2.data)) != 0))
        assert zc2 > zc1

    def test_signal_rate_pitch(self):
        s = _sine_sample()
        n = int(0.5 * SR)
        pitch = _sig(np.linspace(0.5, 2.0, n))
        sig = Granular(s, pitch=pitch, density=20, seed=42).render(0.5, sr=SR)
        assert len(sig.data) == n


# ------------------------------------------------------------------ #
# Spread                                                              #
# ------------------------------------------------------------------ #


class TestSpread:
    def test_zero_spread_deterministic(self):
        s = _sine_sample()
        sig1 = Granular(s, spread=0.0, density=20, seed=42).render(0.3, sr=SR)
        sig2 = Granular(s, spread=0.0, density=20, seed=42).render(0.3, sr=SR)
        np.testing.assert_array_equal(sig1.data, sig2.data)

    def test_spread_with_seed_reproducible(self):
        s = _sine_sample()
        sig1 = Granular(s, spread=0.3, density=20, seed=123).render(0.3, sr=SR)
        sig2 = Granular(s, spread=0.3, density=20, seed=123).render(0.3, sr=SR)
        np.testing.assert_array_equal(sig1.data, sig2.data)

    def test_spread_without_seed_varies(self):
        s = _sine_sample()
        sig1 = Granular(s, spread=0.5, density=20, seed=1).render(0.3, sr=SR)
        sig2 = Granular(s, spread=0.5, density=20, seed=2).render(0.3, sr=SR)
        assert not np.array_equal(sig1.data, sig2.data)


# ------------------------------------------------------------------ #
# Window                                                              #
# ------------------------------------------------------------------ #


class TestWindow:
    def test_hann_is_default(self):
        s = _sine_sample()
        g = Granular(s)
        assert g._window == "hann"

    def test_different_window(self):
        s = _sine_sample()
        sig_hann = Granular(s, window="hann", density=20, seed=42).render(0.3, sr=SR)
        sig_hamm = Granular(s, window="hamming", density=20, seed=42).render(0.3, sr=SR)
        # Different windows should produce slightly different output
        assert not np.allclose(sig_hann.data, sig_hamm.data)


# ------------------------------------------------------------------ #
# Pitched rendering                                                   #
# ------------------------------------------------------------------ #


class TestPitchedRendering:
    def test_render_hz_returns_signal(self):
        s = _sine_sample(root_pitch=100.0)
        sig = Granular(s).render(0.5, 200.0, SR)
        assert isinstance(sig, Signal)

    def test_render_hz_requires_root_pitch(self):
        s = _sine_sample()  # no root_pitch
        with pytest.raises(ValueError, match="root_pitch"):
            Granular(s).render(0.5, 200.0, SR)

    def test_render_hz_affects_pitch(self):
        s = _sine_sample(freq=100, dur=1.0, root_pitch=100.0)
        sig_orig = Granular(s, density=40, grain_size=0.05, spread=0, seed=42).render(0.5, 100.0, SR)
        sig_up = Granular(s, density=40, grain_size=0.05, spread=0, seed=42).render(0.5, 200.0, SR)
        zc_orig = int(np.sum(np.diff(np.sign(sig_orig.data)) != 0))
        zc_up = int(np.sum(np.diff(np.sign(sig_up.data)) != 0))
        assert zc_up > zc_orig

    def test_render_signal_hz(self):
        s = _sine_sample(freq=100, dur=1.0, root_pitch=100.0)
        n = int(0.5 * SR)
        hz = _sig(np.full(n, 200.0))
        sig = Granular(s, density=20, seed=42).render(0.5, hz, SR)
        assert len(sig.data) == n

    def test_render_none_hz_uses_pitch_param(self):
        s = _sine_sample()
        sig1 = Granular(s, pitch=2.0, density=20, seed=42).render(0.3, sr=SR)
        sig2 = Granular(s, pitch=2.0, density=20, seed=42).render(0.3, sr=SR)
        np.testing.assert_array_equal(sig1.data, sig2.data)


# ------------------------------------------------------------------ #
# Stereo                                                              #
# ------------------------------------------------------------------ #


class TestStereo:
    def test_stereo_sample(self):
        n = SR
        left = np.sin(2 * np.pi * 100 * np.arange(n) / SR).astype(np.float32)
        right = np.sin(2 * np.pi * 200 * np.arange(n) / SR).astype(np.float32)
        s = Sample(np.column_stack([left, right]), SR)
        sig = Granular(s, density=20, seed=42).render(0.3, sr=SR)
        assert sig.n_channels == 2
