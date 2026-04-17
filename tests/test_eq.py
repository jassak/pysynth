import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.effects.eq import PeakFilter, LowShelf, HighShelf
from pysynth.mixing.panning import pan


SR = 4000  # low sample rate for fast tests


def _mono_signal(duration=0.1, sr=SR):
    """White noise mono signal."""
    n = int(duration * sr)
    return Signal(np.random.default_rng(42).standard_normal(n).astype(np.float32), sr)


def _stereo_signal(duration=0.1, sr=SR):
    """Stereo signal via pan."""
    return pan(_mono_signal(duration, sr), 0.3)


# ---------------------------------------------------------------------------
# PeakFilter
# ---------------------------------------------------------------------------


class TestPeakFilter:
    def test_mono(self):
        sig = _mono_signal()
        result = PeakFilter(500, gain_db=6.0)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_stereo(self):
        sig = _stereo_signal()
        result = PeakFilter(500, gain_db=6.0)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_zero_gain_passthrough(self):
        sig = _mono_signal()
        result = PeakFilter(500, gain_db=0.0)(sig)
        np.testing.assert_allclose(result.data, sig.data, atol=1e-6)

    def test_modulated_params(self):
        sig = _mono_signal()
        center = Signal(np.linspace(200, 800, len(sig), dtype=np.float32), SR)
        gain = Signal(np.linspace(-6, 6, len(sig), dtype=np.float32), SR)
        result = PeakFilter(center, gain_db=gain, q=2.0)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_cut(self):
        sig = _mono_signal()
        result = PeakFilter(500, gain_db=-6.0)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)


# ---------------------------------------------------------------------------
# LowShelf
# ---------------------------------------------------------------------------


class TestLowShelf:
    def test_mono(self):
        sig = _mono_signal()
        result = LowShelf(300, gain_db=6.0)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_stereo(self):
        sig = _stereo_signal()
        result = LowShelf(300, gain_db=6.0)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_zero_gain_passthrough(self):
        sig = _mono_signal()
        result = LowShelf(300, gain_db=0.0)(sig)
        np.testing.assert_allclose(result.data, sig.data, atol=1e-6)

    def test_modulated_gain(self):
        sig = _mono_signal()
        gain = Signal(np.linspace(-6, 6, len(sig), dtype=np.float32), SR)
        result = LowShelf(300, gain_db=gain)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_cut(self):
        sig = _mono_signal()
        result = LowShelf(300, gain_db=-6.0)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)


# ---------------------------------------------------------------------------
# HighShelf
# ---------------------------------------------------------------------------


class TestHighShelf:
    def test_mono(self):
        sig = _mono_signal()
        result = HighShelf(800, gain_db=6.0)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_stereo(self):
        sig = _stereo_signal()
        result = HighShelf(800, gain_db=6.0)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_zero_gain_passthrough(self):
        sig = _mono_signal()
        result = HighShelf(800, gain_db=0.0)(sig)
        np.testing.assert_allclose(result.data, sig.data, atol=1e-6)

    def test_modulated_gain(self):
        sig = _mono_signal()
        gain = Signal(np.linspace(-6, 6, len(sig), dtype=np.float32), SR)
        result = HighShelf(800, gain_db=gain)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_cut(self):
        sig = _mono_signal()
        result = HighShelf(800, gain_db=-6.0)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)


# ---------------------------------------------------------------------------
# Chaining (|)
# ---------------------------------------------------------------------------


class TestEqChaining:
    def test_chain_all_three(self):
        sig = _mono_signal()
        eq = LowShelf(200, gain_db=3.0) | PeakFilter(500, gain_db=-2.0, q=1.5) | HighShelf(1000, gain_db=-4.0)
        result = eq(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_chain_stereo(self):
        sig = _stereo_signal()
        eq = LowShelf(200, gain_db=3.0) | PeakFilter(500, gain_db=-2.0) | HighShelf(1000, gain_db=2.0)
        result = eq(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2
