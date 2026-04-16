import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.effects.filters import LowPassFilter, HighPassFilter, BandPassFilter
from pysynth.mixing.panning import pan


SR = 4000  # low sample rate for fast tests


def _mono_signal(duration=0.1, sr=SR):
    """White noise mono signal."""
    n = int(duration * sr)
    return Signal(np.random.default_rng(42).standard_normal(n).astype(np.float32), sr)


def _stereo_signal(duration=0.1, sr=SR):
    """Stereo signal via pan."""
    return pan(_mono_signal(duration, sr), 0.3)


class TestLowPassFilterStereo:
    def test_stereo_no_resonance(self):
        sig = _stereo_signal()
        result = LowPassFilter(500)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_stereo_with_resonance(self):
        sig = _stereo_signal()
        result = LowPassFilter(500, resonance=0.3)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_stereo_modulated_cutoff(self):
        sig = _stereo_signal()
        cutoff = Signal(np.linspace(200, 800, len(sig), dtype=np.float32), SR)
        result = LowPassFilter(cutoff)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2


class TestHighPassFilterStereo:
    def test_stereo_no_resonance(self):
        sig = _stereo_signal()
        result = HighPassFilter(500)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_stereo_with_resonance(self):
        sig = _stereo_signal()
        result = HighPassFilter(500, resonance=0.3)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_stereo_modulated_cutoff(self):
        sig = _stereo_signal()
        cutoff = Signal(np.linspace(200, 800, len(sig), dtype=np.float32), SR)
        result = HighPassFilter(cutoff)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2


class TestBandPassFilterStereo:
    def test_stereo_no_resonance(self):
        sig = _stereo_signal()
        result = BandPassFilter(200, 800)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_stereo_with_resonance(self):
        sig = _stereo_signal()
        result = BandPassFilter(200, 800, resonance=0.3)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2

    def test_stereo_modulated_cutoff(self):
        sig = _stereo_signal()
        low = Signal(np.linspace(100, 300, len(sig), dtype=np.float32), SR)
        high = Signal(np.linspace(600, 1000, len(sig), dtype=np.float32), SR)
        result = BandPassFilter(low, high)(sig)
        assert result.data.shape == sig.data.shape
        assert result.n_channels == 2


class TestFiltersMonoStillWork:
    """Ensure mono signals are not broken by the stereo fix."""

    def test_lowpass_mono(self):
        sig = _mono_signal()
        result = LowPassFilter(500)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_highpass_mono(self):
        sig = _mono_signal()
        result = HighPassFilter(500)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)

    def test_bandpass_mono(self):
        sig = _mono_signal()
        result = BandPassFilter(200, 800)(sig)
        assert result.data.ndim == 1
        assert len(result) == len(sig)
