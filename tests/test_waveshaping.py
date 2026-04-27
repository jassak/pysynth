import numpy as np
import pytest

from pysynth._core import Signal, SAMPLE_RATE
from pysynth.effects.waveshaping import (
    SoftClip, Fold, Rectifier, Chebyshev, Shaper,
)


def _sine(freq, dur=0.5, amp=1.0, sr=SAMPLE_RATE):
    n = int(dur * sr)
    t = np.arange(n) / sr
    return Signal((amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)


class TestSoftClip:
    def test_bounded(self):
        sig = _sine(440.0, amp=10.0)
        out = SoftClip(drive=2.0)(sig)
        assert np.max(np.abs(out.data)) < 1.0

    def test_odd_symmetric(self):
        sig = _sine(440.0, amp=0.7)
        pos = SoftClip(drive=3.0)(sig).data
        neg = SoftClip(drive=3.0)(Signal(-sig.data, sig.sample_rate)).data
        np.testing.assert_allclose(pos, -neg, atol=1e-6)

    def test_signal_drive(self):
        sig = _sine(440.0)
        drive = Signal(np.linspace(1.0, 5.0, len(sig.data)).astype(np.float32), sig.sample_rate)
        out = SoftClip(drive=drive)(sig)
        assert out.data.shape == sig.data.shape


class TestFold:
    def test_linear_region_passthrough(self):
        # Input well within |x| <= threshold should pass through unchanged.
        sig = _sine(100.0, amp=0.4)
        out = Fold(threshold=1.0)(sig)
        np.testing.assert_allclose(out.data, sig.data, atol=1e-5)

    def test_bounded_by_threshold(self):
        sig = _sine(100.0, amp=5.0)
        t = 0.5
        out = Fold(threshold=t)(sig)
        assert np.max(np.abs(out.data)) <= t + 1e-5

    def test_signal_threshold(self):
        sig = _sine(440.0)
        thr = Signal(np.linspace(0.3, 1.0, len(sig.data)).astype(np.float32), sig.sample_rate)
        out = Fold(threshold=thr)(sig)
        assert out.data.shape == sig.data.shape


class TestRectifier:
    def test_full_mean_zero(self):
        sig = _sine(440.0)
        out = Rectifier("full")(sig)
        assert abs(float(out.data.mean())) < 1e-4

    def test_half_mean_zero(self):
        sig = _sine(440.0)
        out = Rectifier("half")(sig)
        assert abs(float(out.data.mean())) < 1e-4

    def test_peak_matches_input(self):
        sig = _sine(440.0, amp=0.7)
        out = Rectifier("full")(sig)
        assert np.max(np.abs(out.data)) == pytest.approx(0.7, abs=1e-3)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            Rectifier("quarter")


class TestChebyshev:
    def test_third_harmonic(self):
        sr = SAMPLE_RATE
        f = 1000.0
        sig = _sine(f, dur=1.0, amp=1.0, sr=sr)
        out = Chebyshev([0, 0, 0, 1])(sig)
        spec = np.abs(np.fft.rfft(out.data))
        freqs = np.fft.rfftfreq(len(out.data), 1.0 / sr)
        peak_bin = int(np.argmax(spec))
        assert freqs[peak_bin] == pytest.approx(3 * f, abs=5.0)

    def test_identity_coeff(self):
        # coeffs = [0, 1] is T_1(x) = x, i.e. pass-through (after normalize).
        sig = _sine(440.0, amp=0.5)
        out = Chebyshev([0, 1])(sig)
        # Should be proportional to input; check correlation near 1.
        corr = float(np.corrcoef(out.data, sig.data)[0, 1])
        assert corr > 0.999

    def test_signal_valued_coeffs(self):
        sig = _sine(440.0, dur=0.3)
        n = len(sig.data)
        c1 = Signal(np.linspace(1.0, 0.0, n).astype(np.float32), sig.sample_rate)
        c3 = Signal(np.linspace(0.0, 1.0, n).astype(np.float32), sig.sample_rate)
        out = Chebyshev([0, c1, 0, c3])(sig)
        assert out.data.shape == sig.data.shape
        assert np.isfinite(out.data).all()

    def test_empty_coeffs_raises(self):
        with pytest.raises(ValueError):
            Chebyshev([])


class TestShaper:
    def test_passthrough(self):
        sig = _sine(440.0, amp=0.3)
        out = Shaper(lambda x: x)(sig)
        np.testing.assert_array_equal(out.data, sig.data)

    def test_custom_fn(self):
        sig = _sine(440.0, amp=0.5)
        out = Shaper(lambda x: np.sign(x) * x**2)(sig)
        expected = (np.sign(sig.data) * sig.data**2).astype(np.float32)
        np.testing.assert_allclose(out.data, expected, atol=1e-6)
