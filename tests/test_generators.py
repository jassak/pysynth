import numpy as np
import pytest

from pysynth._core import SAMPLE_RATE, Signal, Generator
from pysynth.generators.oscillators import Oscillator, _shape
from pysynth.generators.noise import WhiteNoise, PinkNoise


SR = 4000  # enough to test waveforms without aliasing concerns


# ------------------------------------------------------------------ #
# Waveform shapes                                                      #
# ------------------------------------------------------------------ #


class TestShape:
    """Test _shape produces correct output at known phase values."""

    def test_sine_zero_crossing(self):
        phase = np.array([0.0, np.pi, 2 * np.pi])
        out = _shape("sine", phase)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-6)

    def test_sine_peaks(self):
        phase = np.array([np.pi / 2, 3 * np.pi / 2])
        out = _shape("sine", phase)
        np.testing.assert_allclose(out, [1.0, -1.0], atol=1e-6)

    def test_square_values(self):
        phase = np.array([0.1, np.pi - 0.1, np.pi + 0.1, 2 * np.pi - 0.1])
        out = _shape("square", phase)
        assert out[0] > 0 and out[1] > 0
        assert out[2] < 0 and out[3] < 0

    def test_saw_range(self):
        phase = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        out = _shape("saw", phase)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_triangle_range(self):
        phase = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        out = _shape("triangle", phase)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_pulse_values(self):
        phase = np.array([0.1, np.pi - 0.1, np.pi + 0.1, 2 * np.pi - 0.1])
        out = _shape("pulse", phase)
        np.testing.assert_allclose(out, [1.0, 1.0, -1.0, -1.0], atol=1e-6)

    def test_unknown_waveform_raises(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            _shape("kazoo", np.array([0.0]))


# ------------------------------------------------------------------ #
# Oscillator basics                                                    #
# ------------------------------------------------------------------ #


class TestOscillator:
    def test_at_returns_generator(self):
        gen = Oscillator("sine").at(440)
        assert isinstance(gen, Generator)

    def test_render_duration(self):
        sig = Oscillator("sine").at(440).render(1.0, SR)
        assert abs(sig.duration - 1.0) < 0.001

    def test_render_sample_rate(self):
        sig = Oscillator("sine").at(440).render(0.5, SR)
        assert sig.sample_rate == SR

    def test_sine_amplitude_bound(self):
        sig = Oscillator("sine").at(100).render(0.5, SR)
        assert np.abs(sig.data).max() <= 1.0 + 1e-6

    def test_ratio_doubles_frequency(self):
        sig1 = Oscillator("sine", ratio=1).at(100).render(0.1, SR)
        sig2 = Oscillator("sine", ratio=2).at(100).render(0.1, SR)
        crossings1 = np.sum(np.diff(np.sign(sig1.data)) != 0)
        crossings2 = np.sum(np.diff(np.sign(sig2.data)) != 0)
        assert crossings2 > crossings1 * 1.5

    def test_default_waveform_is_sine(self):
        osc = Oscillator()
        assert osc._components[0][0] == "sine"


# ------------------------------------------------------------------ #
# Oscillator algebra                                                   #
# ------------------------------------------------------------------ #


class TestOscillatorAlgebra:
    def test_mul_scalar_scales_amplitude(self):
        osc = Oscillator("sine") * 0.5
        assert osc._components[0][2] == 0.5

    def test_rmul_scalar(self):
        osc = 0.3 * Oscillator("sine")
        assert osc._components[0][2] == pytest.approx(0.3)

    def test_mul_does_not_mutate(self):
        osc = Oscillator("sine")
        _ = osc * 0.5
        assert osc._components[0][2] == 1.0

    def test_add_merges_components(self):
        osc = Oscillator("sine") + Oscillator("saw", 2)
        assert len(osc._components) == 2
        assert osc._components[0][0] == "sine"
        assert osc._components[1][0] == "saw"

    def test_add_does_not_mutate(self):
        a = Oscillator("sine")
        b = Oscillator("saw")
        _ = a + b
        assert len(a._components) == 1
        assert len(b._components) == 1

    def test_additive_synthesis_renders(self):
        osc = Oscillator("sine") + Oscillator("sine", 2) * 0.5
        sig = osc.at(220).render(0.5, SR)
        assert len(sig.data) == int(0.5 * SR)

    def test_repr_single_component(self):
        r = repr(Oscillator("saw", ratio=2.0))
        assert "saw" in r

    def test_repr_multi_component(self):
        osc = Oscillator("sine") + Oscillator("saw")
        assert "components=2" in repr(osc)


# ------------------------------------------------------------------ #
# Signal-rate pitch (FM / CV)                                          #
# ------------------------------------------------------------------ #


class TestSignalRatePitch:
    def test_at_signal_returns_generator(self):
        pitch_cv = Signal(np.full(SR, 440.0, dtype=np.float32), SR)
        gen = Oscillator("sine").at(pitch_cv)
        assert isinstance(gen, Generator)

    def test_at_signal_renders(self):
        pitch_cv = Signal(np.full(SR, 440.0, dtype=np.float32), SR)
        sig = Oscillator("sine").at(pitch_cv).render(1.0, SR)
        assert abs(sig.duration - 1.0) < 0.001

    def test_constant_signal_matches_float_frequency(self):
        # Both paths should produce the same fundamental frequency.
        # We compare via zero-crossing count rather than sample correlation,
        # since cumsum phase integration diverges from the direct formula.
        dur = 0.1
        n = int(dur * SR)
        sig_float = Oscillator("sine").at(440.0).render(dur, SR)
        pitch_cv = Signal(np.full(n, 440.0, dtype=np.float32), SR)
        sig_signal = Oscillator("sine").at(pitch_cv).render(dur, SR)
        zc_float = np.sum(np.diff(np.sign(sig_float.data)) != 0)
        zc_signal = np.sum(np.diff(np.sign(sig_signal.data)) != 0)
        assert abs(zc_float - zc_signal) <= 1

    def test_ratio_applied_to_signal_pitch(self):
        dur = 0.1
        n = int(dur * SR)
        pitch_cv = Signal(np.full(n, 100.0, dtype=np.float32), SR)
        sig1 = Oscillator("sine", ratio=1).at(pitch_cv).render(dur, SR)
        sig2 = Oscillator("sine", ratio=2).at(pitch_cv).render(dur, SR)
        crossings1 = np.sum(np.diff(np.sign(sig1.data)) != 0)
        crossings2 = np.sum(np.diff(np.sign(sig2.data)) != 0)
        assert crossings2 > crossings1 * 1.5


# ------------------------------------------------------------------ #
# Noise generators                                                     #
# ------------------------------------------------------------------ #


class TestWhiteNoise:
    def test_duration(self):
        sig = WhiteNoise(sample_rate=SR).render(1.0)
        assert abs(sig.duration - 1.0) < 0.001

    def test_amplitude_scaling(self):
        sig = WhiteNoise(amplitude=0.5, sample_rate=SR).render(1.0)
        assert np.abs(sig.data).max() <= 0.5 + 1e-6

    def test_nonzero(self):
        sig = WhiteNoise(sample_rate=SR).render(0.1)
        assert np.abs(sig.data).max() > 0.0

    def test_approximately_zero_mean(self):
        sig = WhiteNoise(sample_rate=SR).render(2.0)
        assert abs(np.mean(sig.data)) < 0.05


class TestPinkNoise:
    def test_duration(self):
        sig = PinkNoise(sample_rate=SR).render(1.0)
        assert abs(sig.duration - 1.0) < 0.001

    def test_amplitude_scaling(self):
        sig = PinkNoise(amplitude=0.5, sample_rate=SR).render(1.0)
        assert np.abs(sig.data).max() <= 0.5 + 1e-6

    def test_peak_normalized_to_amplitude(self):
        sig = PinkNoise(amplitude=1.0, sample_rate=SR).render(1.0)
        assert np.abs(sig.data).max() <= 1.0 + 1e-6

    def test_nonzero(self):
        sig = PinkNoise(sample_rate=SR).render(0.1)
        assert np.abs(sig.data).max() > 0.0
