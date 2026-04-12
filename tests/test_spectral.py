import numpy as np
import pytest

from pysynth._core import Signal, Effect
from pysynth.spectral._spectrum import Spectrum, stft
from pysynth.spectral._transforms import (
    freeze,
    smear,
    shift_bins,
    cross_synthesize,
    pitch_shift,
)
from pysynth.spectral._effects import SpectralFreeze, SpectralSmear, PitchShift, Vocoder
from pysynth.spectral._convolution import ConvolutionReverb

SR = 4000  # low sample rate for fast tests
N_FFT = 256
HOP = N_FFT // 4


def _sig(values, sr=SR):
    return Signal(np.array(values, dtype=np.float32), sr)


def _tone(freq, duration=0.5, sr=SR):
    """Generate a pure sine tone."""
    t = np.arange(int(duration * sr)) / sr
    return Signal(np.sin(2 * np.pi * freq * t).astype(np.float32), sr)


# ------------------------------------------------------------------ #
# STFT / ISTFT round-trip                                              #
# ------------------------------------------------------------------ #


class TestRoundTrip:
    def test_sine_round_trip(self):
        sig = _tone(440, duration=0.25)
        reconstructed = stft(sig, n_fft=N_FFT).to_signal()
        # Trim to matching length
        n = min(len(sig.data), len(reconstructed.data))
        np.testing.assert_allclose(
            reconstructed.data[:n], sig.data[:n], atol=1e-5
        )

    def test_silence_round_trip(self):
        sig = Signal.silence(0.1, SR)
        reconstructed = stft(sig, n_fft=N_FFT).to_signal()
        n = min(len(sig.data), len(reconstructed.data))
        np.testing.assert_allclose(reconstructed.data[:n], 0.0, atol=1e-7)

    def test_noise_round_trip(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000).astype(np.float32)
        sig = Signal(data, SR)
        reconstructed = stft(sig, n_fft=N_FFT).to_signal()
        n = min(len(sig.data), len(reconstructed.data))
        np.testing.assert_allclose(
            reconstructed.data[:n], sig.data[:n], atol=1e-5
        )

    def test_preserves_sample_rate(self):
        sig = _tone(440, duration=0.1)
        spec = stft(sig, n_fft=N_FFT)
        assert spec.sample_rate == SR
        assert spec.to_signal().sample_rate == SR

    def test_preserves_duration(self):
        sig = _tone(440, duration=0.25)
        reconstructed = stft(sig, n_fft=N_FFT).to_signal()
        assert abs(reconstructed.duration - sig.duration) < 1e-6


# ------------------------------------------------------------------ #
# Spectrum properties                                                  #
# ------------------------------------------------------------------ #


class TestSpectrumProperties:
    def test_n_fft(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        assert spec.n_fft == N_FFT

    def test_n_bins(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        assert spec.n_bins == N_FFT // 2 + 1

    def test_frequencies_shape(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        assert spec.frequencies.shape == (spec.n_bins,)
        assert spec.frequencies[0] == 0.0
        assert abs(spec.frequencies[-1] - SR / 2) < 1e-3

    def test_magnitude_phase_shape(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        assert spec.magnitude.shape == spec.frames.shape
        assert spec.phase.shape == spec.frames.shape

    def test_from_polar_round_trip(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        rebuilt = Spectrum.from_polar(
            spec.magnitude, spec.phase,
            spec.window, spec.hop_size, spec.sample_rate, spec.original_length,
        )
        np.testing.assert_allclose(rebuilt.frames, spec.frames, atol=1e-10)


# ------------------------------------------------------------------ #
# Spectrum algebra                                                     #
# ------------------------------------------------------------------ #


class TestSpectrumAlgebra:
    def test_add_same_length(self):
        a = stft(_tone(440, 0.1), n_fft=N_FFT)
        b = stft(_tone(880, 0.1), n_fft=N_FFT)
        c = a + b
        np.testing.assert_allclose(c.frames, a.frames + b.frames, atol=1e-10)

    def test_add_different_length(self):
        a = stft(_tone(440, 0.2), n_fft=N_FFT)
        b = stft(_tone(880, 0.1), n_fft=N_FFT)
        c = a + b
        assert c.n_frames == a.n_frames

    def test_mul_scalar(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        scaled = spec * 0.5
        np.testing.assert_allclose(scaled.frames, spec.frames * 0.5, atol=1e-10)

    def test_rmul_scalar(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        scaled = 2.0 * spec
        np.testing.assert_allclose(scaled.frames, spec.frames * 2.0, atol=1e-10)

    def test_mul_spectrum(self):
        a = stft(_tone(440, 0.1), n_fft=N_FFT)
        b = stft(_tone(880, 0.1), n_fft=N_FFT)
        c = a * b
        np.testing.assert_allclose(c.frames, a.frames * b.frames, atol=1e-10)

    def test_neg(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        neg = -spec
        np.testing.assert_allclose(neg.frames, -spec.frames, atol=1e-10)

    def test_sub(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        result = spec - spec
        np.testing.assert_allclose(result.frames, 0.0, atol=1e-10)

    def test_incompatible_sample_rate(self):
        a = stft(_tone(440, 0.1, sr=SR), n_fft=N_FFT)
        b = stft(Signal(np.zeros(200, dtype=np.float32), 8000), n_fft=N_FFT)
        with pytest.raises(ValueError, match="sample rates"):
            a + b

    def test_incompatible_n_fft(self):
        sig = _tone(440, 0.1)
        a = stft(sig, n_fft=N_FFT)
        b = stft(sig, n_fft=N_FFT * 2)
        with pytest.raises(ValueError, match="FFT sizes"):
            a + b


# ------------------------------------------------------------------ #
# Transforms                                                           #
# ------------------------------------------------------------------ #


class TestFreeze:
    def test_all_frames_identical(self):
        spec = stft(_tone(440, 0.2), n_fft=N_FFT)
        frozen = freeze(spec, frame=0)
        for i in range(1, frozen.n_frames):
            np.testing.assert_allclose(
                frozen.frames[i], frozen.frames[0], atol=1e-10
            )

    def test_average_freeze(self):
        spec = stft(_tone(440, 0.2), n_fft=N_FFT)
        frozen = freeze(spec)
        assert frozen.n_frames == spec.n_frames

    def test_round_trips(self):
        spec = stft(_tone(440, 0.2), n_fft=N_FFT)
        sig = freeze(spec, frame=0).to_signal()
        assert abs(sig.duration - 0.2) < 0.01


class TestSmear:
    def test_zero_amount_unchanged(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        smeared = smear(spec, amount=0.0)
        np.testing.assert_allclose(smeared.frames, spec.frames, atol=1e-10)

    def test_smear_broadens_peaks(self):
        spec = stft(_tone(440, 0.2), n_fft=N_FFT)
        smeared = smear(spec, amount=5.0)
        # The peak magnitude should decrease (energy spreads)
        assert smeared.magnitude.max() < spec.magnitude.max()

    def test_preserves_total_energy_roughly(self):
        spec = stft(_tone(440, 0.2), n_fft=N_FFT)
        smeared = smear(spec, amount=3.0)
        # Total magnitude should be similar (Gaussian blur conserves integral)
        orig_sum = spec.magnitude.sum()
        smeared_sum = smeared.magnitude.sum()
        assert abs(smeared_sum - orig_sum) / orig_sum < 0.1


class TestShiftBins:
    def test_shift_up(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        peak_bin = np.argmax(spec.magnitude.mean(axis=0))
        shifted = shift_bins(spec, shift=5)
        shifted_peak = np.argmax(shifted.magnitude.mean(axis=0))
        assert shifted_peak == peak_bin + 5

    def test_shift_down(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        peak_bin = np.argmax(spec.magnitude.mean(axis=0))
        shifted = shift_bins(spec, shift=-3)
        shifted_peak = np.argmax(shifted.magnitude.mean(axis=0))
        assert shifted_peak == peak_bin - 3

    def test_zero_shift_unchanged(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        shifted = shift_bins(spec, shift=0)
        np.testing.assert_allclose(
            shifted.magnitude, spec.magnitude, atol=1e-10
        )


class TestCrossSynthesize:
    def test_full_mix_uses_modulator_magnitude(self):
        carrier = stft(_tone(440, 0.1), n_fft=N_FFT)
        modulator = stft(_tone(880, 0.1), n_fft=N_FFT)
        result = cross_synthesize(carrier, modulator, mix=1.0)
        n = min(result.n_frames, modulator.n_frames)
        np.testing.assert_allclose(
            result.magnitude[:n], modulator.magnitude[:n], atol=1e-10
        )

    def test_zero_mix_preserves_carrier(self):
        carrier = stft(_tone(440, 0.1), n_fft=N_FFT)
        modulator = stft(_tone(880, 0.1), n_fft=N_FFT)
        result = cross_synthesize(carrier, modulator, mix=0.0)
        n = min(result.n_frames, carrier.n_frames)
        np.testing.assert_allclose(
            result.magnitude[:n], carrier.magnitude[:n], atol=1e-10
        )

    def test_preserves_carrier_phase(self):
        carrier = stft(_tone(440, 0.1), n_fft=N_FFT)
        modulator = stft(_tone(880, 0.1), n_fft=N_FFT)
        result = cross_synthesize(carrier, modulator, mix=1.0)
        n = min(result.n_frames, carrier.n_frames)
        np.testing.assert_allclose(
            result.phase[:n], carrier.phase[:n], atol=1e-10
        )


class TestPitchShift:
    def test_octave_up(self):
        spec = stft(_tone(440, 0.2), n_fft=N_FFT)
        peak_bin = np.argmax(spec.magnitude.mean(axis=0))
        shifted = pitch_shift(spec, semitones=12)
        shifted_peak = np.argmax(shifted.magnitude.mean(axis=0))
        # Octave up should roughly double the peak bin
        assert abs(shifted_peak - peak_bin * 2) <= 2

    def test_zero_shift_preserves(self):
        spec = stft(_tone(440, 0.1), n_fft=N_FFT)
        shifted = pitch_shift(spec, semitones=0)
        np.testing.assert_allclose(
            shifted.magnitude, spec.magnitude, atol=1e-10
        )


# ------------------------------------------------------------------ #
# ConvolutionReverb                                                    #
# ------------------------------------------------------------------ #


class TestConvolutionReverb:
    def test_delta_ir_preserves_signal(self):
        sig = _tone(440, 0.1)
        ir = _sig([1.0])
        reverb = ConvolutionReverb(ir, wet=1.0)
        result = reverb(sig)
        np.testing.assert_allclose(result.data, sig.data, atol=1e-4)

    def test_is_effect(self):
        ir = _sig([1.0])
        assert isinstance(ConvolutionReverb(ir), Effect)

    def test_wet_dry_mix(self):
        sig = _tone(440, 0.1)
        ir = _sig([1.0])
        dry_result = ConvolutionReverb(ir, wet=0.0)(sig)
        np.testing.assert_allclose(dry_result.data, sig.data, atol=1e-6)

    def test_sample_rate_mismatch(self):
        sig = _tone(440, 0.1, sr=SR)
        ir = Signal(np.array([1.0], dtype=np.float32), 8000)
        with pytest.raises(ValueError, match="sample rate"):
            ConvolutionReverb(ir)(sig)


# ------------------------------------------------------------------ #
# Effect wrappers                                                      #
# ------------------------------------------------------------------ #


class TestEffectWrappers:
    def test_spectral_freeze_is_effect(self):
        assert isinstance(SpectralFreeze(), Effect)

    def test_spectral_smear_is_effect(self):
        assert isinstance(SpectralSmear(), Effect)

    def test_pitch_shift_is_effect(self):
        assert isinstance(PitchShift(), Effect)

    def test_vocoder_is_effect(self):
        mod = _tone(440, 0.1)
        assert isinstance(Vocoder(mod), Effect)

    def test_spectral_freeze_runs(self):
        sig = _tone(440, 0.2)
        result = SpectralFreeze(freeze_time=0.05, n_fft=N_FFT)(sig)
        assert isinstance(result, Signal)
        assert result.duration > 0

    def test_spectral_smear_runs(self):
        sig = _tone(440, 0.1)
        result = SpectralSmear(amount=3.0, n_fft=N_FFT)(sig)
        assert isinstance(result, Signal)

    def test_pitch_shift_runs(self):
        sig = _tone(440, 0.1)
        result = PitchShift(semitones=5, n_fft=N_FFT)(sig)
        assert isinstance(result, Signal)

    def test_vocoder_runs(self):
        carrier = _tone(440, 0.1)
        modulator = _tone(880, 0.1)
        result = Vocoder(modulator, n_fft=N_FFT)(carrier)
        assert isinstance(result, Signal)

    def test_effect_chaining(self):
        sig = _tone(440, 0.1)
        chain = SpectralSmear(amount=2.0, n_fft=N_FFT) | PitchShift(semitones=3, n_fft=N_FFT)
        result = chain(sig)
        assert isinstance(result, Signal)
