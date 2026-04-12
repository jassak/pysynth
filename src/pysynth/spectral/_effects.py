from __future__ import annotations

from pysynth._core import Effect, Signal
from pysynth.spectral._spectrum import stft
from pysynth.spectral._transforms import (
    freeze as _freeze,
    smear as _smear,
    pitch_shift as _pitch_shift,
    cross_synthesize as _cross_synthesize,
)


class SpectralFreeze(Effect):
    """Freeze the spectrum of a signal at a given time point.

    Parameters
    ----------
    freeze_time:
        Time in seconds at which to capture the frozen frame.
    n_fft:
        FFT size for the internal STFT.
    """

    def __init__(self, freeze_time: float = 0.0, n_fft: int = 2048) -> None:
        self.freeze_time = freeze_time
        self.n_fft = n_fft

    def __call__(self, signal: Signal) -> Signal:
        spec = stft(signal, n_fft=self.n_fft)
        frame_idx = int(self.freeze_time * signal.sample_rate / spec.hop_size)
        frame_idx = max(0, min(frame_idx, spec.n_frames - 1))
        return _freeze(spec, frame=frame_idx).to_signal()


class SpectralSmear(Effect):
    """Blur the spectrum across frequency bins.

    Parameters
    ----------
    amount:
        Standard deviation of the Gaussian kernel in bins.
    n_fft:
        FFT size for the internal STFT.
    """

    def __init__(self, amount: float = 5.0, n_fft: int = 2048) -> None:
        self.amount = amount
        self.n_fft = n_fft

    def __call__(self, signal: Signal) -> Signal:
        spec = stft(signal, n_fft=self.n_fft)
        return _smear(spec, amount=self.amount).to_signal()


class PitchShift(Effect):
    """Shift pitch without changing duration.

    Parameters
    ----------
    semitones:
        Number of semitones to shift. Positive = up, negative = down.
    n_fft:
        FFT size for the internal STFT.
    """

    def __init__(self, semitones: float = 0.0, n_fft: int = 2048) -> None:
        self.semitones = semitones
        self.n_fft = n_fft

    def __call__(self, signal: Signal) -> Signal:
        spec = stft(signal, n_fft=self.n_fft)
        return _pitch_shift(spec, semitones=self.semitones).to_signal()


class Vocoder(Effect):
    """Cross-synthesise two signals: spectral envelope from modulator,
    harmonic content from carrier.

    The modulator (typically a voice signal) is supplied at construction.
    The carrier (typically a synth pad or noise) is the signal passed to
    ``__call__``.

    Parameters
    ----------
    modulator:
        Signal whose spectral envelope is extracted.
    n_fft:
        FFT size for the internal STFT.
    mix:
        Cross-synthesis depth (0 = carrier only, 1 = full vocoder).
    """

    def __init__(
        self,
        modulator: Signal,
        n_fft: int = 2048,
        mix: float = 1.0,
    ) -> None:
        self._modulator = modulator
        self.n_fft = n_fft
        self.mix = mix

    def __call__(self, carrier: Signal) -> Signal:
        carrier_spec = stft(carrier, n_fft=self.n_fft)
        modulator_spec = stft(self._modulator, n_fft=self.n_fft)
        return _cross_synthesize(carrier_spec, modulator_spec,
                                 mix=self.mix).to_signal()
