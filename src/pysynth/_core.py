from __future__ import annotations

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from dataclasses import dataclass

SAMPLE_RATE = 44100


@dataclass
class Signal:
    """A finite audio signal — a vector in the space of sampled audio.

    Supports the full vector space interface:
        signal + signal   -> mix (shorter is zero-padded)
        signal + scalar   -> add constant to every sample (DC offset / FM carrier shift)
        scalar + signal   -> same (commutative)
        signal * scalar   -> scale amplitude
        scalar * signal   -> scale amplitude
        -signal           -> phase inversion
    """

    data: np.ndarray  # float32, shape (n_samples,) mono or (n_samples, 2) stereo
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    @property
    def n_channels(self) -> int:
        return 1 if self.data.ndim == 1 else self.data.shape[1]

    # ------------------------------------------------------------------ #
    # Vector space operations — all return new Signal instances            #
    # ------------------------------------------------------------------ #

    def __add__(self, other: Signal | float | int) -> Signal:
        if isinstance(other, Signal):
            if self.sample_rate != other.sample_rate:
                raise ValueError(
                    f"Cannot add signals with different sample rates: "
                    f"{self.sample_rate} vs {other.sample_rate}"
                )
            if len(self.data) >= len(other.data):
                out = self.data.copy()
                out[: len(other.data)] += other.data
            else:
                out = other.data.copy()
                out[: len(self.data)] += self.data
            return Signal(out, self.sample_rate)
        # Scalar: treat as a constant signal (DC offset / frequency shift for FM)
        return Signal(self.data + np.float32(other), self.sample_rate)

    def __radd__(self, other: Signal | float | int) -> Signal:
        """Addition is commutative: a + b == b + a."""
        return self.__add__(other)

    def __mul__(self, scalar: float) -> Signal:
        return Signal(self.data * scalar, self.sample_rate)

    def __rmul__(self, scalar: float) -> Signal:
        return self.__mul__(scalar)

    def __neg__(self) -> Signal:
        return Signal(-self.data, self.sample_rate)

    def __sub__(self, other: Signal | float | int) -> Signal:
        return self.__add__(-other if isinstance(other, Signal) else -float(other))

    # ------------------------------------------------------------------ #
    # REPL convenience                                                     #
    # ------------------------------------------------------------------ #

    def play(self, blocking: bool = True) -> None:
        """Play the signal through the default audio device."""
        data = np.clip(self.data, -1.0, 1.0).astype(np.float32)
        sd.play(data, samplerate=self.sample_rate)
        if blocking:
            sd.wait()

    def plot(self, max_duration: float | None = None) -> None:
        """Plot the waveform using matplotlib."""
        data = self.data
        if max_duration is not None:
            n = int(max_duration * self.sample_rate)
            data = data[:n]
        t = np.arange(len(data)) / self.sample_rate
        plt.figure(figsize=(10, 3))
        plt.plot(t, data, linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(repr(self))
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        channels = "mono" if self.n_channels == 1 else f"{self.n_channels}ch"
        return f"Signal({self.duration:.3f}s, {channels}, {self.sample_rate}Hz)"

    # ------------------------------------------------------------------ #
    # Class methods                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def silence(cls, duration: float, sample_rate: int = SAMPLE_RATE) -> Signal:
        n = int(duration * sample_rate)
        return cls(np.zeros(n, dtype=np.float32), sample_rate)


# ------------------------------------------------------------------ #
# Effect base class                                                    #
# ------------------------------------------------------------------ #


class Effect:
    """Base class for all audio effects.

    Subclasses implement __call__(signal: Signal) -> Signal.
    The | operator chains effects left-to-right:

        (LowPassFilter(800) | Reverb(0.5))(signal)
    """

    def __call__(self, signal: Signal) -> Signal:
        raise NotImplementedError

    def __or__(self, other: Effect) -> Effect:
        return _Chain(self, other)


class _Chain(Effect):
    """Internal effect chain created by the | operator. Not part of the public API."""

    def __init__(self, a: Effect, b: Effect) -> None:
        self._a = a
        self._b = b

    def __call__(self, signal: Signal) -> Signal:
        return self._b(self._a(signal))

    def __repr__(self) -> str:
        return f"_Chain({self._a!r}, {self._b!r})"
