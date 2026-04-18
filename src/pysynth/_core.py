from __future__ import annotations

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from dataclasses import dataclass

SAMPLE_RATE = 44100


@dataclass
class Signal:
    """A finite audio signal — an element of a commutative ℝ-algebra.

    Supports the full algebra interface:
        signal + signal   -> mix (shorter is zero-padded)
        signal + scalar   -> add constant to every sample (DC offset / FM carrier shift)
        scalar + signal   -> same (commutative)
        signal * signal   -> pointwise (ring) modulation (truncated to shorter length)
        signal * scalar   -> scale amplitude
        scalar * signal   -> scale amplitude
        -signal           -> phase inversion
    """

    data: np.ndarray  # float32, shape (n_samples,) mono or (n_samples, 2) stereo
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    @property
    def n_channels(self) -> int:
        return 1 if self.data.ndim == 1 else self.data.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------ #
    # Vector space operations — all return new Signal instances          #
    # ------------------------------------------------------------------ #

    def __add__(self, other: Signal | float | int) -> Signal:
        if isinstance(other, Signal):
            if self.sample_rate != other.sample_rate:
                raise ValueError(
                    f"Cannot add signals with different sample rates: "
                    f"{self.sample_rate} vs {other.sample_rate}"
                )
            a, b = self.data, other.data
            if a.ndim == b.ndim:
                # Same type (mono+mono or stereo+stereo) — fast path
                if len(a) >= len(b):
                    out = a.copy()
                    out[: len(b)] += b
                else:
                    out = b.copy()
                    out[: len(a)] += a
            else:
                # Mixed mono/stereo — promote mono to (n, 1) view, use buffer
                if a.ndim == 1:
                    a = a[:, np.newaxis]
                else:
                    b = b[:, np.newaxis]
                ch = max(a.shape[1], b.shape[1])
                n = max(len(a), len(b))
                out = np.zeros((n, ch), dtype=np.float32)
                out[: len(a)] += a
                out[: len(b)] += b
            return Signal(out, self.sample_rate)
        # Scalar: treat as a constant signal (DC offset / frequency shift for FM)
        return Signal(self.data + np.float32(other), self.sample_rate)

    def __radd__(self, other: Signal | float | int) -> Signal:
        """Addition is commutative: a + b == b + a."""
        return self.__add__(other)

    def __mul__(self, other: Signal | float) -> Signal:
        if isinstance(other, Signal):
            if self.sample_rate != other.sample_rate:
                raise ValueError(
                    f"Cannot multiply signals with different sample rates: "
                    f"{self.sample_rate} vs {other.sample_rate}"
                )
            n = min(len(self.data), len(other.data))
            a, b = self.data[:n], other.data[:n]
            if a.ndim != b.ndim:
                # Mixed mono/stereo — promote mono to (n, 1) view
                if a.ndim == 1:
                    a = a[:, np.newaxis]
                else:
                    b = b[:, np.newaxis]
            return Signal(a * b, self.sample_rate)
        return Signal(self.data * np.float32(other), self.sample_rate)

    def __rmul__(self, other: Signal | float) -> Signal:
        return self.__mul__(other)

    def __neg__(self) -> Signal:
        return Signal(-self.data, self.sample_rate)

    def __sub__(self, other: Signal | float | int) -> Signal:
        return self.__add__(-other if isinstance(other, Signal) else -float(other))

    # ------------------------------------------------------------------ #
    # REPL convenience                                                   #
    # ------------------------------------------------------------------ #

    def play(self, blocking: bool = True) -> None:
        """Play the signal through the default audio device."""
        data = np.clip(self.data, -1.0, 1.0).astype(np.float32)
        sd.play(data, samplerate=self.sample_rate)
        if blocking:
            sd.wait()

    def plot(self, max_duration: float | None = None, ax=None) -> None:
        """Plot the waveform using matplotlib.

        For stereo signals, two axes are drawn: left channel on top, right
        channel on bottom.  When *ax* is ``None`` a new figure is created and
        ``plt.show()`` is called.  For stereo, pass a sequence of two
        ``Axes`` objects; for mono, pass a single ``Axes``.
        """
        data = self.data
        if max_duration is not None:
            n = int(max_duration * self.sample_rate)
            data = data[:n]
        t = np.arange(len(data)) / self.sample_rate
        stereo = data.ndim == 2 and data.shape[1] == 2

        if ax is None:
            if stereo:
                fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            else:
                fig, axes = plt.subplots(figsize=(10, 3))
                axes = [axes]
            show = True
        else:
            axes = list(ax) if stereo else [ax]
            show = False

        channels = [data[:, 0], data[:, 1]] if stereo else [data]
        labels = ["Left", "Right"] if stereo else ["Amplitude"]
        for channel_data, axis, ylabel in zip(channels, axes, labels):
            axis.plot(t, channel_data, linewidth=0.5)
            axis.set_ylabel(ylabel)
        axes[-1].set_xlabel("Time (s)")
        axes[0].set_title(repr(self))

        if show:
            plt.tight_layout()
            plt.show()

    def __repr__(self) -> str:
        channels = "mono" if self.n_channels == 1 else f"{self.n_channels}ch"
        return f"Signal({self.duration:.3f}s, {channels}, {self.sample_rate}Hz)"

    # ------------------------------------------------------------------ #
    # Signal processing                                                  #
    # ------------------------------------------------------------------ #

    def shift(self, seconds: float) -> Signal:
        """Return a copy prepended with *seconds* of silence.

        Composable with ``+`` for layering events at different times::

            result = kick.shift(0.0) + snare.shift(0.5) + hat.shift(0.25)
        """
        if seconds <= 0:
            return Signal(self.data.copy(), self.sample_rate)
        n = int(seconds * self.sample_rate)
        if self.data.ndim == 1:
            pad = np.zeros(n, dtype=np.float32)
        else:
            pad = np.zeros((n, self.data.shape[1]), dtype=np.float32)
        return Signal(np.concatenate([pad, self.data]), self.sample_rate)

    def __getitem__(self, key: slice) -> Signal:
        """Slice by time in seconds: ``signal[0.5:1.2]``."""
        if not isinstance(key, slice):
            raise TypeError("Signal only supports slice indexing (e.g. signal[0.5:1.2])")
        if key.step is not None:
            raise ValueError("Step is not supported in Signal slicing")

        start_idx = 0
        if key.start is not None:
            start_idx = max(0, int(key.start * self.sample_rate))

        end_idx = len(self)
        if key.stop is not None:
            end_idx = min(len(self), int(key.stop * self.sample_rate))

        return Signal(self.data[start_idx:end_idx].copy(), self.sample_rate)

    def normalize(self, mode: str = "peak") -> Signal:
        """Return a peak-normalized copy.

        Parameters
        ----------
        mode:
            Normalization mode.  Currently only ``"peak"`` is supported.
        """
        if mode != "peak":
            raise ValueError(f"Unsupported normalization mode: {mode!r}")
        peak = np.max(np.abs(self.data))
        if peak == 0:
            return Signal(self.data.copy(), self.sample_rate)
        return Signal(self.data / peak, self.sample_rate)

    # ------------------------------------------------------------------ #
    # Class methods                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def silence(cls, duration: float, sample_rate: int = SAMPLE_RATE) -> Signal:
        n = int(duration * sample_rate)
        return cls(np.zeros(n, dtype=np.float32), sample_rate)

    @classmethod
    def concat(cls, *signals: Signal) -> Signal:
        """Concatenate signals sequentially in time.

        ::

            drums = Signal.concat(intro, verse, chorus)
        """
        if len(signals) < 2:
            raise ValueError("concat requires at least two signals")
        sr = signals[0].sample_rate
        for s in signals[1:]:
            if s.sample_rate != sr:
                raise ValueError(
                    f"Cannot concatenate signals with different sample rates: "
                    f"{sr} vs {s.sample_rate}"
                )
        stereo = any(s.data.ndim == 2 for s in signals)
        ch = max(s.data.shape[1] for s in signals if s.data.ndim == 2) if stereo else 0
        arrays = []
        for s in signals:
            d = s.data
            if stereo and d.ndim == 1:
                d = np.column_stack([d] * ch)
            arrays.append(d)
        return cls(np.concatenate(arrays), sr)


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


def _as_array(param: float | Signal, n: int) -> np.ndarray:
    """Convert a float or Signal control parameter to a float64 array of length n.

    Signal inputs are edge-padded (last value held) when shorter than n, or
    truncated when longer. This is the correct semantic for control signals —
    a filter cutoff or gain that runs short should hold its last value, not
    snap to zero.

    Raises ValueError if param is a zero-length Signal.
    """
    if isinstance(param, Signal):
        if len(param.data) == 0:
            raise ValueError("Signal used as control parameter has zero length")
        arr = param.data.astype(np.float64)
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)), mode="edge")
    return np.full(n, float(param), dtype=np.float64)
