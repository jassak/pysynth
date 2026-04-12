from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.signal import get_window

from pysynth._core import Signal, SAMPLE_RATE


@dataclass
class Spectrum:
    """STFT representation of an audio signal — a time-frequency algebra.

    Wraps a 2-D complex array of shape ``(n_frames, n_bins)`` produced by
    :func:`stft`, together with the analysis parameters needed for perfect
    reconstruction via :meth:`to_signal`.

    Supports frame-wise algebra mirroring :class:`Signal`:

        spectrum + spectrum   -> spectral mixing (add complex frames)
        spectrum * spectrum   -> spectral convolution (pointwise complex multiply)
        spectrum * scalar     -> scale magnitude
        -spectrum             -> negate
    """

    frames: np.ndarray  # complex128, (n_frames, n_bins)
    window: np.ndarray  # float64, (n_fft,)
    hop_size: int
    sample_rate: int
    original_length: int

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=np.complex128)
        self.window = np.asarray(self.window, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def n_bins(self) -> int:
        return self.frames.shape[1]

    @property
    def n_fft(self) -> int:
        return len(self.window)

    @property
    def duration(self) -> float:
        return self.original_length / self.sample_rate

    @property
    def frequencies(self) -> np.ndarray:
        """Centre frequency of each bin in Hz."""
        return np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

    @property
    def magnitude(self) -> np.ndarray:
        """Magnitude of each bin per frame — float64, (n_frames, n_bins)."""
        return np.abs(self.frames)

    @property
    def phase(self) -> np.ndarray:
        """Phase angle of each bin per frame — float64, (n_frames, n_bins)."""
        return np.angle(self.frames)

    # ------------------------------------------------------------------ #
    # Algebra                                                              #
    # ------------------------------------------------------------------ #

    def _check_compatible(self, other: Spectrum) -> None:
        if self.sample_rate != other.sample_rate:
            raise ValueError(
                f"Cannot combine spectra with different sample rates: " f"{self.sample_rate} vs {other.sample_rate}"
            )
        if self.n_fft != other.n_fft:
            raise ValueError(f"Cannot combine spectra with different FFT sizes: " f"{self.n_fft} vs {other.n_fft}")
        if self.hop_size != other.hop_size:
            raise ValueError(
                f"Cannot combine spectra with different hop sizes: " f"{self.hop_size} vs {other.hop_size}"
            )

    def __add__(self, other: Spectrum | float | int) -> Spectrum:
        if isinstance(other, Spectrum):
            self._check_compatible(other)
            if self.n_frames >= other.n_frames:
                out = self.frames.copy()
                out[: other.n_frames] += other.frames
                length = self.original_length
            else:
                out = other.frames.copy()
                out[: self.n_frames] += self.frames
                length = other.original_length
            return Spectrum(out, self.window.copy(), self.hop_size, self.sample_rate, length)
        return Spectrum(
            self.frames + np.complex128(other),
            self.window.copy(),
            self.hop_size,
            self.sample_rate,
            self.original_length,
        )

    def __radd__(self, other: Spectrum | float | int) -> Spectrum:
        return self.__add__(other)

    def __mul__(self, other: Spectrum | float) -> Spectrum:
        if isinstance(other, Spectrum):
            self._check_compatible(other)
            n = min(self.n_frames, other.n_frames)
            length = min(self.original_length, other.original_length)
            out = self.frames[:n] * other.frames[:n]
            return Spectrum(out, self.window.copy(), self.hop_size, self.sample_rate, length)
        return Spectrum(
            self.frames * np.float64(other), self.window.copy(), self.hop_size, self.sample_rate, self.original_length
        )

    def __rmul__(self, other: Spectrum | float) -> Spectrum:
        return self.__mul__(other)

    def __neg__(self) -> Spectrum:
        return Spectrum(-self.frames, self.window.copy(), self.hop_size, self.sample_rate, self.original_length)

    def __sub__(self, other: Spectrum | float | int) -> Spectrum:
        if isinstance(other, Spectrum):
            return self.__add__(-other)
        return self.__add__(-float(other))

    # ------------------------------------------------------------------ #
    # Reconstruction                                                       #
    # ------------------------------------------------------------------ #

    def to_signal(self) -> Signal:
        """Reconstruct the time-domain Signal via overlap-add ISTFT."""
        n_fft = self.n_fft
        hop = self.hop_size
        n_frames = self.n_frames
        out_length = (n_frames - 1) * hop + n_fft

        output = np.zeros(out_length, dtype=np.float64)
        window_sum = np.zeros(out_length, dtype=np.float64)
        w = self.window

        for i in range(n_frames):
            frame = np.fft.irfft(self.frames[i], n=n_fft)
            start = i * hop
            output[start : start + n_fft] += frame * w
            window_sum[start : start + n_fft] += w * w

        # Normalise by the sum of squared windows (avoids division by zero
        # in zero-padded regions at the edges).
        nonzero = window_sum > 1e-10
        output[nonzero] /= window_sum[nonzero]

        # Strip the left padding added by stft() and trim to original length
        pad_left = self.n_fft // 2
        output = output[pad_left : pad_left + self.original_length]
        return Signal(output.astype(np.float32), self.sample_rate)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_polar(
        cls,
        magnitude: np.ndarray,
        phase: np.ndarray,
        window: np.ndarray,
        hop_size: int,
        sample_rate: int,
        original_length: int,
    ) -> Spectrum:
        """Construct a Spectrum from separate magnitude and phase arrays."""
        frames = magnitude * np.exp(1j * phase)
        return cls(frames, window, hop_size, sample_rate, original_length)

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #

    def plot(self) -> None:
        """Plot the spectrogram (magnitude in dB)."""
        mag = self.magnitude
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-10))

        times = np.arange(self.n_frames) * self.hop_size / self.sample_rate
        freqs = self.frequencies

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, freqs, mag_db.T, shading="auto", cmap="magma")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title(repr(self))
        plt.colorbar(label="dB")
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"Spectrum({self.n_frames} frames, {self.n_bins} bins, "
            f"n_fft={self.n_fft}, hop={self.hop_size}, {self.sample_rate}Hz)"
        )


# ------------------------------------------------------------------ #
# STFT                                                                 #
# ------------------------------------------------------------------ #


def stft(
    signal: Signal,
    n_fft: int = 2048,
    hop_size: int | None = None,
    window: str = "hann",
) -> Spectrum:
    """Compute the Short-Time Fourier Transform of a Signal.

    Parameters
    ----------
    signal:
        Input Signal (mono). Stereo signals are mixed to mono first.
    n_fft:
        FFT size. Power of 2 recommended.
    hop_size:
        Hop between frames. Defaults to ``n_fft // 4`` (75% overlap).
    window:
        Window function name (passed to ``scipy.signal.get_window``).

    Returns
    -------
    Spectrum
        STFT representation with COLA-compliant parameters.
    """
    if hop_size is None:
        hop_size = n_fft // 4

    x = signal.data.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)

    original_length = len(x)

    # Centre the first window at sample 0 so edge samples reconstruct
    # correctly, and pad the end so the last frame is complete.
    pad_left = n_fft // 2
    x = np.pad(x, (pad_left, 0))

    pad_right = (n_fft - len(x) % hop_size) % hop_size
    if pad_right > 0:
        x = np.pad(x, (0, pad_right))

    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))

    w = get_window(window, n_fft, fftbins=True).astype(np.float64)

    n_frames = 1 + (len(x) - n_fft) // hop_size
    n_bins = n_fft // 2 + 1
    frames = np.empty((n_frames, n_bins), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_size
        frames[i] = np.fft.rfft(x[start : start + n_fft] * w)

    return Spectrum(frames, w, hop_size, signal.sample_rate, original_length)
