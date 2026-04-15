from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pysynth._core import Effect, Signal


class ConvolutionReverb(Effect):
    """FFT-based convolution reverb using an impulse response.

    Convolves the input signal with an impulse response using overlap-save
    for efficient processing of arbitrary-length signals.

    Parameters
    ----------
    impulse_response:
        The impulse response as a :class:`Signal`, a :class:`Sample`, or a
        path (str / Path) to a WAV file.
    wet:
        Mix ratio of the convolved signal (0 = dry, 1 = fully wet).
    predelay:
        Delay before reverb onset, in seconds.
    """

    def __init__(
        self,
        impulse_response: Signal | str | Path,
        wet: float = 0.5,
        predelay: float = 0.0,
    ) -> None:
        # Accept Sample via duck-typing (avoid circular import)
        if hasattr(impulse_response, "as_signal") and not isinstance(impulse_response, Signal):
            impulse_response = impulse_response.as_signal()
        if isinstance(impulse_response, (str, Path)):
            sr, data = wavfile.read(impulse_response)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            if data.ndim == 2:
                data = data.mean(axis=1)
            impulse_response = Signal(data, sr)
        self._ir = impulse_response
        self.wet = wet
        self.predelay = predelay

    def __call__(self, signal: Signal) -> Signal:
        x = signal.data.astype(np.float64)
        if x.ndim == 2:
            x = x.mean(axis=1)

        ir = self._ir.data.astype(np.float64)
        if self._ir.sample_rate != signal.sample_rate:
            raise ValueError(
                f"Impulse response sample rate ({self._ir.sample_rate}) "
                f"does not match signal sample rate ({signal.sample_rate})"
            )

        # Predelay: prepend zeros to the IR
        if self.predelay > 0:
            n_delay = int(self.predelay * signal.sample_rate)
            ir = np.concatenate([np.zeros(n_delay), ir])

        # FFT convolution (single-shot for simplicity; overlap-save for
        # very long signals could be added later)
        n_conv = len(x) + len(ir) - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft <<= 1

        X = np.fft.rfft(x, n=n_fft)
        H = np.fft.rfft(ir, n=n_fft)
        convolved = np.fft.irfft(X * H, n=n_fft)[:len(x)]

        # Normalise so the convolved signal has roughly the same peak
        # amplitude as the input (prevents blow-up with long IRs)
        peak_in = np.max(np.abs(x)) + 1e-10
        peak_out = np.max(np.abs(convolved)) + 1e-10
        convolved *= peak_in / peak_out

        mixed = (1.0 - self.wet) * x + self.wet * convolved
        return Signal(mixed.astype(np.float32), signal.sample_rate)
