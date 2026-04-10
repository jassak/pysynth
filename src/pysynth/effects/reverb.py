from __future__ import annotations

import numpy as np

from pysynth._core import Effect, Signal


class SimpleReverb(Effect):
    """Schroeder-style reverb using a network of comb and allpass filters.

    Parameters
    ----------
    room_size:
        Controls comb filter delay lengths (0..1). Larger = longer decay.
    damping:
        High-frequency damping in the comb filters (0..1).
    wet:
        Mix ratio of processed signal (0 = dry, 1 = fully wet).
    """

    # Comb filter delay times in samples at 44100 Hz (prime-ish values)
    _COMB_DELAYS = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
    _ALLPASS_DELAYS = [225, 556, 441, 341]

    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet: float = 0.3,
    ) -> None:
        self.room_size = np.clip(room_size, 0.0, 1.0)
        self.damping = np.clip(damping, 0.0, 1.0)
        self.wet = np.clip(wet, 0.0, 1.0)

    def __call__(self, sig: Signal) -> Signal:
        x = sig.data.astype(np.float64)
        scale = sig.sample_rate / 44100.0

        feedback = 0.84 + self.room_size * 0.15
        damp1 = self.damping * 0.4
        damp2 = 1.0 - damp1

        # Parallel comb filters
        comb_out = np.zeros_like(x)
        for delay in self._COMB_DELAYS:
            d = max(1, int(delay * scale))
            buf = np.zeros(d)
            filt = 0.0
            out = np.empty_like(x)
            for i, sample in enumerate(x):
                buf_out = buf[i % d]
                filt = buf_out * damp2 + filt * damp1
                buf[i % d] = sample + filt * feedback
                out[i] = buf_out
            comb_out += out

        comb_out /= len(self._COMB_DELAYS)

        # Series allpass filters
        ap_out = comb_out
        for delay in self._ALLPASS_DELAYS:
            d = max(1, int(delay * scale))
            buf = np.zeros(d)
            out = np.empty_like(ap_out)
            for i, sample in enumerate(ap_out):
                buf_out = buf[i % d]
                buf[i % d] = sample + buf_out * 0.5
                out[i] = buf_out - sample * 0.5
            ap_out = out

        mixed = (1.0 - self.wet) * x + self.wet * ap_out
        return Signal(mixed.astype(np.float32), sig.sample_rate)
