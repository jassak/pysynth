from __future__ import annotations

import numpy as np

from pysynth._core import Effect, Signal


class Delay(Effect):
    """Single-tap delay with feedback.

    Parameters
    ----------
    delay_time:
        Delay time in seconds.
    feedback:
        Fraction of the delayed signal fed back into the delay line (0..1).
    wet:
        Mix of delayed signal (0 = dry only, 1 = wet only).
    """

    def __init__(
        self,
        delay_time: float,
        feedback: float = 0.4,
        wet: float = 0.5,
    ) -> None:
        self.delay_time = delay_time
        self.feedback = np.clip(feedback, 0.0, 0.99)
        self.wet = np.clip(wet, 0.0, 1.0)

    def __call__(self, sig: Signal) -> Signal:
        delay_samples = int(self.delay_time * sig.sample_rate)
        x = sig.data
        n = len(x)
        buf = np.zeros(n + delay_samples, dtype=np.float32)
        buf[:n] = x
        out = buf.copy()

        for i in range(delay_samples, n + delay_samples):
            out[i] += self.feedback * out[i - delay_samples]

        # Trim back to original length
        out = out[:n]
        mixed = (1.0 - self.wet) * x + self.wet * out
        return Signal(mixed.astype(np.float32), sig.sample_rate)


class Echo(Effect):
    """Multi-tap echo: a fixed number of evenly-spaced repeats.

    Unlike Delay, Echo does not feed back; each tap decays by ``decay``
    relative to the previous.
    """

    def __init__(
        self,
        delay_time: float,
        repeats: int = 4,
        decay: float = 0.5,
        wet: float = 0.6,
    ) -> None:
        self.delay_time = delay_time
        self.repeats = repeats
        self.decay = decay
        self.wet = np.clip(wet, 0.0, 1.0)

    def __call__(self, sig: Signal) -> Signal:
        delay_samples = int(self.delay_time * sig.sample_rate)
        x = sig.data
        n = len(x)
        extra = delay_samples * self.repeats
        out = np.zeros(n + extra, dtype=np.float32)
        out[:n] = x

        amp = self.decay
        for tap in range(1, self.repeats + 1):
            offset = delay_samples * tap
            out[offset : offset + n] += x * amp
            amp *= self.decay

        out = out[:n]
        mixed = (1.0 - self.wet) * x + self.wet * out
        return Signal(mixed.astype(np.float32), sig.sample_rate)
