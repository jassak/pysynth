from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal


class ADSR:
    """Attack-Decay-Sustain-Release amplitude envelope.

    All time parameters are in seconds. ``sustain`` is the *duration* of the
    sustain phase (not a level); ``sustain_level`` is the amplitude (0..1) held
    during that phase.

    Usage::

        env = ADSR(attack=0.01, decay=0.1, sustain=0.5, sustain_level=0.7, release=0.3)
        sig = env.apply(Oscillator("saw", 220).render(1.0))

    Both ``render`` and ``apply`` return new Signal instances; neither mutates
    the input.
    """

    def __init__(
        self,
        attack: float,
        decay: float,
        sustain: float,
        sustain_level: float,
        release: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.sustain_level = sustain_level
        self.release = release
        self.sample_rate = sample_rate

    def render(self, duration: float | None = None) -> Signal:
        """Return the envelope as a Signal with values in [0, 1].

        If ``duration`` is given the envelope is truncated or zero-padded to
        match the requested length. Otherwise the natural length
        (attack + decay + sustain + release) is used.
        """
        a = int(self.attack * self.sample_rate)
        d = int(self.decay * self.sample_rate)
        s = int(self.sustain * self.sample_rate)
        r = int(self.release * self.sample_rate)

        env = np.concatenate(
            [
                np.linspace(0.0, 1.0, a, dtype=np.float32),
                np.linspace(1.0, self.sustain_level, d, dtype=np.float32),
                np.full(s, self.sustain_level, dtype=np.float32),
                np.linspace(self.sustain_level, 0.0, r, dtype=np.float32),
            ]
        )

        if duration is not None:
            n = int(duration * self.sample_rate)
            if len(env) >= n:
                env = env[:n]
            else:
                env = np.pad(env, (0, n - len(env)))

        return Signal(env, self.sample_rate)

    def apply(self, signal: Signal) -> Signal:
        """Multiply a signal by this envelope. Returns a new Signal."""
        env = self.render(signal.duration)
        n = min(len(signal.data), len(env.data))
        data = signal.data[:n] * env.data[:n]
        return Signal(data, signal.sample_rate)
