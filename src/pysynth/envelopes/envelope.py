from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal


@dataclass
class Segment:
    """A single envelope segment interpolating from start to end over duration seconds.

    ``curve`` controls the shape of the interpolation:

        curve = 0.0  → linear
        curve < 0    → concave (fast initial change, slows toward end)
        curve > 0    → convex (slow initial change, accelerates toward end)

    The mapping uses ``f(t) = (1 − eˢ·ᵗ) / (1 − eˢ)`` (SuperCollider convention)
    for non-zero curve ``s``, where ``t ∈ [0, 1]``.
    """

    duration: float
    start: float
    end: float
    curve: float = 0.0

    def render(self, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Return a float32 array of length ``int(duration * sample_rate)``."""
        n = int(self.duration * sample_rate)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        if abs(self.curve) < 1e-4:
            shape = t
        else:
            shape = (1.0 - np.exp(self.curve * t)) / (1.0 - np.exp(self.curve))
        return (self.start + (self.end - self.start) * shape).astype(np.float32)


class Envelope:
    """An amplitude envelope defined as an ordered list of Segments.

    Parameters
    ----------
    segments:
        Ordered list of Segments defining the envelope shape.
    sustain_node:
        Index of the segment that sustains while a gate is held. Segments
        before it play on gate-on (attack/decay); the sustain_node's end
        level is held while the gate is high; segments after it play on
        gate-off (release). ``None`` means one-shot (no sustain hold).
    sample_rate:
        Default sample rate for rendering.

    Usage::

        env = Envelope([
            Segment(0.01, 0.0, 1.0),           # attack
            Segment(0.1,  1.0, 0.7),            # decay
            Segment(0.5,  0.7, 0.7),            # sustain
            Segment(0.3,  0.7, 0.0, curve=2.0), # release with convex curve
        ], sustain_node=2)

        # One-shot render (no gate)
        sig = env.render(1.0)

        # Gate-triggered render
        pitch, gate = Sequencer(notes, bpm=120).cv()
        amp = env.trigger(gate)
    """

    def __init__(
        self,
        segments: list[Segment],
        sustain_node: int | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.segments = segments
        self.sustain_node = sustain_node
        self.sample_rate = sample_rate

    def render(self, duration: float | None = None, sample_rate: int | None = None) -> Signal:
        """Return the envelope as a Signal.

        If ``duration`` is given the envelope is truncated or zero-padded to
        match the requested length. Otherwise the natural length
        (sum of all segment durations) is used.

        If ``sample_rate`` is given it overrides the envelope's default.
        """
        sr = sample_rate if sample_rate is not None else self.sample_rate
        parts = [seg.render(sr) for seg in self.segments]
        env = np.concatenate(parts).astype(np.float32) if parts else np.empty(0, dtype=np.float32)

        if duration is not None:
            n = int(duration * sr)
            if len(env) >= n:
                env = env[:n]
            else:
                env = np.pad(env, (0, n - len(env))).astype(np.float32)

        return Signal(env, sr)

    def apply(self, signal: Signal) -> Signal:
        """Multiply a signal by this envelope. Returns a new Signal."""
        env = self.render(signal.duration, signal.sample_rate)
        n = min(len(signal.data), len(env.data))
        return Signal(signal.data[:n] * env.data[:n], signal.sample_rate)

    def trigger(self, gate: Signal) -> Signal:
        """Render this envelope triggered by a gate signal.

        On each rising edge of the gate, the envelope restarts from the
        first segment. Segments up to ``sustain_node`` play in sequence;
        the ``sustain_node`` segment's end level is held while the gate
        remains high. When the gate falls, segments after ``sustain_node``
        play (the release phase).

        If ``sustain_node`` is ``None``, the entire envelope plays on each
        rising edge with no sustain hold (one-shot / percussive mode).
        """
        sr = gate.sample_rate
        gate_data = gate.data
        n_samples = len(gate_data)
        out = np.zeros(n_samples, dtype=np.float32)

        # Pre-render each segment as an array
        seg_arrays = [seg.render(sr) for seg in self.segments]
        n_segs = len(seg_arrays)
        sn = self.sustain_node  # None or index

        # Determine sustain level (end value of sustain_node segment)
        if sn is not None and sn < n_segs:
            sustain_level = self.segments[sn].end
        else:
            sustain_level = 0.0

        # State machine
        seg_idx = n_segs  # current segment index (>= n_segs means idle)
        seg_pos = 0       # position within current segment
        level = 0.0       # current output level
        prev_gate = 0.0   # previous gate value for edge detection

        for i in range(n_samples):
            g = gate_data[i]

            # Rising edge: restart from segment 0
            if g > 0.0 and prev_gate <= 0.0:
                seg_idx = 0
                seg_pos = 0

            # Falling edge: jump to release segments
            if g <= 0.0 and prev_gate > 0.0:
                if sn is not None and sn + 1 < n_segs:
                    seg_idx = sn + 1
                    seg_pos = 0
                    # Rescale release: current level may differ from segment start
                    # We'll handle this by outputting the pre-rendered segment
                    # scaled from current level
                else:
                    seg_idx = n_segs  # no release segments, go idle

            if seg_idx < n_segs:
                arr = seg_arrays[seg_idx]

                # At or past sustain node while gate is high: hold
                if sn is not None and seg_idx == sn and g > 0.0 and seg_pos >= len(arr):
                    level = sustain_level
                elif seg_pos < len(arr):
                    level = float(arr[seg_pos])
                    seg_pos += 1
                else:
                    # Segment exhausted, advance to next
                    seg_idx += 1
                    seg_pos = 0
                    # Skip past sustain node if gate is still high
                    if sn is not None and seg_idx == sn and g > 0.0:
                        level = sustain_level
                    elif seg_idx < n_segs:
                        arr = seg_arrays[seg_idx]
                        if seg_pos < len(arr):
                            level = float(arr[seg_pos])
                            seg_pos += 1
                    else:
                        level = 0.0
            else:
                # Idle — decay toward zero (already there for most cases)
                level = 0.0

            out[i] = level
            prev_gate = g

        return Signal(out, sr)


def adsr(
    attack: float,
    decay: float,
    sustain: float,
    sustain_level: float,
    release: float,
    sample_rate: int = SAMPLE_RATE,
) -> Envelope:
    """Construct a standard ADSR Envelope.

    Parameters
    ----------
    attack:       rise time from 0 to 1 (seconds)
    decay:        fall time from 1 to sustain_level (seconds)
    sustain:      duration of the sustain phase (seconds)
    sustain_level: amplitude held during sustain (0..1)
    release:      fall time from sustain_level to 0 (seconds)
    """
    return Envelope(
        [
            Segment(attack, 0.0, 1.0),
            Segment(decay, 1.0, sustain_level),
            Segment(sustain, sustain_level, sustain_level),
            Segment(release, sustain_level, 0.0),
        ],
        sustain_node=2,
        sample_rate=sample_rate,
    )
