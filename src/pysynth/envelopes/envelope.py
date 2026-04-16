from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from pysynth._core import SAMPLE_RATE, Signal


@numba.njit(cache=True)
def _envelope_trigger(gate_data, out, seg_data, offsets, n_segs, sustain_node, sustain_level):
    seg_idx = n_segs
    seg_pos = 0
    level = 0.0
    prev_gate = 0.0

    for i in range(len(gate_data)):
        g = gate_data[i]

        # Rising edge: restart from segment 0
        if g > 0.0 and prev_gate <= 0.0:
            seg_idx = 0
            seg_pos = 0

        # Falling edge: jump to release segments
        if g <= 0.0 and prev_gate > 0.0:
            if sustain_node >= 0 and sustain_node + 1 < n_segs:
                seg_idx = sustain_node + 1
                seg_pos = 0
            else:
                seg_idx = n_segs

        if seg_idx < n_segs:
            seg_start = offsets[seg_idx]
            seg_len = offsets[seg_idx + 1] - seg_start

            if sustain_node >= 0 and seg_idx == sustain_node and g > 0.0 and seg_pos >= seg_len:
                level = sustain_level
            elif seg_pos < seg_len:
                level = seg_data[seg_start + seg_pos]
                seg_pos += 1
            else:
                seg_idx += 1
                seg_pos = 0
                if seg_idx < n_segs:
                    seg_start2 = offsets[seg_idx]
                    seg_len2 = offsets[seg_idx + 1] - seg_start2
                    if seg_pos < seg_len2:
                        level = seg_data[seg_start2 + seg_pos]
                        seg_pos += 1
                else:
                    level = 0.0
        else:
            level = 0.0

        out[i] = level
        prev_gate = g


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

        Internally delegates to :meth:`trigger` with a synthetic gate.
        For sustain-node envelopes the gate drops before the end to allow
        the release phase; for one-shot envelopes the gate stays high.

        If ``duration`` is ``None`` the natural length (sum of segment
        durations) is used.  If ``sample_rate`` is given it overrides the
        envelope's default.
        """
        sr = sample_rate if sample_rate is not None else self.sample_rate

        if duration is None:
            duration = sum(seg.duration for seg in self.segments)

        n = int(duration * sr)
        if n == 0:
            return Signal(np.empty(0, dtype=np.float32), sr)

        if self.sustain_node is not None:
            release_dur = sum(
                seg.duration for seg in self.segments[self.sustain_node + 1:]
            )
            n_high = max(0, n - int(release_dur * sr))
            gate_data = np.zeros(n, dtype=np.float32)
            gate_data[:n_high] = 1.0
        else:
            gate_data = np.ones(n, dtype=np.float32)

        return self.trigger(Signal(gate_data, sr))

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

        # Pre-render each segment as an array
        seg_arrays = [seg.render(sr) for seg in self.segments]
        n_segs = len(seg_arrays)
        sn = self.sustain_node

        # Determine sustain level (end value of sustain_node segment)
        if sn is not None and sn < n_segs:
            sustain_level = float(self.segments[sn].end)
        else:
            sustain_level = 0.0

        # Flatten seg_arrays into CSR-style storage for numba
        if seg_arrays:
            seg_data = np.concatenate([a.astype(np.float64) for a in seg_arrays])
        else:
            seg_data = np.empty(0, dtype=np.float64)
        offsets = np.zeros(n_segs + 1, dtype=np.int64)
        for k, a in enumerate(seg_arrays):
            offsets[k + 1] = offsets[k] + len(a)

        out = np.empty(n_samples, dtype=np.float64)
        _envelope_trigger(
            gate_data.astype(np.float64),
            out,
            seg_data,
            offsets,
            n_segs,
            sn if sn is not None else -1,
            sustain_level,
        )
        return Signal(out.astype(np.float32), sr)


def adsr(
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sample_rate: int = SAMPLE_RATE,
) -> Envelope:
    """Construct a standard ADSR Envelope.

    Parameters
    ----------
    attack:  rise time from 0 to 1 (seconds)
    decay:   fall time from 1 to *sustain* (seconds)
    sustain: amplitude held while the gate is high (0..1)
    release: fall time from *sustain* to 0 (seconds)
    """
    return Envelope([
            Segment(attack, 0.0, 1.0),
            Segment(decay, 1.0, sustain),
            Segment(release, sustain, 0.0),
        ],
        sustain_node=1,
        sample_rate=sample_rate,
    )
