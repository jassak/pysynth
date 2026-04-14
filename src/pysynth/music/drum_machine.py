from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal


class DrumMachine:
    """A multi-track drum pattern sequencer that outputs per-track gate signals.

    Each track is a named velocity pattern — a list of floats (0.0–1.0)
    defining hit velocities on a shared step grid.  Zero means silence;
    any positive value produces a gate pulse at that velocity.

    The DrumMachine outputs one gate Signal per track via :meth:`cv`.  No
    audio is generated — the user connects each gate to their own sound
    source, keeping the modular CV/gate philosophy::

        dm = DrumMachine({
            "kick":  [1, 0, 0, 0,  1, 0, 0, 0],
            "snare": [0, 0, 0, 0,  1, 0, 0, 0],
            "hat":   [.7, 0, .5, 0, .7, 0, .5, 0],
        }, bpm=120)

        gates = dm.cv(repeats=4)
        # gates["kick"]  -> Signal (gate with velocity pulses)
        # gates["snare"] -> Signal
        # gates["hat"]   -> Signal

    Parameters
    ----------
    tracks:
        Mapping of track name to velocity pattern.  All patterns are
        right-padded with zeros to match the longest pattern.
    bpm:
        Tempo in beats per minute.
    step_length:
        Duration of one step in beats (default 0.25 = sixteenth note).
    gate_length:
        Fraction of the step occupied by the gate pulse (0.0–1.0).
        Default 0.5.
    gate_lengths:
        Per-track gate length overrides.  Tracks not listed use the
        global ``gate_length``.
    retrigger_gap:
        Seconds of zero-gate inserted between consecutive hits on the
        same track so that envelopes re-trigger.
    """

    def __init__(
        self,
        tracks: dict[str, list[float]],
        bpm: float = 120.0,
        step_length: float = 0.25,
        gate_length: float = 0.5,
        gate_lengths: dict[str, float] | None = None,
        retrigger_gap: float = 0.002,
    ) -> None:
        self.tracks = tracks
        self.bpm = bpm
        self.step_length = step_length
        self.gate_length = gate_length
        self.gate_lengths = gate_lengths or {}
        self.retrigger_gap = retrigger_gap

    def cv(
        self,
        *,
        repeats: int = 1,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict[str, Signal]:
        """Return per-track gate Signals.

        All returned Signals have identical duration.  Gate height equals
        the velocity value from the pattern (0.0–1.0).

        Parameters
        ----------
        repeats:
            Number of times to repeat the pattern.
        sample_rate:
            Sample rate for the output Signals.
        """
        step_dur = self.step_length * 60.0 / self.bpm
        pattern_len = max((len(p) for p in self.tracks.values()), default=0)
        gap_samples = int(self.retrigger_gap * sample_rate)
        total_samples = int(pattern_len * repeats * step_dur * sample_rate)

        result: dict[str, Signal] = {}
        for name, pattern in self.tracks.items():
            # Pad and repeat
            padded = pattern + [0.0] * (pattern_len - len(pattern))
            steps = padded * repeats

            gl = self.gate_lengths.get(name, self.gate_length)
            gate_buf = np.zeros(total_samples, dtype=np.float32)
            prev_was_active = False

            for i, velocity in enumerate(steps):
                if velocity <= 0.0:
                    prev_was_active = False
                    continue

                start = int(i * step_dur * sample_rate)
                end = min(int((i + 1) * step_dur * sample_rate), total_samples)
                gate_end = min(start + int(gl * step_dur * sample_rate), end)
                gate_buf[start:gate_end] = velocity

                if prev_was_active and gap_samples > 0:
                    gap_end = min(start + gap_samples, gate_end)
                    gate_buf[start:gap_end] = 0.0

                prev_was_active = True

            result[name] = Signal(gate_buf, sample_rate)

        return result

    def rotate(self, n: int = 1) -> DrumMachine:
        """Return a new DrumMachine with all patterns rotated by *n* steps."""
        pattern_len = max((len(p) for p in self.tracks.values()), default=0)
        if pattern_len == 0:
            return DrumMachine({}, self.bpm, self.step_length, self.gate_length,
                               self.gate_lengths, self.retrigger_gap)
        n = n % pattern_len
        rotated = {}
        for name, pattern in self.tracks.items():
            padded = pattern + [0.0] * (pattern_len - len(pattern))
            rotated[name] = padded[n:] + padded[:n]
        return DrumMachine(rotated, self.bpm, self.step_length, self.gate_length,
                           self.gate_lengths, self.retrigger_gap)

    @classmethod
    def from_x0x(
        cls,
        tracks: dict[str, str],
        **kwargs,
    ) -> DrumMachine:
        """Build from x0x-style string patterns.

        ``x`` maps to velocity 1.0, ``-`` maps to 0.0::

            dm = DrumMachine.from_x0x({
                "kick":  "x---x---x---x---",
                "snare": "----x-------x---",
                "hat":   "x-x-x-x-x-x-x-x-",
            }, bpm=120)
        """
        converted: dict[str, list[float]] = {}
        for name, s in tracks.items():
            converted[name] = [1.0 if c == "x" else 0.0 for c in s]
        return cls(converted, **kwargs)
