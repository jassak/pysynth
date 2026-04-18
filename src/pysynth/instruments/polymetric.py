from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.scales import Scale


class PolymetricSequencer:
    """A multi-track tonal sequencer where each track loops independently.

    Each track is a pattern of scale degrees (integers) and rests (``None``)
    that loops within a user-specified beat window.  Tracks can have
    different pattern lengths and step lengths, producing polymetric
    phasing.

    Output is per-track ``(pitch, gate)`` CV pairs — no audio is generated.
    The user connects each track to their own oscillator and envelope::

        scale = Scale(220, [1, 9/8, 5/4, 3/2, 5/3], period=2.0)

        seq = PolymetricSequencer({
            "melody": [0, 2, 4, 3],
            "bass":   [0, None, 4],
        }, scale=scale, bpm=120,
           step_lengths={"bass": 1/3},
        )

        signals = seq.cv(beats=16)
        pitch, gate = signals["melody"]
        audio = Oscillator("saw").render(pitch.duration, pitch)
        output = audio * adsr(0.01, 0.1, 0.7, 0.1).trigger(gate)

    Parameters
    ----------
    tracks:
        Mapping of track name to pattern.  Each pattern is a list of
        scale degree integers (indexing into *scale*) or ``None`` for
        rests.  Also accepts a plain list of patterns, which are
        auto-named ``track_0``, ``track_1``, etc.
    scale:
        The shared Scale instance.  Degrees index into it via
        ``scale[degree]``.
    bpm:
        Tempo in beats per minute.
    step_length:
        Default duration of one step in beats (0.25 = sixteenth note).
    gate_length:
        Default fraction of the step occupied by the gate pulse.
    step_lengths:
        Per-track step length overrides.
    gate_lengths:
        Per-track gate length overrides.
    retrigger_gap:
        Seconds of zero-gate between consecutive non-rest steps so
        that envelopes re-trigger.
    """

    def __init__(
        self,
        tracks: dict[str, list[int | None]] | list[list[int | None]],
        scale: Scale,
        bpm: float = 120.0,
        step_length: float = 0.25,
        gate_length: float = 0.75,
        step_lengths: dict[str, float] | None = None,
        gate_lengths: dict[str, float] | None = None,
        retrigger_gap: float = 0.002,
    ) -> None:
        if isinstance(tracks, list):
            tracks = {i: t for i, t in enumerate(tracks)}
        self.tracks = tracks
        self.scale = scale
        self.bpm = bpm
        self.step_length = step_length
        self.gate_length = gate_length
        self.step_lengths = step_lengths or {}
        self.gate_lengths = gate_lengths or {}
        self.retrigger_gap = retrigger_gap

    def cv(
        self,
        *,
        beats: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict[str, tuple[Signal, Signal]]:
        """Return per-track ``(pitch, gate)`` control signals.

        Each track's pattern loops independently within the beat window.
        All returned Signals have identical duration.

        Parameters
        ----------
        beats:
            Total duration in beats.  Required — polymetric patterns
            have no natural single-cycle length.
        sample_rate:
            Sample rate for the output Signals.
        """
        beat_dur = 60.0 / self.bpm
        total_samples = int(beats * beat_dur * sample_rate)
        gap_samples = int(self.retrigger_gap * sample_rate)

        result: dict[str, tuple[Signal, Signal]] = {}
        for name, pattern in self.tracks.items():
            pitch_buf = np.zeros(total_samples, dtype=np.float32)
            gate_buf = np.zeros(total_samples, dtype=np.float32)

            pattern_len = len(pattern)
            if pattern_len == 0:
                result[name] = (
                    Signal(pitch_buf, sample_rate),
                    Signal(gate_buf, sample_rate),
                )
                continue

            sl = self.step_lengths.get(name, self.step_length)
            gl = self.gate_lengths.get(name, self.gate_length)
            step_dur = sl * beat_dur

            prev_was_active = False
            last_hz = 0.0
            step_index = 0

            while True:
                start = int(step_index * step_dur * sample_rate)
                if start >= total_samples:
                    break
                end = min(int((step_index + 1) * step_dur * sample_rate), total_samples)

                degree = pattern[step_index % pattern_len]

                if degree is None:
                    # Sample-and-hold: keep the last pitch through rests
                    # so the oscillator continues producing audio for the
                    # envelope's release phase.
                    if last_hz > 0.0:
                        pitch_buf[start:end] = last_hz
                    prev_was_active = False
                else:
                    hz = self.scale[degree].hz
                    pitch_buf[start:end] = hz
                    gate_end = min(start + int(gl * step_dur * sample_rate), end)
                    gate_buf[start:gate_end] = 1.0

                    if prev_was_active and gap_samples > 0:
                        gap_end = min(start + gap_samples, gate_end)
                        gate_buf[start:gap_end] = 0.0

                    last_hz = hz
                    prev_was_active = True

                step_index += 1

            result[name] = (
                Signal(pitch_buf, sample_rate),
                Signal(gate_buf, sample_rate),
            )

        return result

    @classmethod
    def from_notation(
        cls,
        tracks: dict[str, str] | list[str],
        scale: Scale,
        **kwargs,
    ) -> PolymetricSequencer:
        """Build from space-separated string notation.

        ``.`` is a rest, integers are scale degrees::

            seq = PolymetricSequencer.from_notation({
                "melody": "0 2 4 3",
                "bass":   "0 . 4",
            }, scale=scale, bpm=120)

        Negative degrees (e.g. ``-1``) work if the scale supports them.
        """
        if isinstance(tracks, list):
            parsed = {i: cls._parse(s) for i, s in enumerate(tracks)}
        else:
            parsed = {name: cls._parse(s) for name, s in tracks.items()}
        return cls(parsed, scale=scale, **kwargs)

    @staticmethod
    def _parse(notation: str) -> list[int | None]:
        result: list[int | None] = []
        for tok in notation.split():
            if tok == ".":
                result.append(None)
            else:
                result.append(int(tok))
        return result
