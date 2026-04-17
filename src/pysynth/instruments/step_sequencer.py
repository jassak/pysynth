from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music import Pitch

_TIE_SENTINEL = float("nan")


@dataclass(frozen=True)
class Step:
    """A single step in a step sequencer.

    ``value`` is intentionally a plain float — it can represent pitch (Hz),
    filter cutoff, resonance, amplitude, or any other parameter.  Use one
    :class:`StepSequencer` per parameter lane and wire the output Signals
    together via the Signal algebra.
    """

    value: float
    gate_length: float = 0.75
    slide: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.value, Pitch):
            object.__setattr__(self, "value", self.value.hz)

    @classmethod
    def rest(cls) -> Step:
        """A silent step (gate off, value 0)."""
        return cls(value=0.0, gate_length=0.0)

    @classmethod
    def tie(cls) -> Step:
        """Extend the previous step (gate stays high, value holds)."""
        return cls(value=_TIE_SENTINEL, gate_length=1.0)

    @property
    def is_rest(self) -> bool:
        return self.gate_length == 0.0 and not self.is_tie

    @property
    def is_tie(self) -> bool:
        return math.isnan(self.value)

    def __repr__(self) -> str:
        if self.is_tie:
            return "Step(tie)"
        if self.is_rest:
            return "Step(rest)"
        parts = [f"{self.value}"]
        if self.gate_length != 0.75:
            parts.append(f"gate={self.gate_length}")
        if self.slide:
            parts.append("slide")
        return f"Step({', '.join(parts)})"


class StepSequencer:
    """A parameter-agnostic step sequencer that outputs generic CV signals.

    Each :class:`Step` carries a ``value`` (any float) and timing metadata.
    The sequencer converts the step grid into two Signals:

    - **value**: the sequenced parameter at each sample
    - **gate**: 1.0 while a step is active (respecting ``gate_length``),
      0.0 otherwise

    Multiple StepSequencers can run in parallel to control different
    parameters (pitch, filter cutoff, resonance, …) over the same grid::

        from pysynth.music import Scale
        from pysynth.instruments import StepSequencer, Step

        scale = Scale(220, [1, 9/8, 5/4, 3/2])

        # Pitch lane
        pitch_steps = [Step(scale[0].hz), Step(scale[2].hz, slide=True),
                       Step.tie(), Step.rest()]
        pitch, gate = StepSequencer(pitch_steps, bpm=130).cv(repeats=4)

        # Filter cutoff lane
        cutoff_steps = [Step(2000), Step(800, slide=True), Step(3000), Step(500)]
        cutoff, _ = StepSequencer(cutoff_steps, bpm=130).cv(repeats=4)
    """

    def __init__(
        self,
        steps: list[Step],
        bpm: float = 120.0,
        step_length: float = 0.25,
        slide_time: float = 0.02,
        retrigger_gap: float = 0.002,
    ) -> None:
        self.steps = steps
        self.bpm = bpm
        self.step_length = step_length
        self.slide_time = slide_time
        self.retrigger_gap = retrigger_gap

    def cv(
        self,
        *,
        repeats: int = 1,
        sample_rate: int = SAMPLE_RATE,
    ) -> tuple[Signal, Signal]:
        """Return ``(value, gate)`` control signals for the step pattern.

        Parameters
        ----------
        repeats:
            Number of times to repeat the step pattern.
        sample_rate:
            Sample rate for the output Signals.
        """
        step_dur = self.step_length * 60.0 / self.bpm
        steps = self.steps * repeats

        total_samples = int(len(steps) * step_dur * sample_rate)
        value_buf = np.zeros(total_samples, dtype=np.float32)
        gate_buf = np.zeros(total_samples, dtype=np.float32)

        gap_samples = int(self.retrigger_gap * sample_rate)
        prev_value = 0.0
        prev_was_active = False
        step_starts: list[int] = []

        for i, step in enumerate(steps):
            start = int(i * step_dur * sample_rate)
            end = min(int((i + 1) * step_dur * sample_rate), total_samples)
            step_starts.append(start)

            if step.is_tie:
                value_buf[start:end] = prev_value
                gate_buf[start:end] = 1.0
                # ties keep prev_was_active as-is (no retrigger needed)
            elif step.is_rest:
                # Sample-and-hold: keep the last value through rests
                # so the oscillator continues producing audio for the
                # envelope's release phase.
                if prev_value != 0.0:
                    value_buf[start:end] = prev_value
                prev_was_active = False
            else:
                value_buf[start:end] = step.value
                gate_end = min(start + int(step.gate_length * step_dur * sample_rate), end)
                gate_buf[start:gate_end] = 1.0
                # Insert retrigger gap when consecutive active steps
                if prev_was_active and gap_samples > 0:
                    gap_end = min(start + gap_samples, gate_end)
                    gate_buf[start:gap_end] = 0.0
                prev_value = step.value
                prev_was_active = True

        # Slide pass — TB-303 style: a step with slide=True glides into the
        # *next* step's value.  The ramp straddles the boundary between this
        # step and the next.
        slide_samples = int(self.slide_time * sample_rate)
        for i, step in enumerate(steps):
            if not step.slide or step.is_rest or step.is_tie:
                continue
            # Find the next active value
            next_val = None
            for j in range(i + 1, len(steps)):
                if steps[j].is_tie:
                    continue
                if not steps[j].is_rest:
                    next_val = steps[j].value
                    break
            if next_val is None or next_val == step.value:
                continue

            # Place the ramp centred on the boundary to the next step
            if i + 1 < len(step_starts):
                boundary = step_starts[i + 1]
            else:
                continue
            ramp_start = max(boundary - slide_samples, 0)
            ramp_end = min(boundary + slide_samples, total_samples)
            ramp_len = ramp_end - ramp_start
            if ramp_len > 0:
                ramp = np.linspace(step.value, next_val, ramp_len, dtype=np.float32)
                value_buf[ramp_start:ramp_end] = ramp

        return Signal(value_buf, sample_rate), Signal(gate_buf, sample_rate)

    def rotate(self, n: int = 1) -> StepSequencer:
        """Return a new StepSequencer with the pattern rotated by *n* steps."""
        if not self.steps:
            return StepSequencer([], self.bpm, self.step_length, self.slide_time, self.retrigger_gap)
        n = n % len(self.steps)
        rotated = self.steps[n:] + self.steps[:n]
        return StepSequencer(rotated, self.bpm, self.step_length, self.slide_time, self.retrigger_gap)

    @classmethod
    def from_values(
        cls,
        values: list[float | None],
        **kwargs,
    ) -> StepSequencer:
        """Build a step sequence from a list of values.

        ``None`` entries become rests.
        """
        steps = [Step(v) if v is not None else Step.rest() for v in values]
        return cls(steps, **kwargs)
