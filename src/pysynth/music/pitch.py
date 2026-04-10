from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pitch:
    """A frequency, measured in Hz.

    Pitches form a **torsor** over the multiplicative group of positive reals
    (intervals expressed as ratios). The natural operations are:

        pitch * ratio  -> Pitch transposed up by that ratio
        pitch / ratio  -> Pitch transposed down by that ratio
        pitch / pitch  -> float ratio between the two pitches

    Addition of two Pitches is intentionally undefined — adding Hz values has
    no musical meaning. Transposition is always multiplicative.

    Examples::

        a = Pitch(440.0)
        octave_up   = a * 2          # Pitch(880.0)
        fifth_up    = a * (3/2)      # Pitch(660.0)  — just intonation
        interval    = Pitch(660) / Pitch(440)  # 1.5  (a perfect fifth)
        tritone_up  = a.up(600)      # Pitch(622.3…) — 600 cents

    The identity element is ``pitch * 1``, and the inverse of ratio ``r`` is
    ``1/r``, so the group of intervals is (ℝ₊, ×).
    """

    hz: float

    def __mul__(self, ratio: float) -> Pitch:
        return Pitch(self.hz * ratio)

    def __rmul__(self, ratio: float) -> Pitch:
        return Pitch(self.hz * ratio)

    def __truediv__(self, other: float | Pitch) -> float | Pitch:
        if isinstance(other, Pitch):
            # pitch / pitch -> ratio (the interval between them)
            return self.hz / other.hz
        # pitch / ratio -> transposed Pitch
        return Pitch(self.hz / other)

    def up(self, cents: float) -> Pitch:
        """Return a new Pitch shifted up by the given number of cents."""
        return Pitch(self.hz * 2.0 ** (cents / 1200.0))

    def down(self, cents: float) -> Pitch:
        """Return a new Pitch shifted down by the given number of cents."""
        return self.up(-cents)

    def __repr__(self) -> str:
        return f"Pitch({self.hz:.4g} Hz)"


@dataclass(frozen=True)
class Note:
    """A pitched event with a duration expressed in beats.

    Duration is in beats so that the same phrase can be rendered at different
    tempos by changing ``bpm`` in the Sequencer without editing note data.
    Use ``bpm=60`` to treat durations as seconds.

    Transposition mirrors Pitch algebra — multiplication shifts the pitch,
    duration and velocity are preserved::

        note * 2        -> octave up
        note * (3/2)    -> just perfect fifth up
        note / 2        -> octave down
        note_b / note_a -> float ratio between the two pitches

    ``Note.rest(duration)`` produces a silent placeholder (pitch.hz == 0).
    """

    pitch: Pitch
    duration: float  # beats
    velocity: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.pitch, (int, float)):
            object.__setattr__(self, "pitch", Pitch(float(self.pitch)))

    def __mul__(self, ratio: float) -> Note:
        return Note(self.pitch * ratio, self.duration, self.velocity)

    def __rmul__(self, ratio: float) -> Note:
        return Note(self.pitch * ratio, self.duration, self.velocity)

    def __truediv__(self, other: float | Note) -> float | Note:
        if isinstance(other, Note):
            return self.pitch.hz / other.pitch.hz
        return Note(self.pitch / other, self.duration, self.velocity)

    @property
    def is_rest(self) -> bool:
        return self.pitch.hz <= 0.0

    @classmethod
    def rest(cls, duration: float) -> Note:
        return cls(Pitch(0.0), duration, 0.0)

    def __repr__(self) -> str:
        if self.is_rest:
            return f"Note(rest, {self.duration}b)"
        return f"Note({self.pitch.hz:.4g} Hz, {self.duration}b, vel={self.velocity})"
