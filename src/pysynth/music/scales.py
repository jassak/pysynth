from __future__ import annotations

from typing import Literal

from pysynth.music.pitch import Pitch


class Scale:
    """A mapping from integer degree indices to Pitch values.

    Defined by a tonic and a list of intervals. No Western or octave-based
    assumptions are made — the intervals can be any positive ratios (or cents),
    span any range, and number any count.

    Parameters
    ----------
    tonic:
        The reference frequency (Hz or Pitch) for degree 0.
    intervals:
        Interval values for each degree. With ``unit="ratio"`` (default),
        each value is a positive multiplier applied to the tonic.
        With ``unit="cents"``, values are cent offsets from the tonic.
    unit:
        ``"ratio"`` (default) or ``"cents"``.

    Examples::

        # Just intonation pentatonic
        scale = Scale(440, [1, 9/8, 5/4, 3/2, 5/3])

        # 12-tone equal temperament (all 12 semitones)
        scale = Scale(440, [2 ** (n / 12) for n in range(13)])

        # Same, expressed in cents
        scale = Scale(440, [n * 100 for n in range(13)], unit="cents")

        # Bohlen-Pierce: 13 steps spanning a tritave (3:1)
        scale = Scale(440, [3 ** (n / 13) for n in range(14)])

        # Completely arbitrary microtonal
        scale = Scale(300, [1, 1.07, 1.17, 1.31, 1.52, 1.78, 2.0])

    Transposing a scale::

        scale_up = scale * 2          # tonic doubled (octave up, ratio-wise)
        scale_Bb = scale * (Bb / A)   # modulate to a new tonic

    Accessing degrees::

        p = scale[0]    # tonic Pitch
        p = scale[3]    # fourth degree
        for p in scale: # iterate all degrees
    """

    def __init__(
        self,
        tonic: float | Pitch,
        intervals: list[float],
        unit: Literal["ratio", "cents"] = "ratio",
        period: float | None = None,
    ) -> None:
        self._tonic = tonic.hz if isinstance(tonic, Pitch) else float(tonic)
        self._unit = unit

        if period is not None and period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self._period = period

        if unit == "cents":
            self._ratios = [2.0 ** (c / 1200.0) for c in intervals]
        else:
            self._ratios = [float(r) for r in intervals]

    @property
    def tonic(self) -> Pitch:
        return Pitch(self._tonic)

    @property
    def period(self) -> float | None:
        return self._period

    def __getitem__(self, n: int) -> Pitch:
        if self._period is None:
            return Pitch(self._tonic * self._ratios[n])
        wrap, degree = divmod(n, len(self._ratios))
        return Pitch(self._tonic * self._ratios[degree] * self._period**wrap)

    def __len__(self) -> int:
        return len(self._ratios)

    def __iter__(self):
        return (self[n] for n in range(len(self)))

    def __mul__(self, ratio: float) -> Scale:
        """Transpose the entire scale up by a ratio."""
        return Scale(self._tonic * ratio, self._ratios, unit="ratio", period=self._period)

    def __truediv__(self, ratio: float) -> Scale:
        """Transpose the entire scale down by a ratio."""
        return Scale(self._tonic / ratio, self._ratios, unit="ratio", period=self._period)

    def __repr__(self) -> str:
        if self._period is not None:
            return f"Scale(tonic={self._tonic:.4g} Hz, degrees={len(self)}, period={self._period})"
        return f"Scale(tonic={self._tonic:.4g} Hz, degrees={len(self)})"
