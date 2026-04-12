from __future__ import annotations

import math
from typing import Literal

from pysynth.music.pitch import Pitch


def _hz_match(target: float, candidates: list[float]) -> bool:
    return any(math.isclose(target, c, rel_tol=1e-9) for c in candidates)


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

    def __getitem__(self, key: int | list[int] | slice) -> Pitch | Scale:
        if isinstance(key, int):
            if self._period is None:
                return Pitch(self._tonic * self._ratios[key])
            wrap, degree = divmod(key, len(self._ratios))
            return Pitch(self._tonic * self._ratios[degree] * self._period**wrap)
        if isinstance(key, slice):
            ratios = self._ratios[key]
        else:
            ratios = [self._ratios[i] for i in key]
        if not ratios:
            raise ValueError("empty scale")
        return Scale(self._tonic, ratios, unit="ratio", period=self._period)

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

    def __or__(self, other: Scale) -> Scale:
        """Union of two scales by absolute Hz, result tonic = lhs tonic."""
        lhs_hz = [self._tonic * r for r in self._ratios]
        rhs_hz = [other._tonic * r for r in other._ratios]
        merged = list(lhs_hz)
        for h in rhs_hz:
            if not _hz_match(h, merged):
                merged.append(h)
        ratios = sorted(h / self._tonic for h in merged)
        return Scale(self._tonic, ratios, unit="ratio")

    def __and__(self, other: Scale) -> Scale:
        """Intersection of two scales by absolute Hz, result tonic = lhs tonic."""
        rhs_hz = [other._tonic * r for r in other._ratios]
        common = [
            self._tonic * r
            for r in self._ratios
            if _hz_match(self._tonic * r, rhs_hz)
        ]
        if not common:
            raise ValueError("empty scale")
        ratios = sorted(h / self._tonic for h in common)
        return Scale(self._tonic, ratios, unit="ratio")

    def __sub__(self, other: Scale) -> Scale:
        """Difference of two scales by absolute Hz, result tonic = lhs tonic."""
        rhs_hz = [other._tonic * r for r in other._ratios]
        diff = [
            self._tonic * r
            for r in self._ratios
            if not _hz_match(self._tonic * r, rhs_hz)
        ]
        if not diff:
            raise ValueError("empty scale")
        ratios = sorted(h / self._tonic for h in diff)
        return Scale(self._tonic, ratios, unit="ratio")

    def mode(self, degree: int) -> Scale:
        """Rotate the scale so that *degree* becomes the new tonic.

        Requires ``period`` to be set — without it, the wrapped-around
        degrees below the rotation point have no defined frequency.
        """
        if self._period is None:
            raise ValueError("mode() requires a scale with a period")
        n = len(self._ratios)
        degree = degree % n
        base = self._ratios[degree]
        ratios = [
            self._ratios[(degree + i) % n]
            * (self._period if (degree + i) >= n else 1.0)
            / base
            for i in range(n)
        ]
        return Scale(self._tonic * base, ratios, unit="ratio", period=self._period)

    def __repr__(self) -> str:
        if self._period is not None:
            return f"Scale(tonic={self._tonic:.4g} Hz, degrees={len(self)}, period={self._period})"
        return f"Scale(tonic={self._tonic:.4g} Hz, degrees={len(self)})"
