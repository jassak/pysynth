from __future__ import annotations

import math
from typing import Literal

from pysynth.music.pitch import Pitch


def _hz_match(target: float, candidates: list[float]) -> bool:
    return any(math.isclose(target, c, rel_tol=1e-9) for c in candidates)


def _fold_ratio(r: float, period: float) -> float:
    """Normalise a ratio into [1, period)."""
    while r >= period - 1e-12:
        r /= period
    while r < 1.0 - 1e-12:
        r *= period
    return r


def _dedup(hz_list: list[float]) -> list[float]:
    """Remove near-duplicate Hz values, keeping the first occurrence."""
    out: list[float] = []
    for h in hz_list:
        if not _hz_match(h, out):
            out.append(h)
    return out


def _fold_hz(hz: float, ref_tonic: float, period: float) -> float:
    """Fold an absolute Hz value into [ref_tonic, ref_tonic * period)."""
    r = hz / ref_tonic
    r = _fold_ratio(r, period)
    return ref_tonic * r


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

    Deriving scales by index::

        chromatic = Scale(440, [2 ** (n / 12) for n in range(12)], period=2.0)
        major      = chromatic[[0, 2, 4, 5, 7, 9, 11]]
        whole_tone = chromatic[::2]

    Set operations (compare by absolute Hz, result tonic = lhs).
    When both scales share the same ``period``, comparison is
    mod-period (pitch-class aware)::

        diminished  = chromatic[::3] | chromatic[1::3]   # union
        common      = A_major & C_major                  # intersection
        accidentals = chromatic - major                   # difference

    Fold ratios into one period (requires ``period``)::

        scale.reduce()               # deduplicate across octaves

    Modes (requires ``period``)::

        dorian    = major.mode(1)
        phrygian  = major.mode(2)
        mixolydian = major.mode(4)

    Audition::

        scale.preview()              # play ascending, sine wave, 120 bpm
        scale.preview(bpm=80)        # slower
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

    def _shared_period(self, other: Scale) -> float | None:
        if (
            self._period is not None
            and other._period is not None
            and math.isclose(self._period, other._period, rel_tol=1e-9)
        ):
            return self._period
        return None

    def _to_hz(self, other: Scale) -> tuple[list[float], list[float], float | None]:
        """Compute comparable Hz lists for set operations.

        When both scales share the same period, Hz values are folded into
        ``[self._tonic, self._tonic * period)`` so that pitch classes match
        across octaves and tonics.
        """
        sp = self._shared_period(other)
        lhs_hz = [self._tonic * r for r in self._ratios]
        rhs_hz = [other._tonic * r for r in other._ratios]
        if sp is not None:
            lhs_hz = [_fold_hz(h, self._tonic, sp) for h in lhs_hz]
            rhs_hz = [_fold_hz(h, self._tonic, sp) for h in rhs_hz]
            # Deduplicate within each side after folding
            lhs_hz = _dedup(lhs_hz)
            rhs_hz = _dedup(rhs_hz)
        return lhs_hz, rhs_hz, sp

    def __or__(self, other: Scale) -> Scale:
        """Union of two scales by absolute Hz, result tonic = lhs tonic.

        When both scales share the same period, comparison is mod-period
        and the result is reduced to one period.
        """
        lhs_hz, rhs_hz, sp = self._to_hz(other)
        merged = list(lhs_hz)
        for h in rhs_hz:
            if not _hz_match(h, merged):
                merged.append(h)
        ratios = sorted(h / self._tonic for h in merged)
        return Scale(self._tonic, ratios, unit="ratio", period=sp)

    def __and__(self, other: Scale) -> Scale:
        """Intersection of two scales by absolute Hz, result tonic = lhs tonic.

        When both scales share the same period, comparison is mod-period
        and the result is reduced to one period.
        """
        lhs_hz, rhs_hz, sp = self._to_hz(other)
        common = [h for h in lhs_hz if _hz_match(h, rhs_hz)]
        if not common:
            raise ValueError("empty scale")
        ratios = sorted(h / self._tonic for h in common)
        return Scale(self._tonic, ratios, unit="ratio", period=sp)

    def __sub__(self, other: Scale) -> Scale:
        """Difference of two scales by absolute Hz, result tonic = lhs tonic.

        When both scales share the same period, comparison is mod-period
        and the result is reduced to one period.
        """
        lhs_hz, rhs_hz, sp = self._to_hz(other)
        diff = [h for h in lhs_hz if not _hz_match(h, rhs_hz)]
        if not diff:
            raise ValueError("empty scale")
        ratios = sorted(h / self._tonic for h in diff)
        return Scale(self._tonic, ratios, unit="ratio", period=sp)

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
            self._ratios[(degree + i) % n] * (self._period if (degree + i) >= n else 1.0) / base for i in range(n)
        ]
        return Scale(self._tonic * base, ratios, unit="ratio", period=self._period)

    def reduce(self) -> Scale:
        """Fold all ratios into one period and deduplicate.

        Requires ``period`` to be set. Each ratio is normalised into
        ``[1, period)`` and duplicates (within float tolerance) are removed.
        The result is sorted ascending with the same tonic and period.
        """
        if self._period is None:
            raise ValueError("reduce() requires a scale with a period")
        folded: list[float] = []
        for r in self._ratios:
            f = _fold_ratio(r, self._period)
            if not any(math.isclose(f, x, rel_tol=1e-9) for x in folded):
                folded.append(f)
        folded.sort()
        return Scale(self._tonic, folded, unit="ratio", period=self._period)

    def plot(self, ax=None):
        """Visualise the scale.

        Non-periodic scales are drawn as a log-frequency line with each
        degree marked and labelled. Periodic scales are drawn as a circle
        (clock-face style) where angular position reflects the log-ratio
        within one period — identical to how a chromatic circle works for
        12-TET but generalised to any period and any set of intervals.

        Parameters
        ----------
        ax:
            An existing ``matplotlib.axes.Axes`` to draw into. If ``None``
            (default), a new figure is created and ``plt.show()`` is called.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        pitches = list(self)
        hz_vals = [p.hz for p in pitches]

        if self._period is None:
            # ── Line plot (linear / non-periodic) ──────────────────────────
            if ax is None:
                fig, ax = plt.subplots(figsize=(max(6, len(pitches) * 0.8), 3))
                show = True
            else:
                ax = ax
                fig = ax.get_figure()
                show = False
            ax.set_xscale("log")

            ax.plot(hz_vals, [0.0] * len(hz_vals), color="steelblue",
                    linewidth=1.2, zorder=2)
            ax.scatter(hz_vals, [0.0] * len(hz_vals), s=70,
                       color="steelblue", zorder=3)

            for i, hz in enumerate(hz_vals):
                ax.annotate(
                    f"{i}\n{hz:.1f} Hz",
                    xy=(hz, 0.0),
                    ha="center", va="bottom",
                    xytext=(0, 14), textcoords="offset points",
                    fontsize=9,
                )

            ax.set_yticks([])
            ax.set_xlabel("Frequency (Hz, log scale)")
            ax.set_title(repr(self))
            for spine in ("left", "top", "right"):
                ax.spines[spine].set_visible(False)
        else:
            # ── Circle plot (periodic) ──────────────────────────────────────
            # Fold each degree's Hz into [tonic, tonic*period) so that pitch
            # classes from different octaves land at the same angle.
            def _angle(hz: float) -> float:
                r = _fold_ratio(hz / self._tonic, self._period)
                return 2.0 * math.pi * math.log(r) / math.log(self._period)

            angles = [_angle(hz) for hz in hz_vals]
            # 12-o'clock = tonic, clockwise → sin/cos with angle measured
            # from the top: x = sin(a), y = cos(a)
            xs = [math.sin(a) for a in angles]
            ys = [math.cos(a) for a in angles]

            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 6))
                show = True
            else:
                ax = ax
                fig = ax.get_figure()
                show = False
            ax.set_aspect("equal")

            # Outer circle
            theta = np.linspace(0.0, 2.0 * math.pi, 300)
            ax.plot(np.sin(theta), np.cos(theta),
                    color="lightgray", linewidth=1.5, zorder=1)

            # Spokes from centre to each degree
            for x, y in zip(xs, ys):
                ax.plot([0.0, x], [0.0, y],
                        color="lightsteelblue", linewidth=0.8, zorder=1)

            # Degree points
            ax.scatter(xs, ys, s=80, color="steelblue", zorder=3)

            # Labels just outside the circle
            label_r = 1.28
            for i, (a, hz) in enumerate(zip(angles, hz_vals)):
                lx = label_r * math.sin(a)
                ly = label_r * math.cos(a)
                ax.annotate(
                    f"{i}\n{hz:.1f} Hz",
                    xy=(lx, ly),
                    ha="center", va="center",
                    fontsize=9,
                )

            # Centre dot
            ax.scatter([0.0], [0.0], s=20, color="gray", zorder=2)

            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.axis("off")
            ax.set_title(repr(self), pad=12)

        fig.tight_layout()
        if show:
            plt.show()

    def preview(self, bpm: float = 120.0, blocking: bool = True) -> None:
        """Play the scale degrees ascending through the default audio device."""
        from pysynth.envelopes import adsr
        from pysynth.generators import Oscillator
        from pysynth.music.pitch import Note
        from pysynth.instruments.sequencer import Sequencer

        notes = [Note(self[i], 0.5) for i in range(len(self))]
        pitch, gate = Sequencer(notes, bpm=bpm, retrigger_gap=0.01).cv()
        audio = Oscillator("sine").render(pitch.duration, pitch)
        env = adsr(0.01, 0.02, 0.6, 0.01).trigger(gate)
        (audio * env * 0.5).play(blocking=blocking)

    def __repr__(self) -> str:
        if self._period is not None:
            return f"Scale(tonic={self._tonic:.4g} Hz, degrees={len(self)}, period={self._period})"
        return f"Scale(tonic={self._tonic:.4g} Hz, degrees={len(self)})"
