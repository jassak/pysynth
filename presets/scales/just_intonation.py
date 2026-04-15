"""Just intonation and Pythagorean scales defined by exact frequency ratios."""

from __future__ import annotations

from pysynth import Pitch, Scale


def _hz(tonic: float | Pitch) -> float:
    return tonic.hz if isinstance(tonic, Pitch) else float(tonic)


def ji_major(tonic: float | Pitch = 220.0) -> Scale:
    """Just intonation major scale (Ptolemy's intense diatonic)."""
    return Scale(_hz(tonic), [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8], period=2)


def ji_minor(tonic: float | Pitch = 220.0) -> Scale:
    """Just intonation natural minor."""
    return Scale(_hz(tonic), [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5], period=2)


def ji_pentatonic(tonic: float | Pitch = 220.0) -> Scale:
    """Just intonation pentatonic."""
    return Scale(_hz(tonic), [1, 9/8, 5/4, 3/2, 5/3], period=2)


def pythagorean(tonic: float | Pitch = 220.0) -> Scale:
    """Pythagorean tuning. All intervals derived from pure fifths (3:2)."""
    return Scale(
        _hz(tonic),
        [1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128],
        period=2,
    )
