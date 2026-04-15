"""Microtonal and xenharmonic scales."""

from __future__ import annotations

from pysynth import Pitch, Scale


def _hz(tonic: float | Pitch) -> float:
    return tonic.hz if isinstance(tonic, Pitch) else float(tonic)


def bohlen_pierce(tonic: float | Pitch = 220.0) -> Scale:
    """Bohlen-Pierce scale. 13 equal divisions of the tritave (3:1)."""
    return Scale(
        _hz(tonic),
        [3 ** (n / 13) for n in range(13)],
        period=3,
    )


def quarter_tone(tonic: float | Pitch = 220.0) -> Scale:
    """24-TET (quarter-tone) chromatic scale."""
    return Scale(
        _hz(tonic),
        [2 ** (n / 24) for n in range(24)],
        period=2,
    )


def edo19(tonic: float | Pitch = 220.0) -> Scale:
    """19 equal divisions of the octave."""
    return Scale(
        _hz(tonic),
        [2 ** (n / 19) for n in range(19)],
        period=2,
    )


def edo31(tonic: float | Pitch = 220.0) -> Scale:
    """31 equal divisions of the octave."""
    return Scale(
        _hz(tonic),
        [2 ** (n / 31) for n in range(31)],
        period=2,
    )
