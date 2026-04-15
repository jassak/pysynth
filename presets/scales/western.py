"""Western scales built from 12-tone equal temperament.

Every function takes an optional *tonic* (Hz float or ``Pitch``, default A 220)
and returns a ``Scale`` with ``period=2`` (octave-repeating).

Scales are organised in a derivation tree:

    chromatic (12 semitones)
    ├── major (Ionian) ─── dorian, phrygian, lydian, mixolydian, minor, locrian
    ├── harmonic_minor ─── locrian_nat6, ionian_sharp5, dorian_sharp4,
    │                      phrygian_dominant, lydian_sharp2, ultralocrian
    ├── melodic_minor ──── dorian_b2, lydian_augmented, lydian_dominant,
    │                      mixolydian_b6, aeolian_b5, altered
    ├── major_pentatonic, minor_pentatonic, blues, blues_major
    ├── whole_tone, diminished_whole_half, diminished_half_whole
    └── bebop_dominant, bebop_major, bebop_dorian, bebop_melodic_minor
"""

from __future__ import annotations

from pysynth import Pitch, Scale


def _hz(tonic: float | Pitch) -> float:
    return tonic.hz if isinstance(tonic, Pitch) else float(tonic)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

def chromatic(tonic: float | Pitch = 220.0) -> Scale:
    """All 12 semitones of 12-TET."""
    return Scale(_hz(tonic), [2 ** (n / 12) for n in range(12)], period=2)


# ---------------------------------------------------------------------------
# Diatonic modes (derived from major)
# ---------------------------------------------------------------------------

def major(tonic: float | Pitch = 220.0) -> Scale:
    """Major scale (Ionian mode)."""
    return chromatic(tonic)[[0, 2, 4, 5, 7, 9, 11]]


def dorian(tonic: float | Pitch = 220.0) -> Scale:
    """Dorian mode. Mode 2 of major."""
    return chromatic(tonic)[[0, 2, 3, 5, 7, 9, 10]]


def phrygian(tonic: float | Pitch = 220.0) -> Scale:
    """Phrygian mode. Mode 3 of major."""
    return chromatic(tonic)[[0, 1, 3, 5, 7, 8, 10]]


def lydian(tonic: float | Pitch = 220.0) -> Scale:
    """Lydian mode. Mode 4 of major."""
    return chromatic(tonic)[[0, 2, 4, 6, 7, 9, 11]]


def mixolydian(tonic: float | Pitch = 220.0) -> Scale:
    """Mixolydian mode. Mode 5 of major."""
    return chromatic(tonic)[[0, 2, 4, 5, 7, 9, 10]]


def minor(tonic: float | Pitch = 220.0) -> Scale:
    """Natural minor (Aeolian). Mode 6 of major."""
    return chromatic(tonic)[[0, 2, 3, 5, 7, 8, 10]]


def locrian(tonic: float | Pitch = 220.0) -> Scale:
    """Locrian mode. Mode 7 of major."""
    return chromatic(tonic)[[0, 1, 3, 5, 6, 8, 10]]


# ---------------------------------------------------------------------------
# Harmonic minor modes
# ---------------------------------------------------------------------------

def harmonic_minor(tonic: float | Pitch = 220.0) -> Scale:
    """Harmonic minor scale."""
    return chromatic(tonic)[[0, 2, 3, 5, 7, 8, 11]]


def locrian_nat6(tonic: float | Pitch = 220.0) -> Scale:
    """Locrian natural 6. Mode 2 of harmonic minor."""
    return chromatic(tonic)[[0, 1, 3, 5, 6, 9, 10]]


def ionian_sharp5(tonic: float | Pitch = 220.0) -> Scale:
    """Ionian #5. Mode 3 of harmonic minor."""
    return chromatic(tonic)[[0, 2, 4, 5, 8, 9, 11]]


def dorian_sharp4(tonic: float | Pitch = 220.0) -> Scale:
    """Dorian #4. Mode 4 of harmonic minor."""
    return chromatic(tonic)[[0, 2, 3, 6, 7, 9, 10]]


def phrygian_dominant(tonic: float | Pitch = 220.0) -> Scale:
    """Phrygian dominant. Mode 5 of harmonic minor."""
    return chromatic(tonic)[[0, 1, 4, 5, 7, 8, 10]]


def lydian_sharp2(tonic: float | Pitch = 220.0) -> Scale:
    """Lydian #2. Mode 6 of harmonic minor."""
    return chromatic(tonic)[[0, 3, 4, 6, 7, 9, 11]]


def ultralocrian(tonic: float | Pitch = 220.0) -> Scale:
    """Ultra-locrian (altered diminished). Mode 7 of harmonic minor."""
    return chromatic(tonic)[[0, 1, 3, 4, 6, 8, 9]]


# ---------------------------------------------------------------------------
# Melodic minor modes
# ---------------------------------------------------------------------------

def melodic_minor(tonic: float | Pitch = 220.0) -> Scale:
    """Melodic minor (ascending form)."""
    return chromatic(tonic)[[0, 2, 3, 5, 7, 9, 11]]


def dorian_b2(tonic: float | Pitch = 220.0) -> Scale:
    """Dorian b2 (Phrygian natural 6). Mode 2 of melodic minor."""
    return chromatic(tonic)[[0, 1, 3, 5, 7, 9, 10]]


def lydian_augmented(tonic: float | Pitch = 220.0) -> Scale:
    """Lydian augmented. Mode 3 of melodic minor."""
    return chromatic(tonic)[[0, 2, 4, 6, 8, 9, 11]]


def lydian_dominant(tonic: float | Pitch = 220.0) -> Scale:
    """Lydian dominant. Mode 4 of melodic minor."""
    return chromatic(tonic)[[0, 2, 4, 6, 7, 9, 10]]


def mixolydian_b6(tonic: float | Pitch = 220.0) -> Scale:
    """Mixolydian b6 (Hindu scale). Mode 5 of melodic minor."""
    return chromatic(tonic)[[0, 2, 4, 5, 7, 8, 10]]


def aeolian_b5(tonic: float | Pitch = 220.0) -> Scale:
    """Aeolian b5 (Locrian natural 2). Mode 6 of melodic minor."""
    return chromatic(tonic)[[0, 2, 3, 5, 6, 8, 10]]


def altered(tonic: float | Pitch = 220.0) -> Scale:
    """Altered scale (Super Locrian). Mode 7 of melodic minor."""
    return chromatic(tonic)[[0, 1, 3, 4, 6, 8, 10]]


# ---------------------------------------------------------------------------
# Pentatonic & blues
# ---------------------------------------------------------------------------

def major_pentatonic(tonic: float | Pitch = 220.0) -> Scale:
    """Major pentatonic. Degrees 1, 2, 3, 5, 6 of major."""
    return chromatic(tonic)[[0, 2, 4, 7, 9]]


def minor_pentatonic(tonic: float | Pitch = 220.0) -> Scale:
    """Minor pentatonic. Degrees 1, b3, 4, 5, b7 of minor."""
    return chromatic(tonic)[[0, 3, 5, 7, 10]]


def blues(tonic: float | Pitch = 220.0) -> Scale:
    """Blues scale. Minor pentatonic + b5."""
    return chromatic(tonic)[[0, 3, 5, 6, 7, 10]]


def blues_major(tonic: float | Pitch = 220.0) -> Scale:
    """Major blues scale. Major pentatonic + b3."""
    return chromatic(tonic)[[0, 2, 3, 4, 7, 9]]


# ---------------------------------------------------------------------------
# Symmetric
# ---------------------------------------------------------------------------

def whole_tone(tonic: float | Pitch = 220.0) -> Scale:
    """Whole-tone scale. 6 equally spaced notes."""
    return chromatic(tonic)[::2]


def diminished_whole_half(tonic: float | Pitch = 220.0) -> Scale:
    """Diminished scale (whole-half). Alternating W-H steps."""
    return chromatic(tonic)[[0, 2, 3, 5, 6, 8, 9, 11]]


def diminished_half_whole(tonic: float | Pitch = 220.0) -> Scale:
    """Diminished scale (half-whole). Alternating H-W steps."""
    return chromatic(tonic)[[0, 1, 3, 4, 6, 7, 9, 10]]


# ---------------------------------------------------------------------------
# Bebop (8-note scales with chromatic passing tones)
# ---------------------------------------------------------------------------

def bebop_dominant(tonic: float | Pitch = 220.0) -> Scale:
    """Bebop dominant. Mixolydian + natural 7."""
    return chromatic(tonic)[[0, 2, 4, 5, 7, 9, 10, 11]]


def bebop_major(tonic: float | Pitch = 220.0) -> Scale:
    """Bebop major. Major + #5 passing tone."""
    return chromatic(tonic)[[0, 2, 4, 5, 7, 8, 9, 11]]


def bebop_dorian(tonic: float | Pitch = 220.0) -> Scale:
    """Bebop dorian. Dorian + natural 3 passing tone."""
    return chromatic(tonic)[[0, 2, 3, 4, 5, 7, 9, 10]]


def bebop_melodic_minor(tonic: float | Pitch = 220.0) -> Scale:
    """Bebop melodic minor. Melodic minor + natural 5 passing tone."""
    return chromatic(tonic)[[0, 2, 3, 5, 7, 8, 9, 11]]
