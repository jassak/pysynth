"""Non-western scales: Arabic maqamat, Ottoman makams, Indian ragas, Gamelan."""

from __future__ import annotations

from pysynth import Pitch, Scale


def _hz(tonic: float | Pitch) -> float:
    return tonic.hz if isinstance(tonic, Pitch) else float(tonic)


# ---------------------------------------------------------------------------
# Arabic Maqamat (24-TET quarter-tone system)
# Intervals given in cents for clarity.
# ---------------------------------------------------------------------------

def maqam_rast(tonic: float | Pitch = 220.0) -> Scale:
    """Maqam Rast. The foundational Arabic maqam."""
    return Scale(_hz(tonic), [0, 200, 350, 500, 700, 900, 1050], unit="cents", period=2)


def maqam_bayati(tonic: float | Pitch = 220.0) -> Scale:
    """Maqam Bayati. Common in Egyptian music."""
    return Scale(_hz(tonic), [0, 150, 300, 500, 700, 850, 1000], unit="cents", period=2)


def maqam_hijaz(tonic: float | Pitch = 220.0) -> Scale:
    """Maqam Hijaz. Distinctive augmented-second character."""
    return Scale(_hz(tonic), [0, 100, 400, 500, 700, 850, 1000], unit="cents", period=2)


def maqam_saba(tonic: float | Pitch = 220.0) -> Scale:
    """Maqam Saba. Melancholic, used in taqasim improvisations."""
    return Scale(_hz(tonic), [0, 150, 300, 450, 700, 800, 1000], unit="cents", period=2)


def maqam_nahawand(tonic: float | Pitch = 220.0) -> Scale:
    """Maqam Nahawand. Similar to Western harmonic minor."""
    return Scale(_hz(tonic), [0, 200, 300, 500, 700, 800, 1100], unit="cents", period=2)


# ---------------------------------------------------------------------------
# Ottoman Makams (53-TET Holderian comma system)
# 1 comma ≈ 22.64 cents. Intervals specified as comma counts.
# ---------------------------------------------------------------------------

def tet53(tonic: float | Pitch = 220.0) -> Scale:
    """Full 53-TET chromatic scale. Building block for Ottoman makams."""
    return Scale(_hz(tonic), [2 ** (n / 53) for n in range(53)], period=2)


def makam_rast(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Rast. 9-8-5-9-9-8-5 commas."""
    return tet53(tonic)[[0, 9, 17, 22, 31, 40, 48]]


def makam_ussak(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Ussak. 8-9-5-9-9-8-5 commas."""
    return tet53(tonic)[[0, 8, 17, 22, 31, 40, 48]]


def makam_hicaz(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Hicaz. 5-12-5-9-9-8-5 commas."""
    return tet53(tonic)[[0, 5, 17, 22, 31, 40, 48]]


def makam_huseyni(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Huseyni. 8-9-5-9-8-9-5 commas."""
    return tet53(tonic)[[0, 8, 17, 22, 31, 39, 48]]


def makam_segah(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Segah. 5-8-9-9-4-9-9 commas."""
    return tet53(tonic)[[0, 5, 13, 22, 31, 35, 44]]


def makam_kurdilihicazkar(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Kurdilihicazkar. 9-4-9-9-4-13-5 commas."""
    return tet53(tonic)[[0, 9, 13, 22, 31, 35, 48]]


def makam_nihavend(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Nihavend. 9-4-9-9-4-9-9 commas."""
    return tet53(tonic)[[0, 9, 13, 22, 31, 35, 44]]


def makam_karcigar(tonic: float | Pitch = 220.0) -> Scale:
    """Ottoman Makam Karcigar. 8-5-9-9-8-9-5 commas."""
    return tet53(tonic)[[0, 8, 13, 22, 31, 39, 48]]


# ---------------------------------------------------------------------------
# Indian Ragas (12-TET approximation via cents)
# ---------------------------------------------------------------------------

def raga_bilawal(tonic: float | Pitch = 220.0) -> Scale:
    """Raga Bilawal (Shankarabharanam). Equivalent to Western major."""
    return Scale(_hz(tonic), [0, 200, 400, 500, 700, 900, 1100], unit="cents", period=2)


def raga_kafi(tonic: float | Pitch = 220.0) -> Scale:
    """Raga Kafi. Similar to Dorian mode."""
    return Scale(_hz(tonic), [0, 200, 300, 500, 700, 900, 1000], unit="cents", period=2)


def raga_bhairavi(tonic: float | Pitch = 220.0) -> Scale:
    """Raga Bhairavi. Similar to Phrygian mode."""
    return Scale(_hz(tonic), [0, 100, 300, 500, 700, 800, 1000], unit="cents", period=2)


def raga_yaman(tonic: float | Pitch = 220.0) -> Scale:
    """Raga Yaman (Kalyan). Similar to Lydian mode."""
    return Scale(_hz(tonic), [0, 200, 400, 600, 700, 900, 1100], unit="cents", period=2)


def raga_todi(tonic: float | Pitch = 220.0) -> Scale:
    """Raga Todi. Distinctive with b2, b3, #4, b6."""
    return Scale(_hz(tonic), [0, 100, 300, 600, 700, 800, 1100], unit="cents", period=2)


def raga_marwa(tonic: float | Pitch = 220.0) -> Scale:
    """Raga Marwa. b2, #4, no 5th."""
    return Scale(_hz(tonic), [0, 100, 400, 600, 900, 1100], unit="cents", period=2)


# ---------------------------------------------------------------------------
# Gamelan (approximate tunings — real gamelans vary by ensemble)
# ---------------------------------------------------------------------------

def slendro(tonic: float | Pitch = 220.0) -> Scale:
    """Javanese Slendro. 5-tone near-equidistant scale."""
    return Scale(_hz(tonic), [1, 1.125, 1.260, 1.500, 1.680], period=2)


def pelog(tonic: float | Pitch = 220.0) -> Scale:
    """Javanese Pelog. 7-tone scale with unequal steps."""
    return Scale(_hz(tonic), [1, 1.067, 1.200, 1.333, 1.500, 1.600, 1.778], period=2)
