"""Scale presets: ready-to-use scales from many traditions.

Every export is a function ``(tonic=220.0) -> Scale``.

    from presets.scales import major, blues, maqam_rast
    from presets.pitches import C

    major()       # A major (default)
    major(C)      # C major
    blues(C)      # C blues
"""

from presets.scales.western import (
    chromatic,
    # diatonic modes
    major, dorian, phrygian, lydian, mixolydian, minor, locrian,
    # harmonic minor modes
    harmonic_minor, locrian_nat6, ionian_sharp5, dorian_sharp4,
    phrygian_dominant, lydian_sharp2, ultralocrian,
    # melodic minor modes
    melodic_minor, dorian_b2, lydian_augmented, lydian_dominant,
    mixolydian_b6, aeolian_b5, altered,
    # pentatonic & blues
    major_pentatonic, minor_pentatonic, blues, blues_major,
    # symmetric
    whole_tone, diminished_whole_half, diminished_half_whole,
    # bebop
    bebop_dominant, bebop_major, bebop_dorian, bebop_melodic_minor,
)
from presets.scales.just_intonation import (
    ji_major, ji_minor, ji_pentatonic, pythagorean,
)
from presets.scales.non_western import (
    # arabic maqamat
    maqam_rast, maqam_bayati, maqam_hijaz, maqam_saba, maqam_nahawand,
    # ottoman makams
    tet53, makam_rast, makam_ussak, makam_hicaz,
    makam_huseyni, makam_segah, makam_kurdilihicazkar, makam_nihavend,
    makam_karcigar,
    # indian ragas
    raga_bilawal, raga_kafi, raga_bhairavi, raga_yaman, raga_todi, raga_marwa,
    # gamelan
    slendro, pelog,
)
from presets.scales.microtonal import (
    bohlen_pierce, quarter_tone, edo19, edo31,
)
