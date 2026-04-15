"""Tests for the presets.scales module."""

import math

from pysynth import Pitch, Scale

from presets.pitches import A, C, D, E, G
from presets.scales import (
    # western
    chromatic, major, dorian, phrygian, lydian, mixolydian, minor, locrian,
    harmonic_minor, phrygian_dominant,
    melodic_minor, altered,
    major_pentatonic, minor_pentatonic, blues,
    whole_tone, diminished_whole_half, diminished_half_whole,
    bebop_dominant,
    # just intonation
    ji_major, pythagorean,
    # non-western
    maqam_rast, makam_rast, slendro, pelog,
    # microtonal
    bohlen_pierce, quarter_tone, edo19,
)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-4)


class TestScaleBasics:
    def test_returns_scale(self):
        assert isinstance(major(), Scale)

    def test_default_tonic_is_a220(self):
        s = major()
        assert _close(s[0].hz, 220.0)

    def test_tonic_with_pitch(self):
        s = major(C)
        assert _close(s[0].hz, C.hz)

    def test_tonic_with_float(self):
        s = major(440.0)
        assert _close(s[0].hz, 440.0)


class TestWesternScaleDegrees:
    def test_chromatic_has_12_degrees(self):
        assert len(chromatic()) == 12

    def test_major_has_7_degrees(self):
        assert len(major()) == 7

    def test_major_intervals(self):
        """Major scale should follow W-W-H-W-W-W-H pattern."""
        s = major()
        ratios = [s[i].hz / s[0].hz for i in range(7)]
        expected = [2 ** (n / 12) for n in [0, 2, 4, 5, 7, 9, 11]]
        for r, e in zip(ratios, expected):
            assert _close(r, e)

    def test_minor_intervals(self):
        """Natural minor: W-H-W-W-H-W-W."""
        s = minor()
        ratios = [s[i].hz / s[0].hz for i in range(7)]
        expected = [2 ** (n / 12) for n in [0, 2, 3, 5, 7, 8, 10]]
        for r, e in zip(ratios, expected):
            assert _close(r, e)

    def test_all_modes_have_7_degrees(self):
        for fn in [major, dorian, phrygian, lydian, mixolydian, minor, locrian]:
            assert len(fn()) == 7, fn.__name__


class TestHarmonicMinorModes:
    def test_harmonic_minor_has_7_degrees(self):
        assert len(harmonic_minor()) == 7

    def test_phrygian_dominant_intervals(self):
        """Mode 5 of harmonic minor: H-A2-H-W-H-W-W."""
        s = phrygian_dominant()
        semitones = [0, 1, 4, 5, 7, 8, 10]
        expected = [2 ** (n / 12) for n in semitones]
        for i, e in enumerate(expected):
            assert _close(s[i].hz / s[0].hz, e)


class TestMelodicMinorModes:
    def test_melodic_minor_has_7_degrees(self):
        assert len(melodic_minor()) == 7

    def test_altered_intervals(self):
        """Super Locrian: H-W-H-W-W-W-W."""
        s = altered()
        semitones = [0, 1, 3, 4, 6, 8, 10]
        expected = [2 ** (n / 12) for n in semitones]
        for i, e in enumerate(expected):
            assert _close(s[i].hz / s[0].hz, e)


class TestPentatonicAndBlues:
    def test_major_pentatonic_has_5_degrees(self):
        assert len(major_pentatonic()) == 5

    def test_blues_has_6_degrees(self):
        assert len(blues()) == 6


class TestSymmetric:
    def test_whole_tone_has_6_degrees(self):
        assert len(whole_tone()) == 6

    def test_diminished_whole_half_has_8_degrees(self):
        assert len(diminished_whole_half()) == 8

    def test_diminished_half_whole_has_8_degrees(self):
        assert len(diminished_half_whole()) == 8


class TestBebop:
    def test_bebop_dominant_has_8_degrees(self):
        assert len(bebop_dominant()) == 8


class TestTransposition:
    def test_c_major_tonic(self):
        s = major(C)
        assert _close(s[0].hz, C.hz)

    def test_d_major_tonic(self):
        s = major(D)
        assert _close(s[0].hz, D.hz)

    def test_intervals_preserved_after_transposition(self):
        """Same intervals regardless of tonic."""
        a = major(A)
        c = major(C)
        for i in range(7):
            ratio_a = a[i].hz / a[0].hz
            ratio_c = c[i].hz / c[0].hz
            assert _close(ratio_a, ratio_c)


class TestJustIntonation:
    def test_ji_major_fifth(self):
        """JI perfect fifth should be exactly 3/2."""
        s = ji_major()
        assert _close(s[4].hz / s[0].hz, 3 / 2)

    def test_pythagorean_fifth(self):
        s = pythagorean()
        assert _close(s[4].hz / s[0].hz, 3 / 2)


class TestNonWestern:
    def test_maqam_rast_has_7_degrees(self):
        assert len(maqam_rast()) == 7

    def test_makam_rast_has_7_degrees(self):
        assert len(makam_rast()) == 7

    def test_slendro_has_5_degrees(self):
        assert len(slendro()) == 5

    def test_pelog_has_7_degrees(self):
        assert len(pelog()) == 7


class TestMicrotonal:
    def test_bohlen_pierce_has_13_degrees(self):
        s = bohlen_pierce()
        assert len(s) == 13

    def test_bohlen_pierce_period_is_3(self):
        assert bohlen_pierce().period == 3

    def test_quarter_tone_has_24_degrees(self):
        assert len(quarter_tone()) == 24

    def test_edo19_has_19_degrees(self):
        assert len(edo19()) == 19


class TestScalePeriod:
    def test_western_scales_have_octave_period(self):
        for fn in [major, minor, chromatic, blues]:
            assert fn().period == 2, fn.__name__

    def test_wrapping_works(self):
        """Degree 7 of a 7-note scale should be one octave above degree 0."""
        s = major()
        assert _close(s[7].hz, s[0].hz * 2)

    def test_mode_works_on_preset(self):
        """Preset scales support .mode() since they have period."""
        s = major()
        d = s.mode(1)  # dorian from the same major scale
        assert len(d) == 7
