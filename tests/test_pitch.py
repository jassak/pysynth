import pytest
from pysynth.music.pitch import Pitch, Note


# ------------------------------------------------------------------ #
# Pitch                                                                #
# ------------------------------------------------------------------ #


class TestPitchAlgebra:
    def test_mul_ratio(self):
        p = Pitch(440.0)
        assert (p * 2).hz == pytest.approx(880.0)

    def test_rmul_ratio(self):
        p = Pitch(440.0)
        assert (2 * p).hz == pytest.approx(880.0)

    def test_div_ratio(self):
        p = Pitch(440.0) / 2
        assert p.hz == pytest.approx(220.0)

    def test_div_pitch_gives_interval(self):
        ratio = Pitch(660.0) / Pitch(440.0)
        assert ratio == pytest.approx(1.5)

    def test_up_cents_octave(self):
        p = Pitch(440.0).up(1200)
        assert p.hz == pytest.approx(880.0)

    def test_up_cents_semitone(self):
        p = Pitch(440.0).up(100)
        assert p.hz == pytest.approx(440.0 * 2 ** (1 / 12))

    def test_down_cents(self):
        p = Pitch(880.0).down(1200)
        assert p.hz == pytest.approx(440.0)

    def test_up_down_roundtrip(self):
        p = Pitch(440.0)
        assert p.up(700).down(700).hz == pytest.approx(440.0)

    def test_identity(self):
        p = Pitch(440.0) * 1
        assert p.hz == pytest.approx(440.0)

    def test_inverse(self):
        p = Pitch(440.0)
        assert (p * 3 / 3).hz == pytest.approx(440.0)

    def test_repr(self):
        assert "440" in repr(Pitch(440.0))


# ------------------------------------------------------------------ #
# Note                                                                 #
# ------------------------------------------------------------------ #


class TestNote:
    def test_basic_construction(self):
        n = Note(Pitch(440), 1.0, 0.8)
        assert n.pitch.hz == pytest.approx(440.0)
        assert n.duration == 1.0
        assert n.velocity == 0.8

    def test_default_velocity(self):
        n = Note(Pitch(440), 1.0)
        assert n.velocity == 1.0

    def test_pitch_coercion_from_float(self):
        n = Note(440.0, 1.0)
        assert isinstance(n.pitch, Pitch)
        assert n.pitch.hz == pytest.approx(440.0)

    def test_pitch_coercion_from_int(self):
        n = Note(440, 1.0)
        assert isinstance(n.pitch, Pitch)

    def test_mul_transposes_pitch(self):
        n = Note(Pitch(440), 1.0, 0.8) * 2
        assert n.pitch.hz == pytest.approx(880.0)
        assert n.duration == 1.0
        assert n.velocity == 0.8

    def test_rmul(self):
        n = 2 * Note(Pitch(440), 1.0)
        assert n.pitch.hz == pytest.approx(880.0)

    def test_div_ratio(self):
        n = Note(Pitch(880), 1.0) / 2
        assert n.pitch.hz == pytest.approx(440.0)
        assert n.duration == 1.0

    def test_div_note_gives_interval(self):
        a = Note(Pitch(660), 1.0)
        b = Note(Pitch(440), 1.0)
        assert a / b == pytest.approx(1.5)

    def test_rest(self):
        r = Note.rest(2.0)
        assert r.is_rest
        assert r.duration == 2.0
        assert r.velocity == 0.0

    def test_is_rest_false_for_pitched(self):
        assert not Note(Pitch(440), 1.0).is_rest

    def test_repr_pitched(self):
        r = repr(Note(Pitch(440), 1.0, 0.8))
        assert "440" in r
        assert "vel=0.8" in r

    def test_repr_rest(self):
        assert "rest" in repr(Note.rest(1.0))

    def test_frozen(self):
        n = Note(Pitch(440), 1.0)
        with pytest.raises(AttributeError):
            n.duration = 2.0
