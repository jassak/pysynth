import pytest
from pysynth.music.pitch import Pitch
from pysynth.music.scales import Scale


class TestScaleBasics:
    def test_degree_zero_is_tonic(self):
        s = Scale(440, [1, 5 / 4, 3 / 2])
        assert s[0].hz == pytest.approx(440.0)

    def test_degree_values(self):
        s = Scale(440, [1, 5 / 4, 3 / 2, 2])
        assert s[1].hz == pytest.approx(550.0)
        assert s[2].hz == pytest.approx(660.0)
        assert s[3].hz == pytest.approx(880.0)

    def test_len(self):
        s = Scale(440, [1, 5 / 4, 3 / 2])
        assert len(s) == 3

    def test_tonic_property(self):
        s = Scale(440, [1])
        assert isinstance(s.tonic, Pitch)
        assert s.tonic.hz == pytest.approx(440.0)

    def test_tonic_from_pitch(self):
        s = Scale(Pitch(440), [1, 2])
        assert s[0].hz == pytest.approx(440.0)

    def test_iter(self):
        s = Scale(100, [1, 2, 3])
        pitches = list(s)
        assert len(pitches) == 3
        assert pitches[0].hz == pytest.approx(100.0)
        assert pitches[1].hz == pytest.approx(200.0)
        assert pitches[2].hz == pytest.approx(300.0)

    def test_index_out_of_range(self):
        s = Scale(440, [1, 2])
        with pytest.raises(IndexError):
            s[5]


class TestScaleCents:
    def test_cents_unit(self):
        s = Scale(440, [0, 1200], unit="cents")
        assert s[0].hz == pytest.approx(440.0)
        assert s[1].hz == pytest.approx(880.0)

    def test_cents_semitone(self):
        s = Scale(440, [0, 100], unit="cents")
        assert s[1].hz == pytest.approx(440.0 * 2 ** (1 / 12))

    def test_equal_temperament_12(self):
        s = Scale(440, [n * 100 for n in range(13)], unit="cents")
        assert len(s) == 13
        assert s[12].hz == pytest.approx(880.0, rel=1e-6)


class TestScaleTransposition:
    def test_mul_transposes_tonic(self):
        s = Scale(440, [1, 2]) * 2
        assert s[0].hz == pytest.approx(880.0)
        assert s[1].hz == pytest.approx(1760.0)

    def test_div_transposes_down(self):
        s = Scale(440, [1, 2]) / 2
        assert s[0].hz == pytest.approx(220.0)

    def test_transposition_preserves_intervals(self):
        s1 = Scale(440, [1, 5 / 4, 3 / 2])
        s2 = s1 * 2
        # Intervals should be the same
        for i in range(len(s1)):
            assert (s2[i].hz / s2[0].hz) == pytest.approx(s1[i].hz / s1[0].hz)

    def test_repr(self):
        r = repr(Scale(440, [1, 2, 3]))
        assert "440" in r
        assert "3" in r
