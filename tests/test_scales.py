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


class TestScaleExtrapolation:
    def test_octave_extrapolation_up(self):
        s = Scale(440, [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3], period=2.0)
        assert s[5].hz == pytest.approx(880.0)  # degree 0, one period up
        assert s[6].hz == pytest.approx(440 * 9 / 8 * 2)  # degree 1, one period up

    def test_octave_extrapolation_two_periods(self):
        s = Scale(440, [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3], period=2.0)
        assert s[10].hz == pytest.approx(1760.0)  # degree 0, two periods up

    def test_negative_index(self):
        s = Scale(440, [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3], period=2.0)
        assert s[-5].hz == pytest.approx(220.0)  # degree 0, one period down
        assert s[-1].hz == pytest.approx(440 * 5 / 3 / 2)  # last degree, one period down

    def test_tritave_bohlen_pierce(self):
        s = Scale(440, [3 ** (n / 13) for n in range(13)], period=3.0)
        assert s[13].hz == pytest.approx(1320.0)  # one tritave up
        assert s[26].hz == pytest.approx(3960.0)  # two tritaves up

    def test_single_degree_scale(self):
        s = Scale(440, [1], period=2.0)
        assert s[0].hz == pytest.approx(440.0)
        assert s[3].hz == pytest.approx(3520.0)  # 440 * 2**3

    def test_transposition_preserves_period(self):
        s = Scale(440, [1, 5 / 4, 3 / 2], period=2.0)
        s2 = s * 2
        assert s2.period == 2.0
        assert s2[3].hz == pytest.approx(s[3].hz * 2)
        s3 = s / 2
        assert s3.period == 2.0

    def test_no_period_still_raises(self):
        s = Scale(440, [1, 2])
        with pytest.raises(IndexError):
            s[5]

    def test_invalid_period_zero(self):
        with pytest.raises(ValueError):
            Scale(440, [1], period=0.0)

    def test_invalid_period_negative(self):
        with pytest.raises(ValueError):
            Scale(440, [1], period=-1.0)

    def test_period_property(self):
        assert Scale(440, [1], period=2.0).period == 2.0
        assert Scale(440, [1]).period is None

    def test_repr_with_period(self):
        r = repr(Scale(440, [1, 2], period=2.0))
        assert "period=2.0" in r

    def test_iter_stays_within_base_degrees(self):
        s = Scale(440, [1, 5 / 4, 3 / 2], period=2.0)
        pitches = list(s)
        assert len(pitches) == 3


class TestSmartIndexing:
    eq_temp = Scale(440, [2 ** (n / 12) for n in range(12)])

    def test_list_index_major(self):
        major = self.eq_temp[[0, 2, 4, 5, 7, 9, 11]]
        assert len(major) == 7
        assert major[0].hz == pytest.approx(440.0)
        assert major[1].hz == pytest.approx(440 * 2 ** (2 / 12))

    def test_slice_whole_tone(self):
        whole_tone = self.eq_temp[::2]
        assert len(whole_tone) == 6
        assert whole_tone[0].hz == pytest.approx(440.0)
        assert whole_tone[1].hz == pytest.approx(440 * 2 ** (2 / 12))

    def test_slice_returns_scale(self):
        sub = self.eq_temp[:3]
        assert isinstance(sub, Scale)
        assert len(sub) == 3

    def test_list_returns_scale(self):
        sub = self.eq_temp[[0, 4, 7]]
        assert isinstance(sub, Scale)
        assert len(sub) == 3

    def test_int_still_returns_pitch(self):
        p = self.eq_temp[0]
        assert isinstance(p, Pitch)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty scale"):
            self.eq_temp[[]]

    def test_empty_slice_raises(self):
        with pytest.raises(ValueError, match="empty scale"):
            self.eq_temp[10:10]

    def test_list_index_out_of_range(self):
        with pytest.raises(IndexError):
            self.eq_temp[[0, 99]]

    def test_tonic_preserved(self):
        sub = self.eq_temp[[3, 7]]
        assert sub.tonic.hz == pytest.approx(440.0)


class TestScaleSetOps:
    eq_temp = Scale(440, [2 ** (n / 12) for n in range(12)])

    def test_union_diminished(self):
        dim = self.eq_temp[::3] | self.eq_temp[1::3]
        assert len(dim) == 8

    def test_union_dedup(self):
        s = self.eq_temp[:6]
        result = s | s
        assert len(result) == len(s)

    def test_union_tonic_is_lhs(self):
        a = Scale(440, [1, 5 / 4, 3 / 2])
        b = Scale(330, [1, 5 / 4, 3 / 2])
        result = a | b
        assert result.tonic.hz == pytest.approx(440.0)

    def test_intersection_same_tonic(self):
        major = self.eq_temp[[0, 2, 4, 5, 7, 9, 11]]
        penta = self.eq_temp[[0, 2, 4, 7, 9]]
        common = major & penta
        assert len(common) == 5

    def test_intersection_cross_tonic(self):
        a_major = Scale(440, [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8])
        # Scale rooted a fifth up, sharing some absolute Hz
        e_scale = Scale(660, [1, 4 / 3])  # 660 Hz and 880 Hz
        common = a_major & e_scale
        # 660 = 440 * 3/2 and 880 = 440 * 2 — but 2 not in a_major ratios
        assert any(pytest.approx(p.hz) == 660.0 for p in common)

    def test_intersection_empty_raises(self):
        a = Scale(440, [1])
        b = Scale(100, [1])
        with pytest.raises(ValueError, match="empty scale"):
            a & b

    def test_difference(self):
        chromatic = self.eq_temp
        major = self.eq_temp[[0, 2, 4, 5, 7, 9, 11]]
        accidentals = chromatic - major
        assert len(accidentals) == 5

    def test_difference_empty_raises(self):
        s = Scale(440, [1, 2])
        with pytest.raises(ValueError, match="empty scale"):
            s - s

    def test_union_sorted(self):
        a = self.eq_temp[[7, 9, 11]]
        b = self.eq_temp[[0, 2, 4]]
        result = a | b
        hz_values = [result[i].hz for i in range(len(result))]
        assert hz_values == sorted(hz_values)

    def test_set_ops_no_period(self):
        a = Scale(440, [1, 5 / 4, 3 / 2], period=2.0)
        b = Scale(440, [1, 3 / 2], period=2.0)
        assert (a | b).period is None
        assert (a & b).period is None
        assert (a - b).period is None


class TestModes:
    # 12-TET major scale with period
    major = Scale(440, [2 ** (n / 12) for n in [0, 2, 4, 5, 7, 9, 11]], period=2.0)

    def test_ionian_is_identity(self):
        ionian = self.major.mode(0)
        assert ionian.tonic.hz == pytest.approx(440.0)
        for i in range(len(self.major)):
            assert ionian[i].hz == pytest.approx(self.major[i].hz)

    def test_dorian_tonic(self):
        dorian = self.major.mode(1)
        assert dorian.tonic.hz == pytest.approx(self.major[1].hz)

    def test_dorian_degree_count(self):
        dorian = self.major.mode(1)
        assert len(dorian) == len(self.major)

    def test_dorian_ratios_start_at_one(self):
        dorian = self.major.mode(1)
        assert dorian[0].hz == pytest.approx(dorian.tonic.hz)

    def test_dorian_spans_one_period(self):
        dorian = self.major.mode(1)
        assert dorian[len(dorian) - 1].hz < dorian.tonic.hz * 2.0
        # Next period wraps correctly
        assert dorian[len(dorian)].hz == pytest.approx(dorian.tonic.hz * 2.0)

    def test_all_seven_modes_share_pitch_classes(self):
        """All modes of the same scale contain the same pitch classes mod period."""
        def pitch_classes(scale):
            return sorted(scale[i].hz / scale.tonic.hz for i in range(len(scale)))

        base_pcs = pitch_classes(self.major)
        for m in range(7):
            mode = self.major.mode(m)
            mode_pcs = pitch_classes(mode)
            # Fold into same period and compare as sets
            base_abs = sorted(self.major.tonic.hz * r for r in base_pcs)
            mode_abs = sorted(mode.tonic.hz * r for r in mode_pcs)
            # Normalize all into the base octave [tonic, tonic*period)
            def fold(hz, tonic, period):
                while hz >= tonic * period - 1e-9:
                    hz /= period
                while hz < tonic - 1e-9:
                    hz *= period
                return hz
            base_folded = sorted(fold(h, self.major.tonic.hz, 2.0) for h in base_abs)
            mode_folded = sorted(fold(h, self.major.tonic.hz, 2.0) for h in mode_abs)
            assert len(base_folded) == len(mode_folded)
            for a, b in zip(base_folded, mode_folded):
                assert a == pytest.approx(b)

    def test_mode_preserves_period(self):
        dorian = self.major.mode(1)
        assert dorian.period == 2.0

    def test_mode_extrapolation(self):
        dorian = self.major.mode(1)
        assert dorian[7].hz == pytest.approx(dorian[0].hz * 2.0)
        assert dorian[14].hz == pytest.approx(dorian[0].hz * 4.0)

    def test_negative_degree_wraps(self):
        # mode(-1) == mode(6) for a 7-degree scale
        m1 = self.major.mode(-1)
        m2 = self.major.mode(6)
        for i in range(7):
            assert m1[i].hz == pytest.approx(m2[i].hz)

    def test_no_period_raises(self):
        s = Scale(440, [1, 5 / 4, 3 / 2])
        with pytest.raises(ValueError, match="period"):
            s.mode(1)

    def test_just_intonation_modes(self):
        ji = Scale(440, [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8], period=2.0)
        dorian = ji.mode(1)
        assert dorian.tonic.hz == pytest.approx(440 * 9 / 8)
        assert dorian[0].hz == pytest.approx(440 * 9 / 8)
        # Second degree of dorian = third degree of parent / dorian tonic
        assert dorian[1].hz == pytest.approx(440 * 5 / 4)
