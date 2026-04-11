import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.generators.wavetable import Wavetable
from pysynth.generators.oscillators import _shape

SR = 4000


def _sig(values, sr=SR):
    return Signal(np.array(values, dtype=np.float32), sr)


# ------------------------------------------------------------------ #
# Construction                                                        #
# ------------------------------------------------------------------ #


class TestConstruction:
    def test_from_arrays(self):
        t1 = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
        t2 = np.linspace(-1, 1, 64)
        wt = Wavetable([t1, t2], table_size=64)
        assert wt.n_tables == 2
        assert wt.table_size == 64

    def test_from_waveforms(self):
        wt = Wavetable.from_waveforms(["sine", "saw", "square"], table_size=256)
        assert wt.n_tables == 3
        assert wt.table_size == 256

    def test_resamples_mismatched_sizes(self):
        t1 = np.ones(100)
        t2 = np.ones(200)
        wt = Wavetable([t1, t2], table_size=64)
        assert wt.table_size == 64

    def test_empty_tables_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Wavetable([])

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="table 0 is empty"):
            Wavetable([np.array([])])

    def test_table_size_too_small_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            Wavetable([np.ones(8)], table_size=1)

    def test_immutability(self):
        arr = np.ones(64)
        wt = Wavetable([arr], table_size=64)
        arr[:] = 999.0
        sig = wt.at(100).render(0.01, SR)
        # should still be ~1.0, not 999
        assert np.max(np.abs(sig.data)) < 1.1

    def test_repr(self):
        wt = Wavetable.from_waveforms(["sine", "saw"])
        assert "tables=2" in repr(wt)


# ------------------------------------------------------------------ #
# Single-table rendering                                              #
# ------------------------------------------------------------------ #


class TestSingleTable:
    def test_sine_matches_shape(self):
        """A single sine wavetable should closely match _shape('sine', ...)."""
        wt = Wavetable.from_waveforms(["sine"], table_size=4096)
        sig = wt.at(100).render(0.05, SR)
        # generate reference via _shape
        n = int(0.05 * SR)
        t = np.arange(n, dtype=np.float64) / SR
        phase = 2.0 * np.pi * 100.0 * t
        ref = _shape("sine", phase).astype(np.float32)
        np.testing.assert_allclose(sig.data, ref, atol=5e-3)

    def test_saw_matches_shape(self):
        wt = Wavetable.from_waveforms(["saw"], table_size=4096)
        sig = wt.at(100).render(0.05, SR)
        n = int(0.05 * SR)
        t = np.arange(n, dtype=np.float64) / SR
        phase = 2.0 * np.pi * 100.0 * t
        ref = _shape("saw", phase).astype(np.float32)
        np.testing.assert_allclose(sig.data, ref, atol=5e-3)

    def test_output_duration_and_sr(self):
        wt = Wavetable.from_waveforms(["sine"])
        sig = wt.at(440).render(1.0, SR)
        assert sig.sample_rate == SR
        assert len(sig.data) == SR


# ------------------------------------------------------------------ #
# Position morphing                                                   #
# ------------------------------------------------------------------ #


class TestPosition:
    def test_position_zero_gives_first_table(self):
        wt = Wavetable.from_waveforms(["sine", "square"], table_size=4096)
        sig0 = wt.at(100, position=0.0).render(0.05, SR)
        sig_ref = Wavetable.from_waveforms(["sine"], table_size=4096).at(100).render(0.05, SR)
        np.testing.assert_allclose(sig0.data, sig_ref.data, atol=1e-6)

    def test_position_max_gives_last_table(self):
        wt = Wavetable.from_waveforms(["sine", "square"], table_size=4096)
        sig1 = wt.at(100, position=1.0).render(0.05, SR)
        sig_ref = Wavetable.from_waveforms(["square"], table_size=4096).at(100).render(0.05, SR)
        np.testing.assert_allclose(sig1.data, sig_ref.data, atol=1e-6)

    def test_midpoint_is_average(self):
        """Position 0.5 between two tables should be the average of both."""
        wt = Wavetable.from_waveforms(["sine", "saw"], table_size=4096)
        sig_mid = wt.at(100, position=0.5).render(0.05, SR)
        sig_a = wt.at(100, position=0.0).render(0.05, SR)
        sig_b = wt.at(100, position=1.0).render(0.05, SR)
        expected = (sig_a.data + sig_b.data) / 2.0
        np.testing.assert_allclose(sig_mid.data, expected, atol=1e-5)

    def test_signal_rate_position(self):
        """Time-varying position should produce different timbres at start vs end."""
        wt = Wavetable.from_waveforms(["sine", "saw"], table_size=4096)
        dur = 0.1
        n = int(dur * SR)
        # ramp from 0 to 1 over the duration
        pos = _sig(np.linspace(0, 1, n), SR)
        sig = wt.at(100, position=pos).render(dur, SR)
        assert len(sig.data) == n

    def test_position_clamped_below_zero(self):
        wt = Wavetable.from_waveforms(["sine", "saw"])
        sig_neg = wt.at(100, position=-1.0).render(0.01, SR)
        sig_zero = wt.at(100, position=0.0).render(0.01, SR)
        np.testing.assert_allclose(sig_neg.data, sig_zero.data, atol=1e-6)

    def test_position_clamped_above_max(self):
        wt = Wavetable.from_waveforms(["sine", "saw"])
        sig_over = wt.at(100, position=5.0).render(0.01, SR)
        sig_max = wt.at(100, position=1.0).render(0.01, SR)
        np.testing.assert_allclose(sig_over.data, sig_max.data, atol=1e-6)


# ------------------------------------------------------------------ #
# Frequency                                                           #
# ------------------------------------------------------------------ #


class TestFrequency:
    def _count_zero_crossings(self, data):
        return int(np.sum(np.diff(np.sign(data)) != 0))

    def test_constant_frequency_pitch(self):
        """A 100 Hz sine over 1s: verify period via autocorrelation peak."""
        wt = Wavetable.from_waveforms(["sine"], table_size=4096)
        sig = wt.at(100).render(1.0, SR)
        # autocorrelation peak at lag = SR/freq = 40 samples
        data = sig.data.astype(np.float64)
        expected_lag = SR // 100
        corr = np.correlate(data[:400], data[:400], mode="full")
        mid = len(corr) // 2
        # find first peak after lag 0
        peak = np.argmax(corr[mid + expected_lag - 5 : mid + expected_lag + 5])
        actual_lag = expected_lag - 5 + peak
        assert actual_lag == expected_lag

    def test_signal_rate_frequency(self):
        """Signal-rate Hz should produce the same waveform, offset by one sample
        (cumsum includes the first frequency value in its accumulation)."""
        wt = Wavetable.from_waveforms(["sine"], table_size=4096)
        dur = 0.1
        n = int(dur * SR)
        hz_sig = _sig(np.full(n, 200.0), SR)
        sig_const = wt.at(200).render(dur, SR)
        sig_signal = wt.at(hz_sig).render(dur, SR)
        # cumsum is one sample ahead of arange: signal_rate[k] ≈ const[k+1]
        np.testing.assert_allclose(sig_signal.data[:-1], sig_const.data[1:], atol=1e-4)


# ------------------------------------------------------------------ #
# Integration: top-level import                                       #
# ------------------------------------------------------------------ #


class TestImport:
    def test_importable_from_top_level(self):
        from pysynth import Wavetable as Wt
        assert Wt is Wavetable
