import numpy as np
import pytest
import tempfile
from pathlib import Path
from scipy.io import wavfile

from pysynth._core import Signal
from pysynth.generators.sample import Sample

SR = 4000


def _sig(values, sr=SR):
    return Signal(np.array(values, dtype=np.float32), sr)


def _sine_data(freq, dur, sr=SR):
    """Generate a pure sine wave as float32 array."""
    t = np.arange(int(dur * sr), dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


# ------------------------------------------------------------------ #
# Construction                                                        #
# ------------------------------------------------------------------ #


class TestConstruction:
    def test_direct_mono(self):
        data = np.ones(100, dtype=np.float32)
        s = Sample(data, SR)
        assert s.n_channels == 1
        assert len(s) == 100

    def test_direct_stereo(self):
        data = np.ones((100, 2), dtype=np.float32)
        s = Sample(data, SR)
        assert s.n_channels == 2
        assert len(s) == 100

    def test_dtype_coercion(self):
        data = np.ones(100, dtype=np.float64)
        s = Sample(data, SR)
        assert s.data.dtype == np.float32

    def test_immutability(self):
        data = np.ones(100, dtype=np.float32)
        s = Sample(data, SR)
        data[:] = 999.0
        assert np.allclose(s.data, 1.0)

    def test_from_signal(self):
        sig = _sig(np.linspace(-1, 1, 200))
        s = Sample.from_signal(sig, root_pitch=440.0)
        assert s.sample_rate == SR
        assert s.root_pitch == 440.0
        np.testing.assert_array_equal(s.data, sig.data)

    def test_from_signal_no_alias(self):
        sig = _sig(np.ones(100))
        s = Sample.from_signal(sig)
        sig.data[:] = 0.0
        assert np.allclose(s.data, 1.0)

    def test_from_file_int16(self, tmp_path):
        data_int16 = (np.sin(np.linspace(0, 2 * np.pi, 100)) * 16000).astype(np.int16)
        path = tmp_path / "test.wav"
        wavfile.write(path, SR, data_int16)
        s = Sample.from_file(path, root_pitch=220.0)
        assert s.sample_rate == SR
        assert s.root_pitch == 220.0
        assert s.data.dtype == np.float32
        assert np.max(np.abs(s.data)) <= 1.0

    def test_from_file_float32(self, tmp_path):
        data_f32 = np.sin(np.linspace(0, 2 * np.pi, 100)).astype(np.float32)
        path = tmp_path / "test.wav"
        wavfile.write(path, SR, data_f32)
        s = Sample.from_file(path)
        assert s.data.dtype == np.float32
        np.testing.assert_allclose(s.data, data_f32, atol=1e-6)

    def test_from_file_unsupported_format(self, tmp_path):
        path = tmp_path / "test.mp3"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            Sample.from_file(path)

    def test_repr(self):
        s = Sample(np.zeros(SR), SR, root_pitch=440.0)
        r = repr(s)
        assert "1.000s" in r
        assert "mono" in r
        assert "root=440.0Hz" in r


# ------------------------------------------------------------------ #
# Properties                                                          #
# ------------------------------------------------------------------ #


class TestProperties:
    def test_duration(self):
        s = Sample(np.zeros(SR * 2), SR)
        assert s.duration == pytest.approx(2.0)

    def test_n_channels_mono(self):
        assert Sample(np.zeros(100), SR).n_channels == 1

    def test_n_channels_stereo(self):
        assert Sample(np.zeros((100, 2)), SR).n_channels == 2

    def test_len(self):
        assert len(Sample(np.zeros(123), SR)) == 123


# ------------------------------------------------------------------ #
# Slicing                                                             #
# ------------------------------------------------------------------ #


class TestSlicing:
    def test_basic_slice(self):
        s = Sample(np.arange(SR, dtype=np.float32), SR)  # 1 second
        sliced = s[0.25:0.75]
        assert len(sliced) == int(0.5 * SR)
        assert sliced.duration == pytest.approx(0.5)

    def test_none_start(self):
        s = Sample(np.arange(SR, dtype=np.float32), SR)
        sliced = s[:0.5]
        assert len(sliced) == int(0.5 * SR)

    def test_none_end(self):
        s = Sample(np.arange(SR, dtype=np.float32), SR)
        sliced = s[0.5:]
        assert len(sliced) == int(0.5 * SR)

    def test_preserves_root_pitch(self):
        s = Sample(np.zeros(SR), SR, root_pitch=440.0)
        assert s[0.0:0.5].root_pitch == 440.0

    def test_out_of_bounds_clamped(self):
        s = Sample(np.zeros(SR), SR)
        sliced = s[-1.0:5.0]
        assert len(sliced) == SR

    def test_non_slice_raises(self):
        s = Sample(np.zeros(SR), SR)
        with pytest.raises(TypeError):
            s[42]

    def test_step_raises(self):
        s = Sample(np.zeros(SR), SR)
        with pytest.raises(ValueError, match="Step"):
            s[0.0:1.0:0.1]


# ------------------------------------------------------------------ #
# Normalize                                                           #
# ------------------------------------------------------------------ #


class TestNormalize:
    def test_peak_normalize(self):
        data = np.array([0.0, 0.25, -0.5, 0.1], dtype=np.float32)
        s = Sample(data, SR).normalize()
        assert np.max(np.abs(s.data)) == pytest.approx(1.0)

    def test_already_normalized(self):
        data = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        s = Sample(data, SR).normalize()
        np.testing.assert_allclose(s.data, data, atol=1e-7)

    def test_zero_signal(self):
        s = Sample(np.zeros(100), SR).normalize()
        assert np.all(s.data == 0.0)

    def test_preserves_metadata(self):
        s = Sample(np.array([0.5, -0.5]), SR, root_pitch=220.0,
                   loop_start=0, loop_end=1)
        n = s.normalize()
        assert n.root_pitch == 220.0
        assert n.loop_start == 0
        assert n.loop_end == 1

    def test_unsupported_mode(self):
        with pytest.raises(ValueError, match="Unsupported"):
            Sample(np.zeros(10), SR).normalize(mode="rms")


# ------------------------------------------------------------------ #
# as_signal                                                           #
# ------------------------------------------------------------------ #


class TestAsSignal:
    def test_returns_signal(self):
        s = Sample(np.ones(100), SR)
        sig = s.as_signal()
        assert isinstance(sig, Signal)
        assert sig.sample_rate == SR
        np.testing.assert_array_equal(sig.data, s.data)

    def test_no_alias(self):
        s = Sample(np.ones(100), SR)
        sig = s.as_signal()
        sig.data[:] = 0.0
        assert np.allclose(s.data, 1.0)


# ------------------------------------------------------------------ #
# Rendering                                                           #
# ------------------------------------------------------------------ #


class TestRender:
    def test_render_returns_signal(self):
        s = Sample(_sine_data(100, 1.0), SR, root_pitch=100.0)
        sig = s.render(0.5, sr=SR)
        assert isinstance(sig, Signal)

    def test_original_rate_playback(self):
        data = _sine_data(100, 0.5)
        s = Sample(data, SR)
        sig = s.render(0.5, sr=SR)
        assert sig.sample_rate == SR
        np.testing.assert_allclose(sig.data, data, atol=1e-5)

    def test_original_rate_trim(self):
        s = Sample(_sine_data(100, 1.0), SR)
        sig = s.render(0.5, sr=SR)
        assert len(sig.data) == int(0.5 * SR)

    def test_original_rate_pad(self):
        s = Sample(_sine_data(100, 0.5), SR)
        sig = s.render(1.0, sr=SR)
        assert len(sig.data) == SR
        # Tail should be zero (zero-padded beyond sample)
        assert np.allclose(sig.data[int(0.5 * SR) + 10:], 0.0, atol=1e-5)

    def test_pitch_shift_octave_up(self):
        """Shifting up an octave should double the frequency."""
        s = Sample(_sine_data(100, 1.0), SR, root_pitch=100.0)
        sig = s.render(0.5, 200.0, SR)
        # Count zero crossings — should be ~2x more than original rate
        sig_orig = s.render(0.5, 100.0, SR)
        zc_orig = int(np.sum(np.diff(np.sign(sig_orig.data)) != 0))
        zc_shifted = int(np.sum(np.diff(np.sign(sig.data)) != 0))
        assert zc_shifted == pytest.approx(zc_orig * 2, abs=4)

    def test_pitch_shift_octave_down(self):
        """Shifting down an octave should halve the frequency."""
        s = Sample(_sine_data(200, 1.0), SR, root_pitch=200.0)
        sig = s.render(0.5, 100.0, SR)
        sig_orig = s.render(0.5, 200.0, SR)
        zc_orig = int(np.sum(np.diff(np.sign(sig_orig.data)) != 0))
        zc_shifted = int(np.sum(np.diff(np.sign(sig.data)) != 0))
        assert zc_shifted == pytest.approx(zc_orig // 2, abs=4)

    def test_signal_hz_tracks_pitch(self):
        """Signal-rate Hz should produce varying pitch."""
        s = Sample(_sine_data(100, 1.0), SR, root_pitch=100.0)
        n = int(0.5 * SR)
        # First half at 100Hz, second half at 200Hz
        hz_data = np.concatenate([
            np.full(n // 2, 100.0),
            np.full(n - n // 2, 200.0),
        ])
        hz_sig = _sig(hz_data)
        sig = s.render(0.5, hz_sig, SR)
        assert len(sig.data) == n

    def test_render_hz_requires_root_pitch(self):
        s = Sample(np.zeros(100), SR)
        with pytest.raises(ValueError, match="root_pitch"):
            s.render(0.5, 440.0, SR)

    def test_render_no_hz_no_root_pitch_ok(self):
        s = Sample(np.zeros(100), SR)
        sig = s.render(0.5, sr=SR)
        assert isinstance(sig, Signal)

    def test_loop_points(self):
        """Loop points should cause the sample to repeat the loop region."""
        # Create a sample: [0, 0, 0, ... | 1, 1, 1, ... | 0, 0, 0, ...]
        #                    pre-loop      loop region      post (never reached)
        n = 300
        data = np.zeros(n, dtype=np.float32)
        loop_s, loop_e = 100, 200
        data[loop_s:loop_e] = 1.0
        s = Sample(data, SR, root_pitch=100.0, loop_start=loop_s, loop_end=loop_e)
        # Render longer than the sample — should loop the 1.0 region
        sig = s.render(1.0, 100.0, SR)
        # After the loop region starts, all values should be ~1.0
        # (the loop region is all 1.0)
        tail = sig.data[loop_s + 50:]
        # Most of the tail should be close to 1.0
        assert np.mean(np.abs(tail)) > 0.8

    def test_stereo_playback(self):
        left = _sine_data(100, 0.5)
        right = _sine_data(200, 0.5)
        stereo = np.column_stack([left, right])
        s = Sample(stereo, SR, root_pitch=100.0)
        sig = s.render(0.5, 100.0, SR)
        assert sig.n_channels == 2


# ------------------------------------------------------------------ #
# Import                                                              #
# ------------------------------------------------------------------ #


class TestImport:
    def test_importable_from_generators(self):
        from pysynth.generators.sample import Sample as S
        assert S is Sample
