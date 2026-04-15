import numpy as np
import pytest

from pysynth._core import SAMPLE_RATE, Signal, Effect, _Chain, Generator, _as_array


SR = 1000  # low sample rate for fast tests


def _sig(values, sr=SR):
    return Signal(np.array(values, dtype=np.float32), sr)


# ------------------------------------------------------------------ #
# Signal basics                                                        #
# ------------------------------------------------------------------ #


class TestSignalProperties:
    def test_duration(self):
        s = _sig([0.0] * 500)
        assert abs(s.duration - 0.5) < 1e-6

    def test_mono_channels(self):
        assert _sig([1, 2, 3]).n_channels == 1

    def test_stereo_channels(self):
        data = np.zeros((100, 2), dtype=np.float32)
        assert Signal(data, SR).n_channels == 2

    def test_data_cast_to_float32(self):
        s = Signal(np.array([1, 2, 3], dtype=np.float64), SR)
        assert s.data.dtype == np.float32

    def test_silence(self):
        s = Signal.silence(0.5, SR)
        assert len(s) == 500
        assert np.all(s.data == 0.0)

    def test_len(self):
        s = _sig([1.0, 2.0, 3.0])
        assert len(s) == 3


# ------------------------------------------------------------------ #
# Signal slicing                                                       #
# ------------------------------------------------------------------ #


class TestSignalSlicing:
    def test_basic_slice(self):
        s = _sig(np.arange(SR, dtype=np.float32))  # 1 second
        sliced = s[0.25:0.75]
        assert len(sliced) == int(0.5 * SR)
        assert sliced.duration == pytest.approx(0.5)

    def test_none_start(self):
        s = _sig(np.arange(SR, dtype=np.float32))
        sliced = s[:0.5]
        assert len(sliced) == int(0.5 * SR)

    def test_none_end(self):
        s = _sig(np.arange(SR, dtype=np.float32))
        sliced = s[0.5:]
        assert len(sliced) == int(0.5 * SR)

    def test_out_of_bounds_clamped(self):
        s = _sig(np.zeros(SR))
        sliced = s[-1.0:5.0]
        assert len(sliced) == SR

    def test_preserves_sample_rate(self):
        s = Signal(np.zeros(22050, dtype=np.float32), 22050)
        assert s[0.0:0.5].sample_rate == 22050

    def test_returns_copy(self):
        s = _sig([1.0, 2.0, 3.0, 4.0])
        sliced = s[0.0:0.003]
        sliced.data[:] = 0.0
        assert s.data[0] == 1.0

    def test_non_slice_raises(self):
        s = _sig(np.zeros(SR))
        with pytest.raises(TypeError):
            s[42]

    def test_step_raises(self):
        s = _sig(np.zeros(SR))
        with pytest.raises(ValueError, match="Step"):
            s[0.0:1.0:0.1]


# ------------------------------------------------------------------ #
# Signal normalize                                                     #
# ------------------------------------------------------------------ #


class TestSignalNormalize:
    def test_peak_normalize(self):
        s = _sig([0.0, 0.25, -0.5, 0.1]).normalize()
        assert np.max(np.abs(s.data)) == pytest.approx(1.0)

    def test_already_normalized(self):
        data = [0.0, 1.0, -1.0]
        s = _sig(data).normalize()
        np.testing.assert_allclose(s.data, data, atol=1e-7)

    def test_zero_signal(self):
        s = _sig(np.zeros(100)).normalize()
        assert np.all(s.data == 0.0)

    def test_returns_copy(self):
        s = _sig([0.5, -0.5])
        n = s.normalize()
        assert n.data is not s.data

    def test_unsupported_mode(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _sig([1.0]).normalize(mode="rms")


# ------------------------------------------------------------------ #
# Signal algebra — addition                                            #
# ------------------------------------------------------------------ #


class TestSignalAddition:
    def test_add_same_length(self):
        a = _sig([1.0, 2.0, 3.0])
        b = _sig([0.5, 0.5, 0.5])
        c = a + b
        np.testing.assert_allclose(c.data, [1.5, 2.5, 3.5], atol=1e-6)

    def test_add_zero_pads_shorter(self):
        a = _sig([1.0, 2.0, 3.0])
        b = _sig([10.0])
        c = a + b
        assert len(c.data) == 3
        np.testing.assert_allclose(c.data, [11.0, 2.0, 3.0], atol=1e-6)

    def test_add_commutative(self):
        a = _sig([1.0, 2.0])
        b = _sig([3.0, 4.0, 5.0])
        np.testing.assert_allclose((a + b).data, (b + a).data, atol=1e-6)

    def test_add_scalar(self):
        s = _sig([1.0, 2.0])
        c = s + 10.0
        np.testing.assert_allclose(c.data, [11.0, 12.0], atol=1e-6)

    def test_radd_scalar(self):
        s = _sig([1.0, 2.0])
        c = 10.0 + s
        np.testing.assert_allclose(c.data, [11.0, 12.0], atol=1e-6)

    def test_add_different_sample_rates_raises(self):
        a = Signal(np.zeros(10, dtype=np.float32), 44100)
        b = Signal(np.zeros(10, dtype=np.float32), 22050)
        with pytest.raises(ValueError, match="sample rates"):
            a + b


# ------------------------------------------------------------------ #
# Signal algebra — multiplication                                     #
# ------------------------------------------------------------------ #


class TestSignalMultiplication:
    def test_mul_same_length(self):
        a = _sig([2.0, 3.0])
        b = _sig([4.0, 5.0])
        c = a * b
        np.testing.assert_allclose(c.data, [8.0, 15.0], atol=1e-6)

    def test_mul_truncates_to_shorter(self):
        a = _sig([1.0, 2.0, 3.0])
        b = _sig([10.0])
        c = a * b
        assert len(c.data) == 1
        np.testing.assert_allclose(c.data, [10.0], atol=1e-6)

    def test_mul_commutative(self):
        a = _sig([1.0, 2.0])
        b = _sig([3.0, 4.0])
        np.testing.assert_allclose((a * b).data, (b * a).data, atol=1e-6)

    def test_mul_scalar(self):
        s = _sig([1.0, 2.0, 3.0])
        c = s * 0.5
        np.testing.assert_allclose(c.data, [0.5, 1.0, 1.5], atol=1e-6)

    def test_rmul_scalar(self):
        s = _sig([1.0, 2.0])
        c = 3.0 * s
        np.testing.assert_allclose(c.data, [3.0, 6.0], atol=1e-6)

    def test_mul_different_sample_rates_raises(self):
        a = Signal(np.zeros(10, dtype=np.float32), 44100)
        b = Signal(np.zeros(10, dtype=np.float32), 22050)
        with pytest.raises(ValueError, match="sample rates"):
            a * b


# ------------------------------------------------------------------ #
# Signal algebra — negation and subtraction                            #
# ------------------------------------------------------------------ #


class TestSignalNegSub:
    def test_neg(self):
        s = _sig([1.0, -2.0, 3.0])
        np.testing.assert_allclose((-s).data, [-1.0, 2.0, -3.0], atol=1e-6)

    def test_sub_signal(self):
        a = _sig([5.0, 6.0])
        b = _sig([1.0, 2.0])
        np.testing.assert_allclose((a - b).data, [4.0, 4.0], atol=1e-6)

    def test_sub_scalar(self):
        s = _sig([5.0, 6.0])
        np.testing.assert_allclose((s - 1.0).data, [4.0, 5.0], atol=1e-6)


# ------------------------------------------------------------------ #
# Signal algebra — immutability                                        #
# ------------------------------------------------------------------ #


def _stereo(left, right, sr=SR):
    return Signal(np.column_stack([
        np.array(left, dtype=np.float32),
        np.array(right, dtype=np.float32),
    ]), sr)


# ------------------------------------------------------------------ #
# Signal algebra — mono/stereo interop                                 #
# ------------------------------------------------------------------ #


class TestMonoStereoAdd:
    def test_mono_plus_stereo(self):
        m = _sig([1.0, 2.0])
        s = _stereo([10.0, 20.0], [100.0, 200.0])
        c = m + s
        assert c.n_channels == 2
        np.testing.assert_allclose(c.data[:, 0], [11.0, 22.0], atol=1e-6)
        np.testing.assert_allclose(c.data[:, 1], [101.0, 202.0], atol=1e-6)

    def test_stereo_plus_mono(self):
        m = _sig([1.0, 2.0])
        s = _stereo([10.0, 20.0], [100.0, 200.0])
        np.testing.assert_allclose((s + m).data, (m + s).data, atol=1e-6)

    def test_mono_plus_stereo_different_lengths(self):
        m = _sig([1.0, 2.0, 3.0])
        s = _stereo([10.0], [100.0])
        c = m + s
        assert c.n_channels == 2
        assert len(c.data) == 3
        np.testing.assert_allclose(c.data[0], [11.0, 101.0], atol=1e-6)
        np.testing.assert_allclose(c.data[1], [2.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(c.data[2], [3.0, 3.0], atol=1e-6)

    def test_mono_sub_stereo(self):
        m = _sig([5.0, 6.0])
        s = _stereo([1.0, 2.0], [3.0, 4.0])
        c = m - s
        assert c.n_channels == 2
        np.testing.assert_allclose(c.data[:, 0], [4.0, 4.0], atol=1e-6)
        np.testing.assert_allclose(c.data[:, 1], [2.0, 2.0], atol=1e-6)


class TestMonoStereoMul:
    def test_mono_times_stereo(self):
        m = _sig([2.0, 3.0])
        s = _stereo([10.0, 20.0], [100.0, 200.0])
        c = m * s
        assert c.n_channels == 2
        np.testing.assert_allclose(c.data[:, 0], [20.0, 60.0], atol=1e-6)
        np.testing.assert_allclose(c.data[:, 1], [200.0, 600.0], atol=1e-6)

    def test_stereo_times_mono(self):
        m = _sig([2.0, 3.0])
        s = _stereo([10.0, 20.0], [100.0, 200.0])
        np.testing.assert_allclose((s * m).data, (m * s).data, atol=1e-6)

    def test_mono_times_stereo_truncates(self):
        m = _sig([2.0, 3.0, 4.0])
        s = _stereo([10.0], [100.0])
        c = m * s
        assert len(c.data) == 1
        np.testing.assert_allclose(c.data, [[20.0, 200.0]], atol=1e-6)


class TestSignalShift:
    def test_shift_prepends_silence(self):
        s = _sig([1.0, 2.0, 3.0])
        shifted = s.shift(0.002)  # 2 samples at SR=1000
        assert len(shifted.data) == 5
        np.testing.assert_allclose(shifted.data, [0.0, 0.0, 1.0, 2.0, 3.0], atol=1e-6)

    def test_shift_zero_returns_copy(self):
        s = _sig([1.0, 2.0])
        shifted = s.shift(0.0)
        np.testing.assert_array_equal(shifted.data, s.data)
        assert shifted.data is not s.data

    def test_shift_negative_returns_copy(self):
        s = _sig([1.0, 2.0])
        shifted = s.shift(-1.0)
        np.testing.assert_array_equal(shifted.data, s.data)
        assert shifted.data is not s.data

    def test_shift_does_not_mutate_original(self):
        s = _sig([1.0, 2.0])
        original = s.data.copy()
        _ = s.shift(0.005)
        np.testing.assert_array_equal(s.data, original)

    def test_shift_preserves_sample_rate(self):
        s = _sig([1.0], sr=22050)
        shifted = s.shift(0.1)
        assert shifted.sample_rate == 22050

    def test_shift_stereo(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        s = Signal(data, SR)
        shifted = s.shift(0.001)  # 1 sample at SR=1000
        assert shifted.data.shape == (3, 2)
        np.testing.assert_allclose(shifted.data[0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(shifted.data[1], [1.0, 2.0], atol=1e-6)

    def test_shift_composes_with_add(self):
        a = _sig([1.0, 1.0])
        b = _sig([2.0, 2.0])
        mixed = a + b.shift(0.001)  # b starts 1 sample later
        # a=[1,1,0] + b_shifted=[0,2,2] = [1,3,2]
        np.testing.assert_allclose(mixed.data, [1.0, 3.0, 2.0], atol=1e-6)


class TestConcat:
    def test_concat_two_signals(self):
        a = _sig([1.0, 2.0])
        b = _sig([3.0, 4.0, 5.0])
        c = Signal.concat(a, b)
        np.testing.assert_allclose(c.data, [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-6)
        assert c.duration == pytest.approx(a.duration + b.duration, abs=1e-6)

    def test_concat_multiple(self):
        a = _sig([1.0])
        b = _sig([2.0])
        c = _sig([3.0])
        result = Signal.concat(a, b, c)
        np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0], atol=1e-6)

    def test_concat_does_not_mutate(self):
        a = _sig([1.0, 2.0])
        b = _sig([3.0, 4.0])
        a_orig = a.data.copy()
        b_orig = b.data.copy()
        _ = Signal.concat(a, b)
        np.testing.assert_array_equal(a.data, a_orig)
        np.testing.assert_array_equal(b.data, b_orig)

    def test_concat_mismatched_sample_rate(self):
        a = Signal(np.array([1.0]), sample_rate=44100)
        b = Signal(np.array([2.0]), sample_rate=22050)
        with pytest.raises(ValueError, match="sample rates"):
            Signal.concat(a, b)

    def test_concat_mono_stereo(self):
        m = _sig([1.0, 2.0])
        s = _stereo([3.0, 4.0], [5.0, 6.0])
        result = Signal.concat(m, s)
        assert result.n_channels == 2
        assert len(result.data) == 4

    def test_concat_preserves_sample_rate(self):
        a = Signal(np.array([1.0]), sample_rate=22050)
        b = Signal(np.array([2.0]), sample_rate=22050)
        assert Signal.concat(a, b).sample_rate == 22050

    def test_concat_requires_two_signals(self):
        with pytest.raises(ValueError, match="at least two"):
            Signal.concat(_sig([1.0]))


class TestSignalImmutability:
    def test_add_does_not_mutate(self):
        a = _sig([1.0, 2.0])
        original = a.data.copy()
        _ = a + _sig([10.0, 20.0])
        np.testing.assert_array_equal(a.data, original)

    def test_mul_does_not_mutate(self):
        a = _sig([1.0, 2.0])
        original = a.data.copy()
        _ = a * _sig([10.0, 20.0])
        np.testing.assert_array_equal(a.data, original)


# ------------------------------------------------------------------ #
# Effect and _Chain                                                    #
# ------------------------------------------------------------------ #


class TestEffect:
    def test_base_raises(self):
        with pytest.raises(NotImplementedError):
            Effect()(_sig([1.0]))

    def test_chain(self):
        class Double(Effect):
            def __call__(self, signal):
                return signal * 2.0

        class AddOne(Effect):
            def __call__(self, signal):
                return signal + 1.0

        chain = Double() | AddOne()
        assert isinstance(chain, _Chain)
        result = chain(_sig([3.0]))
        # Double first (3*2=6), then add one (6+1=7)
        np.testing.assert_allclose(result.data, [7.0], atol=1e-6)


# ------------------------------------------------------------------ #
# Generator                                                            #
# ------------------------------------------------------------------ #


class TestGenerator:
    def test_render(self):
        gen = Generator(lambda dur, sr: Signal.silence(dur, sr))
        sig = gen.render(0.5, SR)
        assert abs(sig.duration - 0.5) < 1e-6

    def test_repr_with_name(self):
        gen = Generator(lambda d, s: None, name="test")
        assert "test" in repr(gen)

    def test_repr_without_name(self):
        gen = Generator(lambda d, s: None)
        assert "..." in repr(gen)


# ------------------------------------------------------------------ #
# _as_array helper                                                     #
# ------------------------------------------------------------------ #


class TestAsArray:
    def test_float_to_array(self):
        arr = _as_array(5.0, 3)
        np.testing.assert_allclose(arr, [5.0, 5.0, 5.0])
        assert arr.dtype == np.float64

    def test_signal_exact_length(self):
        s = _sig([1.0, 2.0, 3.0])
        arr = _as_array(s, 3)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_signal_truncated(self):
        s = _sig([1.0, 2.0, 3.0])
        arr = _as_array(s, 2)
        np.testing.assert_allclose(arr, [1.0, 2.0])

    def test_signal_edge_padded(self):
        s = _sig([1.0, 2.0, 5.0])
        arr = _as_array(s, 5)
        np.testing.assert_allclose(arr, [1.0, 2.0, 5.0, 5.0, 5.0])

    def test_zero_length_signal_raises(self):
        s = _sig([])
        with pytest.raises(ValueError, match="zero length"):
            _as_array(s, 3)
