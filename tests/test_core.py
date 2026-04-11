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
        assert len(s.data) == 500
        assert np.all(s.data == 0.0)


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
