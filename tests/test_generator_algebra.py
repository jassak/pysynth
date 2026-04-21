"""Tests for Generator base class algebra.

Verifies the algebraic operations (+, *, -, neg) on Generator and the
homomorphism property: composing then rendering == rendering then composing.
"""

import numpy as np
import pytest

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.generators.base import (
    Generator, _SumGens, _ProdGens, _ScaledGen, _OffsetGen,
)
from pysynth.generators.oscillators import Oscillator
from pysynth.generators.noise import WhiteNoise, PinkNoise
from pysynth.generators.wavetable import Wavetable
from pysynth.generators.sample import Sample
from pysynth.generators.granular import Granular


SR = 4000


# ------------------------------------------------------------------ #
# isinstance checks                                                    #
# ------------------------------------------------------------------ #


class TestIsInstance:
    def test_oscillator_is_generator(self):
        assert isinstance(Oscillator("sine"), Generator)

    def test_white_noise_is_generator(self):
        assert isinstance(WhiteNoise(), Generator)

    def test_pink_noise_is_generator(self):
        assert isinstance(PinkNoise(), Generator)

    def test_wavetable_is_generator(self):
        wt = Wavetable.from_waveforms(["sine"])
        assert isinstance(wt, Generator)

    def test_sample_is_generator(self):
        data = np.zeros(100, dtype=np.float32)
        assert isinstance(Sample(data), Generator)

    def test_composite_is_generator(self):
        gen = Oscillator("sine") + WhiteNoise()
        assert isinstance(gen, Generator)

    def test_scaled_is_generator(self):
        gen = Oscillator("sine") * 0.5
        assert isinstance(gen, Generator)


# ------------------------------------------------------------------ #
# Addition                                                             #
# ------------------------------------------------------------------ #


class TestAddition:
    def test_gen_plus_gen(self):
        gen = Oscillator("sine") + WhiteNoise(sample_rate=SR)
        assert isinstance(gen, _SumGens)

    def test_osc_plus_osc(self):
        gen = Oscillator("sine") + Oscillator("saw")
        assert isinstance(gen, _SumGens)

    def test_gen_plus_scalar(self):
        gen = Oscillator("sine") + 1.0
        assert isinstance(gen, _OffsetGen)

    def test_scalar_plus_gen(self):
        gen = 1.0 + Oscillator("sine")
        assert isinstance(gen, _OffsetGen)

    def test_add_renders(self):
        gen = Oscillator("sine") + Oscillator("saw")
        sig = gen.render(0.1, hz=440, sr=SR)
        assert isinstance(sig, Signal)
        assert abs(sig.duration - 0.1) < 0.01

    def test_add_does_not_mutate(self):
        a = Oscillator("sine")
        b = WhiteNoise()
        _ = a + b
        # a and b are unchanged (no shared state modified)
        assert isinstance(a, Oscillator)
        assert isinstance(b, WhiteNoise)

    def test_offset_renders(self):
        osc = Oscillator("sine")
        gen = osc + 0.5
        sig = gen.render(0.1, hz=440, sr=SR)
        sig_raw = osc.render(0.1, hz=440, sr=SR)
        np.testing.assert_allclose(sig.data, sig_raw.data + 0.5, atol=1e-6)


# ------------------------------------------------------------------ #
# Multiplication                                                       #
# ------------------------------------------------------------------ #


class TestMultiplication:
    def test_gen_times_scalar(self):
        gen = Oscillator("sine") * 0.5
        assert isinstance(gen, _ScaledGen)

    def test_scalar_times_gen(self):
        gen = 0.5 * Oscillator("sine")
        assert isinstance(gen, _ScaledGen)

    def test_gen_times_gen_ring_mod(self):
        gen = Oscillator("sine") * Oscillator("saw")
        assert isinstance(gen, _ProdGens)

    def test_scale_renders(self):
        osc = Oscillator("sine")
        gen = osc * 0.5
        sig = gen.render(0.1, hz=440, sr=SR)
        sig_raw = osc.render(0.1, hz=440, sr=SR)
        np.testing.assert_allclose(sig.data, sig_raw.data * 0.5, atol=1e-6)

    def test_ring_mod_renders(self):
        gen = Oscillator("sine") * Oscillator("saw")
        sig = gen.render(0.1, hz=440, sr=SR)
        assert isinstance(sig, Signal)


# ------------------------------------------------------------------ #
# Negation / subtraction                                               #
# ------------------------------------------------------------------ #


class TestNegationSubtraction:
    def test_negation(self):
        osc = Oscillator("sine")
        gen = -osc
        sig_pos = osc.render(0.1, hz=440, sr=SR)
        sig_neg = gen.render(0.1, hz=440, sr=SR)
        np.testing.assert_allclose(sig_neg.data, -sig_pos.data, atol=1e-6)

    def test_subtraction(self):
        a = Oscillator("sine")
        b = Oscillator("sine")
        gen = a - b
        sig = gen.render(0.1, hz=440, sr=SR)
        # Same oscillator subtracted from itself should be ~zero
        np.testing.assert_allclose(sig.data, 0.0, atol=1e-6)

    def test_scalar_minus_gen(self):
        osc = Oscillator("sine")
        gen = 1.0 - osc
        sig = gen.render(0.1, hz=440, sr=SR)
        sig_raw = osc.render(0.1, hz=440, sr=SR)
        np.testing.assert_allclose(sig.data, 1.0 - sig_raw.data, atol=1e-6)

    def test_gen_minus_scalar(self):
        osc = Oscillator("sine")
        gen = osc - 0.5
        sig = gen.render(0.1, hz=440, sr=SR)
        sig_raw = osc.render(0.1, hz=440, sr=SR)
        np.testing.assert_allclose(sig.data, sig_raw.data - 0.5, atol=1e-6)


# ------------------------------------------------------------------ #
# Homomorphism                                                         #
# ------------------------------------------------------------------ #


class TestHomomorphism:
    """render(g1 op g2) == render(g1) op render(g2)"""

    def test_add_homomorphism(self):
        g1 = Oscillator("sine")
        g2 = Oscillator("saw")
        composed = (g1 + g2).render(0.1, hz=440, sr=SR)
        separate = g1.render(0.1, hz=440, sr=SR) + g2.render(0.1, hz=440, sr=SR)
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)

    def test_mul_homomorphism(self):
        g1 = Oscillator("sine")
        g2 = Oscillator("saw")
        composed = (g1 * g2).render(0.1, hz=440, sr=SR)
        separate = g1.render(0.1, hz=440, sr=SR) * g2.render(0.1, hz=440, sr=SR)
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)

    def test_scalar_mul_homomorphism(self):
        g = Oscillator("sine")
        composed = (g * 0.3).render(0.1, hz=440, sr=SR)
        separate = g.render(0.1, hz=440, sr=SR) * 0.3
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)

    def test_scalar_add_homomorphism(self):
        g = Oscillator("sine")
        composed = (g + 0.5).render(0.1, hz=440, sr=SR)
        separate = g.render(0.1, hz=440, sr=SR) + 0.5
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)

    def test_neg_homomorphism(self):
        g = Oscillator("sine")
        composed = (-g).render(0.1, hz=440, sr=SR)
        separate = -(g.render(0.1, hz=440, sr=SR))
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)


# ------------------------------------------------------------------ #
# Cross-type composition                                               #
# ------------------------------------------------------------------ #


class TestCrossType:
    def test_osc_plus_noise(self):
        gen = Oscillator("sine") + WhiteNoise(sample_rate=SR)
        sig = gen.render(0.1, hz=440, sr=SR)
        assert isinstance(sig, Signal)
        assert abs(sig.duration - 0.1) < 0.01

    def test_noise_ignores_hz(self):
        """WhiteNoise should silently ignore hz kwarg via **_kwargs."""
        noise = WhiteNoise(sample_rate=SR)
        sig = noise.render(0.1, hz=440, sr=SR)
        assert isinstance(sig, Signal)

    def test_cross_type_homomorphism(self):
        """Homomorphism holds across different generator types."""
        np.random.seed(42)
        g1 = Oscillator("sine")
        np.random.seed(42)
        g2_a = WhiteNoise(sample_rate=SR)
        np.random.seed(42)
        g2_b = WhiteNoise(sample_rate=SR)

        np.random.seed(42)
        composed = (g1 + g2_a).render(0.1, hz=440, sr=SR)

        s1 = g1.render(0.1, hz=440, sr=SR)
        np.random.seed(42)
        s2 = g2_b.render(0.1, sr=SR)
        separate = s1 + s2

        np.testing.assert_allclose(composed.data, separate.data, atol=1e-5)


# ------------------------------------------------------------------ #
# Chaining                                                             #
# ------------------------------------------------------------------ #


class TestChaining:
    def test_triple_add(self):
        gen = Oscillator("sine") + Oscillator("saw") + Oscillator("triangle")
        sig = gen.render(0.1, hz=440, sr=SR)
        assert isinstance(sig, Signal)

    def test_mixed_ops(self):
        gen = (Oscillator("sine") + Oscillator("saw")) * 0.5
        sig = gen.render(0.1, hz=440, sr=SR)
        assert isinstance(sig, Signal)

    def test_not_implemented_for_non_generator(self):
        osc = Oscillator("sine")
        assert osc.__add__("string") is NotImplemented
        assert osc.__mul__("string") is NotImplemented
        assert osc.__sub__("string") is NotImplemented
