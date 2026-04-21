"""Tests for the Operator class — algebra, FM synthesis, homomorphism."""

import numpy as np
import pytest

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.envelopes.envelope import Envelope, Segment, adsr
from pysynth.generators.oscillators import Oscillator
from pysynth.generators.noise import WhiteNoise
from pysynth.operators import Operator, _SumOps, _ScaledOp, _FMOp


SR = 4000


def _gate(dur: float, high: float | None = None, sr: int = SR) -> Signal:
    """Create a gate signal: high for *high* seconds, then low."""
    n = int(dur * sr)
    if high is None:
        high = dur
    n_high = min(int(high * sr), n)
    data = np.zeros(n, dtype=np.float32)
    data[:n_high] = 1.0
    return Signal(data, sr)


# ------------------------------------------------------------------ #
# Basic render                                                         #
# ------------------------------------------------------------------ #


class TestBasicRender:
    def test_returns_signal(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        sig = op.render(440.0, _gate(0.5, 0.3))
        assert isinstance(sig, Signal)

    def test_duration_matches_gate(self):
        gate = _gate(0.5, 0.3)
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        sig = op.render(440.0, gate)
        assert abs(sig.duration - gate.duration) < 0.01

    def test_envelope_applied(self):
        """After gate release, signal should decay toward zero."""
        gate = _gate(0.5, 0.1)  # gate on for 0.1s, total 0.5s
        op = Operator(Oscillator("sine"), adsr(0.01, 0.02, 0.5, 0.05))
        sig = op.render(440.0, gate)
        # Last 10% of signal should be near zero (release done)
        tail = sig.data[int(0.45 * SR):]
        assert np.abs(tail).max() < 0.01

    def test_pitch_cv_signal(self):
        """Pitch can be a time-varying Signal."""
        gate = _gate(0.2)
        pitch_cv = Signal(np.full(int(0.2 * SR), 440.0, dtype=np.float32), SR)
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        sig = op.render(pitch_cv, gate)
        assert isinstance(sig, Signal)

    def test_unpitched_generator(self):
        """WhiteNoise ignores pitch — should still render."""
        gate = _gate(0.2)
        op = Operator(WhiteNoise(sample_rate=SR), adsr(0.01, 0.05, 0.7, 0.05))
        sig = op.render(440.0, gate)
        assert isinstance(sig, Signal)
        assert abs(sig.duration - 0.2) < 0.01


# ------------------------------------------------------------------ #
# Algebra                                                              #
# ------------------------------------------------------------------ #


class TestAlgebra:
    def test_add_returns_sum(self):
        op1 = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        op2 = Operator(Oscillator("saw"), adsr(0.01, 0.1, 0.5, 0.1))
        assert isinstance(op1 + op2, _SumOps)

    def test_mul_scalar(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        assert isinstance(op * 0.5, _ScaledOp)

    def test_rmul_scalar(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        assert isinstance(0.5 * op, _ScaledOp)

    def test_negation(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        gate = _gate(0.1)
        sig_pos = op.render(440.0, gate)
        sig_neg = (-op).render(440.0, gate)
        np.testing.assert_allclose(sig_neg.data, -sig_pos.data, atol=1e-6)

    def test_scalar_mul_renders(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        gate = _gate(0.1)
        sig_full = op.render(440.0, gate)
        sig_half = (op * 0.5).render(440.0, gate)
        np.testing.assert_allclose(sig_half.data, sig_full.data * 0.5, atol=1e-6)

    def test_add_renders(self):
        op1 = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        op2 = Operator(Oscillator("saw"), adsr(0.01, 0.05, 0.7, 0.05))
        gate = _gate(0.1)
        sig = (op1 + op2).render(440.0, gate)
        assert isinstance(sig, Signal)

    def test_not_implemented_for_non_operator(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        assert op.__add__("string") is NotImplemented
        assert op.__mul__("string") is NotImplemented
        assert op.__lshift__("string") is NotImplemented


# ------------------------------------------------------------------ #
# Homomorphism                                                         #
# ------------------------------------------------------------------ #


class TestHomomorphism:
    def test_add_homomorphism(self):
        op1 = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        op2 = Operator(Oscillator("saw"), adsr(0.01, 0.05, 0.7, 0.05))
        gate = _gate(0.1)
        composed = (op1 + op2).render(440.0, gate)
        separate = op1.render(440.0, gate) + op2.render(440.0, gate)
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)

    def test_scalar_mul_homomorphism(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        gate = _gate(0.1)
        composed = (op * 0.3).render(440.0, gate)
        separate = op.render(440.0, gate) * 0.3
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)

    def test_neg_homomorphism(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        gate = _gate(0.1)
        composed = (-op).render(440.0, gate)
        separate = -(op.render(440.0, gate))
        np.testing.assert_allclose(composed.data, separate.data, atol=1e-6)


# ------------------------------------------------------------------ #
# FM synthesis                                                         #
# ------------------------------------------------------------------ #


class TestFM:
    def test_fm_returns_fm_op(self):
        carrier = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        mod = Operator(Oscillator("sine", ratio=2) * 200, adsr(0.01, 0.05, 0.7, 0.05))
        assert isinstance(carrier << mod, _FMOp)

    def test_fm_renders(self):
        carrier = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        mod = Operator(Oscillator("sine", ratio=2) * 200, adsr(0.01, 0.05, 0.7, 0.05))
        gate = _gate(0.2)
        sig = (carrier << mod).render(440.0, gate)
        assert isinstance(sig, Signal)
        assert abs(sig.duration - 0.2) < 0.01

    def test_fm_differs_from_unmodulated(self):
        """FM output should be different from the unmodulated carrier."""
        env = adsr(0.01, 0.05, 0.7, 0.05)
        carrier = Operator(Oscillator("sine"), env)
        mod = Operator(Oscillator("sine", ratio=2) * 200, env)
        gate = _gate(0.2)
        sig_plain = carrier.render(440.0, gate)
        sig_fm = (carrier << mod).render(440.0, gate)
        # They should differ significantly
        diff = np.abs(sig_fm.data - sig_plain.data).max()
        assert diff > 0.01

    def test_fm_cascade(self):
        """carrier << (mod1 << mod2) — 3-operator cascade."""
        env = adsr(0.01, 0.05, 0.7, 0.05)
        carrier = Operator(Oscillator("sine"), env)
        mod1 = Operator(Oscillator("sine", ratio=2) * 100, env)
        mod2 = Operator(Oscillator("sine", ratio=3) * 50, env)
        gate = _gate(0.2)
        sig = (carrier << (mod1 << mod2)).render(440.0, gate)
        assert isinstance(sig, Signal)

    def test_fm_parallel_via_add(self):
        """carrier << (mod1 + mod2) — parallel modulation."""
        env = adsr(0.01, 0.05, 0.7, 0.05)
        carrier = Operator(Oscillator("sine"), env)
        mod1 = Operator(Oscillator("sine", ratio=2) * 100, env)
        mod2 = Operator(Oscillator("sine", ratio=3) * 50, env)
        gate = _gate(0.2)
        sig = (carrier << (mod1 + mod2)).render(440.0, gate)
        assert isinstance(sig, Signal)

    def test_fm_non_associativity(self):
        """(a << b) << c should differ from a << (b << c)."""
        env = adsr(0.01, 0.05, 0.7, 0.05)
        a = Operator(Oscillator("sine"), env)
        b = Operator(Oscillator("sine", ratio=2) * 200, env)
        c = Operator(Oscillator("sine", ratio=3) * 100, env)
        gate = _gate(0.2)
        left_assoc = ((a << b) << c).render(440.0, gate)
        right_assoc = (a << (b << c)).render(440.0, gate)
        # These should produce different results
        diff = np.abs(left_assoc.data - right_assoc.data).max()
        assert diff > 0.01

    def test_fm_with_different_envelopes(self):
        """Carrier and modulator can have different envelopes."""
        carrier = Operator(Oscillator("sine"), adsr(0.01, 0.1, 0.7, 0.3))
        mod = Operator(
            Oscillator("sine", ratio=2) * 200,
            adsr(0.01, 0.3, 0.0, 0.1),  # modulator envelope decays to 0
        )
        gate = _gate(0.5, 0.3)
        sig = (carrier << mod).render(440.0, gate)
        assert isinstance(sig, Signal)

    def test_fm_can_chain_with_algebra(self):
        """FM result supports further algebra."""
        env = adsr(0.01, 0.05, 0.7, 0.05)
        fm = Operator(Oscillator("sine"), env) << Operator(Oscillator("sine", ratio=2) * 200, env)
        scaled = fm * 0.5
        gate = _gate(0.1)
        sig = scaled.render(440.0, gate)
        sig_fm = fm.render(440.0, gate)
        np.testing.assert_allclose(sig.data, sig_fm.data * 0.5, atol=1e-6)


# ------------------------------------------------------------------ #
# Repr                                                                 #
# ------------------------------------------------------------------ #


class TestRepr:
    def test_operator_repr(self):
        op = Operator(Oscillator("sine"), adsr(0.01, 0.05, 0.7, 0.05))
        r = repr(op)
        assert "Operator" in r

    def test_fm_repr(self):
        env = adsr(0.01, 0.05, 0.7, 0.05)
        fm = Operator(Oscillator("sine"), env) << Operator(Oscillator("saw"), env)
        r = repr(fm)
        assert "<<" in r
