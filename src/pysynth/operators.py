"""Operator: a Generator + Envelope pair with FM synthesis support.

An Operator is the atomic unit for building synthesizer voices.  It pairs
a :class:`Generator` (oscillator, noise, wavetable, ...) with an
:class:`Envelope` and supports algebraic composition:

    ``op1 + op2``   — additive (independent envelopes, summed output)
    ``op + scalar``  — DC offset
    ``op1 * op2``   — ring modulation (independent envelopes, multiplied)
    ``op * scalar``  — gain scaling
    ``-op``          — phase inversion
    ``op1 << op2``  — FM synthesis (op2 modulates op1's pitch)

``render`` is a **homomorphism** — composing then rendering gives
the same result as rendering then composing::

    (op1 + op2).render(pitch, gate) == op1.render(pitch, gate) + op2.render(pitch, gate)

FM example::

    carrier   = Operator(Oscillator('sine'),           adsr(0.01, 0.1, 0.7, 0.3))
    modulator = Operator(Oscillator('sine', ratio=2) * 200, adsr(0.01, 0.3, 0.0, 0.1))
    output    = (carrier << modulator).render(440.0, gate)

The ``<<`` operator means "right-hand side modulates left-hand side's pitch".
It is **non-associative**: ``(a << b) << c`` differs from ``a << (b << c)``.
Use parentheses to express the desired topology::

    carrier << (mod1 << mod2)   # cascade: mod2 → mod1 → carrier
    carrier << (mod1 + mod2)    # parallel: both modulate carrier independently
"""

from __future__ import annotations

from pysynth._core import Signal
from pysynth.envelopes.envelope import Envelope
from pysynth.generators.base import Generator


class _OperatorAlgebra:
    """Mixin providing the algebraic operations for all operator-like types."""

    def __add__(self, other):
        if isinstance(other, _OperatorAlgebra):
            return _SumOps(self, other)
        if isinstance(other, (int, float)):
            return _OffsetOp(self, float(other))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return _OffsetOp(self, float(other))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, _OperatorAlgebra):
            return _ProdOps(self, other)
        if isinstance(other, (int, float)):
            return _ScaledOp(self, float(other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _ScaledOp(self, float(other))
        return NotImplemented

    def __neg__(self):
        return _ScaledOp(self, -1.0)

    def __sub__(self, other):
        if isinstance(other, (_OperatorAlgebra, int, float)):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _OffsetOp(-self, float(other))
        return NotImplemented

    def __lshift__(self, modulator):
        if isinstance(modulator, _OperatorAlgebra):
            return _FMOp(self, modulator)
        return NotImplemented


class Operator(_OperatorAlgebra):
    """A Generator + Envelope pair — the atomic unit of synthesis voices.

    Parameters
    ----------
    generator:
        The audio source (Oscillator, WhiteNoise, Wavetable, ...).
    envelope:
        The amplitude envelope applied to the generator's output.
    """

    def __init__(self, generator: Generator, envelope: Envelope) -> None:
        self._generator = generator
        self._envelope = envelope

    def render(self, pitch: float | Signal, gate: Signal) -> Signal:
        """Render this operator to audio.

        Parameters
        ----------
        pitch:
            Fundamental frequency in Hz, or a time-varying pitch CV Signal.
        gate:
            Gate signal controlling the envelope (high = note on).
        """
        dur = gate.duration
        sr = gate.sample_rate
        raw = self._generator.render(dur, hz=pitch, sr=sr)
        env = self._envelope.trigger(gate)
        return raw * env

    def __repr__(self):
        return f"Operator({self._generator!r}, {self._envelope!r})"


# ------------------------------------------------------------------ #
# Composite operator types                                             #
# ------------------------------------------------------------------ #


class _SumOps(_OperatorAlgebra):
    """Additive composite — renders both children and sums the signals."""

    def __init__(self, left, right) -> None:
        self._left = left
        self._right = right

    def render(self, pitch: float | Signal, gate: Signal) -> Signal:
        return self._left.render(pitch, gate) + self._right.render(pitch, gate)

    def __repr__(self):
        return f"({self._left!r} + {self._right!r})"


class _ProdOps(_OperatorAlgebra):
    """Ring-modulation composite — renders both children and multiplies."""

    def __init__(self, left, right) -> None:
        self._left = left
        self._right = right

    def render(self, pitch: float | Signal, gate: Signal) -> Signal:
        return self._left.render(pitch, gate) * self._right.render(pitch, gate)

    def __repr__(self):
        return f"({self._left!r} * {self._right!r})"


class _ScaledOp(_OperatorAlgebra):
    """Gain-scaled operator — multiplies the rendered signal by a scalar."""

    def __init__(self, inner, gain: float) -> None:
        self._inner = inner
        self._gain = gain

    def render(self, pitch: float | Signal, gate: Signal) -> Signal:
        return self._inner.render(pitch, gate) * self._gain

    def __repr__(self):
        return f"({self._inner!r} * {self._gain})"


class _OffsetOp(_OperatorAlgebra):
    """DC-offset operator — adds a scalar to the rendered signal."""

    def __init__(self, inner, offset: float) -> None:
        self._inner = inner
        self._offset = offset

    def render(self, pitch: float | Signal, gate: Signal) -> Signal:
        return self._inner.render(pitch, gate) + self._offset

    def __repr__(self):
        return f"({self._inner!r} + {self._offset})"


class _FMOp(_OperatorAlgebra):
    """FM synthesis — modulator output is added to carrier's pitch.

    ``carrier << modulator`` creates an ``_FMOp`` where the modulator's
    rendered output (shaped by its own envelope) shifts the carrier's
    frequency.
    """

    def __init__(self, carrier, modulator) -> None:
        self._carrier = carrier
        self._modulator = modulator

    def render(self, pitch: float | Signal, gate: Signal) -> Signal:
        mod_signal = self._modulator.render(pitch, gate)
        modulated_pitch = pitch + mod_signal
        return self._carrier.render(modulated_pitch, gate)

    def __repr__(self):
        return f"({self._carrier!r} << {self._modulator!r})"
