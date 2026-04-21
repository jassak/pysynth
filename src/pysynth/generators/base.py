from __future__ import annotations

from abc import ABC


class Generator(ABC):
    """Abstract base for all audio generators.

    Generators are free-running signal sources that produce a finite
    :class:`Signal` when rendered.  The algebra defined here lets you
    compose generators *before* rendering:

        ``gen1 + gen2``   — additive synthesis (sum rendered signals)
        ``gen + scalar``  — DC offset
        ``gen1 * gen2``   — ring modulation (pointwise product)
        ``gen * scalar``  — gain scaling
        ``-gen``          — phase inversion
        ``gen1 - gen2``   — subtraction

    All binary operations are commutative where applicable.

    ``render`` is a **homomorphism** — composing then rendering gives
    the same result as rendering then composing::

        (g1 + g2).render(dur, hz=440)  ==  g1.render(dur, hz=440) + g2.render(dur, hz=440)
        (g1 * g2).render(dur, hz=440)  ==  g1.render(dur, hz=440) * g2.render(dur, hz=440)
    """

    # ------------------------------------------------------------------ #
    # Addition                                                             #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        if isinstance(other, Generator):
            return _SumGens(self, other)
        if isinstance(other, (int, float)):
            return _OffsetGen(self, float(other))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return _OffsetGen(self, float(other))
        return NotImplemented

    # ------------------------------------------------------------------ #
    # Multiplication                                                       #
    # ------------------------------------------------------------------ #

    def __mul__(self, other):
        if isinstance(other, Generator):
            return _ProdGens(self, other)
        if isinstance(other, (int, float)):
            return _ScaledGen(self, float(other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _ScaledGen(self, float(other))
        return NotImplemented

    # ------------------------------------------------------------------ #
    # Negation / subtraction                                               #
    # ------------------------------------------------------------------ #

    def __neg__(self):
        return _ScaledGen(self, -1.0)

    def __sub__(self, other):
        if isinstance(other, (Generator, int, float)):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _OffsetGen(-self, float(other))
        return NotImplemented


# ------------------------------------------------------------------ #
# Composite generator types                                            #
# ------------------------------------------------------------------ #


class _SumGens(Generator):
    """Additive composite — renders both children and sums the signals."""

    def __init__(self, left: Generator, right: Generator) -> None:
        self._left = left
        self._right = right

    def render(self, dur, **kwargs):
        return self._left.render(dur, **kwargs) + self._right.render(dur, **kwargs)

    def __repr__(self):
        return f"({self._left!r} + {self._right!r})"


class _ProdGens(Generator):
    """Ring-modulation composite — renders both children and multiplies."""

    def __init__(self, left: Generator, right: Generator) -> None:
        self._left = left
        self._right = right

    def render(self, dur, **kwargs):
        return self._left.render(dur, **kwargs) * self._right.render(dur, **kwargs)

    def __repr__(self):
        return f"({self._left!r} * {self._right!r})"


class _ScaledGen(Generator):
    """Gain-scaled generator — multiplies the rendered signal by a scalar."""

    def __init__(self, inner: Generator, gain: float) -> None:
        self._inner = inner
        self._gain = gain

    def render(self, dur, **kwargs):
        return self._inner.render(dur, **kwargs) * self._gain

    def __repr__(self):
        return f"({self._inner!r} * {self._gain})"


class _OffsetGen(Generator):
    """DC-offset generator — adds a scalar to the rendered signal."""

    def __init__(self, inner: Generator, offset: float) -> None:
        self._inner = inner
        self._offset = offset

    def render(self, dur, **kwargs):
        return self._inner.render(dur, **kwargs) + self._offset

    def __repr__(self):
        return f"({self._inner!r} + {self._offset})"
