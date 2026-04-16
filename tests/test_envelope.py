import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.envelopes import Segment, Envelope, adsr


SR = 1000


def _gate(high_seconds, total_seconds):
    """Create a gate signal: high for *high_seconds*, then low."""
    n_total = int(total_seconds * SR)
    n_high = int(high_seconds * SR)
    data = np.zeros(n_total, dtype=np.float32)
    data[:n_high] = 1.0
    return Signal(data, SR)


class TestAdsrNormalRelease:
    """Gate held through full attack+decay — release starts from sustain."""

    def test_release_starts_at_sustain(self):
        env = adsr(0.01, 0.05, 0.7, 0.1, sample_rate=SR)
        gate = _gate(0.2, 0.4)
        out = env.trigger(gate)
        # Just after gate drops at 0.2s, level should be ~sustain
        idx = int(0.2 * SR) + 1
        assert out.data[idx] == pytest.approx(0.7, abs=0.05)

    def test_release_ends_at_zero(self):
        env = adsr(0.01, 0.05, 0.7, 0.1, sample_rate=SR)
        gate = _gate(0.2, 0.4)
        out = env.trigger(gate)
        # Well after release completes (0.2 + 0.1 = 0.3s)
        assert out.data[-1] == pytest.approx(0.0, abs=0.01)


class TestEarlyGateRelease:
    """Gate drops before attack+decay completes — the core bug fix."""

    def test_no_discontinuity_during_attack(self):
        # Attack = 0.1s, decay = 0.1s, but gate only lasts 0.05s
        # (mid-attack). Release should start from whatever level the
        # attack reached, not jump to sustain.
        env = adsr(0.1, 0.1, 0.5, 0.1, sample_rate=SR)
        gate = _gate(0.05, 0.3)
        out = env.trigger(gate)

        gate_off = int(0.05 * SR)
        level_at_drop = out.data[gate_off - 1]
        level_after_drop = out.data[gate_off]

        # Level at drop should be mid-attack (~0.5), not near sustain (0.5)
        # — coincidental match here, but the key test is continuity:
        assert abs(level_after_drop - level_at_drop) < 0.05

    def test_no_discontinuity_during_decay(self):
        # Attack = 0.01s (fast), decay = 0.2s, gate lasts 0.05s.
        # At 0.05s we're mid-decay, level ~0.85 (between 1.0 and 0.5).
        # Release must start from ~0.85, not jump to 0.5.
        env = adsr(0.01, 0.2, 0.5, 0.1, sample_rate=SR)
        gate = _gate(0.05, 0.3)
        out = env.trigger(gate)

        gate_off = int(0.05 * SR)
        level_at_drop = out.data[gate_off - 1]
        level_after_drop = out.data[gate_off]

        # Must be continuous — no jump
        assert abs(level_after_drop - level_at_drop) < 0.05
        # And the level should be above sustain (we're still decaying)
        assert level_at_drop > 0.6

    def test_release_reaches_zero_after_early_drop(self):
        env = adsr(0.1, 0.1, 0.5, 0.1, sample_rate=SR)
        gate = _gate(0.05, 0.3)
        out = env.trigger(gate)
        # After release finishes, should be at zero
        assert out.data[-1] == pytest.approx(0.0, abs=0.01)

    def test_release_shape_preserved(self):
        # With a curved release, the shape should be proportionally
        # scaled, not replaced with a linear ramp. Both releases should
        # have the same normalised curve (just different amplitudes).
        env = Envelope([
            Segment(0.01, 0.0, 1.0),           # attack
            Segment(0.2, 1.0, 0.6),             # decay
            Segment(0.1, 0.6, 0.0, curve=2.0),  # convex release
        ], sustain_node=1, sample_rate=SR)

        # Normal release (gate held through full A+D)
        gate_normal = _gate(0.25, 0.4)
        out_normal = env.trigger(gate_normal)

        # Early release (gate drops mid-decay)
        gate_early = _gate(0.05, 0.3)
        out_early = env.trigger(gate_early)

        # Both releases should end at zero
        assert out_normal.data[-1] == pytest.approx(0.0, abs=0.01)
        assert out_early.data[-1] == pytest.approx(0.0, abs=0.01)

        # Normalise both release curves to [0, 1] and compare shape.
        # They should be identical since scaling preserves the curve.
        gate_off_normal = int(0.25 * SR)
        gate_off_early = int(0.05 * SR)
        rel_len = int(0.1 * SR)
        normal_release = out_normal.data[gate_off_normal:gate_off_normal + rel_len]
        early_release = out_early.data[gate_off_early:gate_off_early + rel_len]
        norm_normal = normal_release / normal_release[0]
        norm_early = early_release / early_release[0]
        np.testing.assert_allclose(norm_early, norm_normal, atol=0.01)


class TestRetriggerContinuity:
    """A new gate rising during release should continue from current level."""

    def test_no_discontinuity_on_retrigger_during_release(self):
        # Release = 0.2s. Gate drops, then retriggers 0.05s into release.
        # Level at retrigger is mid-release (~0.5ish). Attack should start
        # from there, not jump down to 0.
        env = adsr(0.05, 0.05, 0.7, 0.2, sample_rate=SR)
        n = int(0.5 * SR)
        data = np.zeros(n, dtype=np.float32)
        data[:int(0.15 * SR)] = 1.0           # first note
        data[int(0.20 * SR):int(0.40 * SR)] = 1.0  # retrigger mid-release
        gate = Signal(data, SR)
        out = env.trigger(gate)

        retrigger = int(0.20 * SR)
        level_before = out.data[retrigger - 1]
        level_after = out.data[retrigger]

        # Must be continuous — no jump
        assert abs(level_after - level_before) < 0.05
        # Level should be above zero (we're mid-release, not finished)
        assert level_before > 0.1

    def test_retrigger_still_reaches_peak(self):
        # Even when starting from a non-zero level, the attack should
        # still reach the segment's end value (1.0).
        env = adsr(0.05, 0.05, 0.7, 0.2, sample_rate=SR)
        n = int(0.5 * SR)
        data = np.zeros(n, dtype=np.float32)
        data[:int(0.15 * SR)] = 1.0
        data[int(0.20 * SR):int(0.40 * SR)] = 1.0
        gate = Signal(data, SR)
        out = env.trigger(gate)

        # After retrigger attack completes (0.20 + 0.05 = 0.25s), should be ~1.0
        peak_idx = int(0.25 * SR) - 1
        assert out.data[peak_idx] == pytest.approx(1.0, abs=0.05)

    def test_retrigger_from_silence_is_normal(self):
        # If release has fully completed (level=0), retrigger is identical
        # to a fresh start — no remapping needed.
        env = adsr(0.05, 0.05, 0.7, 0.1, sample_rate=SR)
        n = int(0.5 * SR)
        data = np.zeros(n, dtype=np.float32)
        data[:int(0.10 * SR)] = 1.0
        # Release finishes by 0.20s. Retrigger at 0.30s from silence.
        data[int(0.30 * SR):int(0.45 * SR)] = 1.0
        gate = Signal(data, SR)
        out = env.trigger(gate)

        retrigger = int(0.30 * SR)
        # Should start from ~0
        assert out.data[retrigger] == pytest.approx(0.0, abs=0.02)
        # And ramp up normally
        assert out.data[retrigger + int(0.05 * SR) - 1] == pytest.approx(1.0, abs=0.05)

    def test_retrigger_attack_shape_preserved(self):
        # A retrigger from mid-release should produce a compressed
        # version of the same attack curve, not a different shape.
        env = adsr(0.05, 0.05, 0.7, 0.2, sample_rate=SR)

        # Normal attack from silence
        gate_normal = _gate(0.2, 0.3)
        out_normal = env.trigger(gate_normal)
        attack_normal = out_normal.data[:int(0.05 * SR)]

        # Retrigger from mid-release
        n = int(0.5 * SR)
        data = np.zeros(n, dtype=np.float32)
        data[:int(0.15 * SR)] = 1.0
        data[int(0.20 * SR):int(0.40 * SR)] = 1.0
        gate = Signal(data, SR)
        out = env.trigger(gate)

        retrigger = int(0.20 * SR)
        attack_retrigger = out.data[retrigger:retrigger + int(0.05 * SR)]

        # Normalise both to [0, 1] and compare shape
        norm_normal = (attack_normal - attack_normal[0]) / (attack_normal[-1] - attack_normal[0])
        norm_retrigger = (attack_retrigger - attack_retrigger[0]) / (attack_retrigger[-1] - attack_retrigger[0])
        np.testing.assert_allclose(norm_retrigger, norm_normal, atol=0.02)


class TestOneShotEnvelope:
    """One-shot (no sustain_node) should be unaffected by the change."""

    def test_one_shot_plays_through(self):
        env = Envelope([
            Segment(0.05, 0.0, 1.0),
            Segment(0.05, 1.0, 0.0),
        ], sustain_node=None, sample_rate=SR)

        gate = _gate(0.2, 0.2)
        out = env.trigger(gate)
        # Peak should reach ~1.0
        assert np.max(out.data) == pytest.approx(1.0, abs=0.05)
        # Should decay back to ~0 after 0.1s
        assert out.data[int(0.1 * SR) - 1] == pytest.approx(0.0, abs=0.05)
