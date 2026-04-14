import numpy as np
import pytest

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.percussion import Percussion


SR = 1000  # low sample rate for fast tests


def _sig(values, sr=SR):
    return Signal(np.array(values, dtype=np.float32), sr)


def _gate_pulse(sr, at_sample, pulse_len, velocity=1.0, total=None):
    """Build a gate Signal with a single pulse."""
    if total is None:
        total = at_sample + pulse_len + sr  # some padding
    g = np.zeros(total, dtype=np.float32)
    g[at_sample:at_sample + pulse_len] = velocity
    return Signal(g, sr)


class TestPercussionTrigger:
    def test_places_hit_at_gate_edge(self):
        hit = _sig([1.0, 0.5, 0.25])
        gate = _gate_pulse(SR, at_sample=10, pulse_len=5, total=20)
        result = Percussion(hit).trigger(gate)
        assert len(result.data) == 20
        # Hit should appear starting at sample 10
        np.testing.assert_allclose(result.data[10], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.data[11], 0.5, atol=1e-6)
        np.testing.assert_allclose(result.data[12], 0.25, atol=1e-6)
        # Before the hit should be silence
        assert np.all(result.data[:10] == 0.0)

    def test_velocity_scales_amplitude(self):
        hit = _sig([1.0, 1.0, 1.0])
        gate = _gate_pulse(SR, at_sample=0, pulse_len=5, velocity=0.5, total=10)
        result = Percussion(hit).trigger(gate)
        np.testing.assert_allclose(result.data[:3], [0.5, 0.5, 0.5], atol=1e-6)

    def test_output_duration_matches_gate(self):
        hit = _sig([1.0, 0.5])
        gate = Signal(np.zeros(500, dtype=np.float32), SR)
        gate.data[100:110] = 1.0
        result = Percussion(hit).trigger(gate)
        assert len(result.data) == 500

    def test_silent_gate_returns_silence(self):
        hit = _sig([1.0, 0.5, 0.25])
        gate = Signal(np.zeros(100, dtype=np.float32), SR)
        result = Percussion(hit).trigger(gate)
        assert np.all(result.data == 0.0)

    def test_multiple_hits(self):
        hit = _sig([1.0])
        gate = np.zeros(100, dtype=np.float32)
        gate[10:15] = 1.0
        gate[50:55] = 0.8
        gate_sig = Signal(gate, SR)
        result = Percussion(hit).trigger(gate_sig)
        np.testing.assert_allclose(result.data[10], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.data[50], 0.8, atol=1e-6)
        # Between hits should be silent
        assert np.all(result.data[11:50] == 0.0)

    def test_overlapping_hits_sum(self):
        hit = _sig([1.0, 1.0, 1.0, 1.0, 1.0])
        # Two hits 3 samples apart — they overlap
        gate = np.zeros(20, dtype=np.float32)
        gate[2:4] = 1.0
        gate[5:7] = 1.0
        gate_sig = Signal(gate, SR)
        result = Percussion(hit).trigger(gate_sig)
        # At sample 5, both hits overlap: first hit's tail + second hit's start
        np.testing.assert_allclose(result.data[5], 2.0, atol=1e-6)

    def test_hit_truncated_at_end(self):
        hit = _sig([1.0, 1.0, 1.0, 1.0, 1.0])
        gate = _gate_pulse(SR, at_sample=8, pulse_len=3, total=10)
        result = Percussion(hit).trigger(gate)
        # Only 2 samples of the hit fit (indices 8 and 9)
        assert len(result.data) == 10
        np.testing.assert_allclose(result.data[8], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.data[9], 1.0, atol=1e-6)

    def test_preserves_sample_rate(self):
        hit = _sig([1.0], sr=22050)
        gate = Signal(np.zeros(100, dtype=np.float32), 22050)
        gate.data[10:15] = 1.0
        result = Percussion(hit).trigger(gate)
        assert result.sample_rate == 22050
