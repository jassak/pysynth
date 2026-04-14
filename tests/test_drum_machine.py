import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.music.drum_machine import DrumMachine


SR = 1000  # low sample rate for fast tests


class TestDrumMachineCv:
    def test_returns_dict_of_signals(self):
        dm = DrumMachine({"a": [1, 0], "b": [0, 1]}, bpm=120)
        gates = dm.cv(sample_rate=SR)
        assert isinstance(gates, dict)
        assert set(gates.keys()) == {"a", "b"}
        assert all(isinstance(v, Signal) for v in gates.values())

    def test_all_tracks_same_duration(self):
        dm = DrumMachine({"a": [1, 0, 1, 0], "b": [1, 0]}, bpm=120)
        gates = dm.cv(sample_rate=SR)
        assert len(gates["a"].data) == len(gates["b"].data)

    def test_velocity_in_gate(self):
        dm = DrumMachine({"x": [0.7]}, bpm=120, step_length=0.25,
                         gate_length=1.0, retrigger_gap=0.0)
        gates = dm.cv(sample_rate=SR)
        g = gates["x"].data
        # The gate should contain 0.7 where active
        active = g[g > 0]
        assert len(active) > 0
        np.testing.assert_allclose(active, 0.7, atol=1e-6)

    def test_zero_is_silent(self):
        dm = DrumMachine({"x": [0, 0, 0]}, bpm=120)
        gates = dm.cv(sample_rate=SR)
        np.testing.assert_array_equal(gates["x"].data, 0.0)

    def test_repeats_doubles_duration(self):
        dm = DrumMachine({"x": [1, 0]}, bpm=120, step_length=0.25)
        g1 = dm.cv(repeats=1, sample_rate=SR)
        g2 = dm.cv(repeats=2, sample_rate=SR)
        assert len(g2["x"].data) == 2 * len(g1["x"].data)

    def test_repeats_cycles_pattern(self):
        dm = DrumMachine({"x": [1, 0]}, bpm=60, step_length=1.0,
                         gate_length=0.5, retrigger_gap=0.0)
        gates = dm.cv(repeats=2, sample_rate=SR)
        g = gates["x"].data
        # Two hits: one at start, one at the repeat
        half = len(g) // 2
        # First half and second half should be identical
        np.testing.assert_array_equal(g[:half], g[half:])


class TestDrumMachineGateLength:
    def test_global_gate_length(self):
        dm = DrumMachine({"x": [1]}, bpm=60, step_length=1.0,
                         gate_length=0.5, retrigger_gap=0.0)
        gates = dm.cv(sample_rate=SR)
        g = gates["x"].data
        step_samples = int(1.0 * 60.0 / 60.0 * SR)
        gate_samples = int(0.5 * step_samples)
        # First gate_samples should be 1.0, rest should be 0.0
        assert np.all(g[:gate_samples] == 1.0)
        assert np.all(g[gate_samples:step_samples] == 0.0)

    def test_per_track_gate_length(self):
        dm = DrumMachine({"a": [1], "b": [1]}, bpm=60, step_length=1.0,
                         gate_length=0.5, gate_lengths={"b": 0.8},
                         retrigger_gap=0.0)
        gates = dm.cv(sample_rate=SR)
        step_samples = int(1.0 * SR)
        a_active = np.sum(gates["a"].data > 0)
        b_active = np.sum(gates["b"].data > 0)
        assert b_active > a_active


class TestDrumMachineRetrigger:
    def test_consecutive_hits_get_gap(self):
        dm = DrumMachine({"x": [1, 1]}, bpm=60, step_length=1.0,
                         gate_length=1.0, retrigger_gap=0.01)
        gates = dm.cv(sample_rate=SR)
        g = gates["x"].data
        step_samples = int(1.0 * SR)
        gap_samples = int(0.01 * SR)
        # At the boundary of step 1, first gap_samples should be 0
        assert np.all(g[step_samples:step_samples + gap_samples] == 0.0)

    def test_no_gap_after_rest(self):
        dm = DrumMachine({"x": [1, 0, 1]}, bpm=60, step_length=1.0,
                         gate_length=1.0, retrigger_gap=0.01)
        gates = dm.cv(sample_rate=SR)
        g = gates["x"].data
        step_samples = int(1.0 * SR)
        # Step 2 (index 2) follows a rest — no gap needed
        assert g[2 * step_samples] == 1.0


class TestDrumMachinePadding:
    def test_shorter_patterns_padded(self):
        dm = DrumMachine({"a": [1, 0, 0, 0], "b": [1]}, bpm=120)
        gates = dm.cv(sample_rate=SR)
        assert len(gates["a"].data) == len(gates["b"].data)
        # b only has 1 hit in first step, rest should be silent
        step_dur = 0.25 * 60.0 / 120.0
        step_samples = int(step_dur * SR)
        assert np.all(gates["b"].data[step_samples:] == 0.0)


class TestDrumMachineRotate:
    def test_rotate_shifts_pattern(self):
        dm = DrumMachine({"x": [1, 0, 0, 0]}, bpm=60, step_length=1.0,
                         gate_length=0.5, retrigger_gap=0.0)
        rotated = dm.rotate(1)
        gates = rotated.cv(sample_rate=SR)
        g = gates["x"].data
        step_samples = int(1.0 * SR)
        # After rotation by 1, first step should be silent
        assert np.all(g[:step_samples] == 0.0)
        # Hit should now be in the last step
        assert np.any(g[3 * step_samples:] > 0)

    def test_rotate_preserves_params(self):
        dm = DrumMachine({"x": [1, 0]}, bpm=140, step_length=0.5,
                         gate_length=0.3, retrigger_gap=0.005)
        rotated = dm.rotate(1)
        assert rotated.bpm == 140
        assert rotated.step_length == 0.5
        assert rotated.gate_length == 0.3
        assert rotated.retrigger_gap == 0.005


class TestDrumMachineFromX0x:
    def test_x_maps_to_one(self):
        dm = DrumMachine.from_x0x({"x": "x-x-"}, bpm=120)
        assert dm.tracks["x"] == [1.0, 0.0, 1.0, 0.0]

    def test_dash_maps_to_zero(self):
        dm = DrumMachine.from_x0x({"x": "----"}, bpm=120)
        assert dm.tracks["x"] == [0.0, 0.0, 0.0, 0.0]

    def test_kwargs_passed_through(self):
        dm = DrumMachine.from_x0x({"x": "x---"}, bpm=140, step_length=0.5)
        assert dm.bpm == 140
        assert dm.step_length == 0.5


class TestDrumMachineEmpty:
    def test_empty_tracks(self):
        dm = DrumMachine({}, bpm=120)
        gates = dm.cv(sample_rate=SR)
        assert gates == {}

    def test_rotate_empty(self):
        dm = DrumMachine({}, bpm=120)
        rotated = dm.rotate(3)
        assert rotated.cv(sample_rate=SR) == {}
