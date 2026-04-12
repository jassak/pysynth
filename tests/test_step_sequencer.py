import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.music.step_sequencer import Step, StepSequencer


SR = 1000
BPM = 120.0
STEP_LEN = 0.25  # beats per step


def _step_samples(n_steps=1):
    """Number of samples for *n_steps* at default BPM/step_length."""
    return int(n_steps * STEP_LEN * 60.0 / BPM * SR)


class TestStep:
    def test_active_step(self):
        s = Step(440.0)
        assert not s.is_rest
        assert not s.is_tie
        assert s.value == 440.0

    def test_rest(self):
        s = Step.rest()
        assert s.is_rest
        assert not s.is_tie

    def test_tie(self):
        s = Step.tie()
        assert s.is_tie
        assert not s.is_rest

    def test_repr_active(self):
        assert "440" in repr(Step(440.0))

    def test_repr_rest(self):
        assert "rest" in repr(Step.rest())

    def test_repr_tie(self):
        assert "tie" in repr(Step.tie())

    def test_repr_slide(self):
        assert "slide" in repr(Step(440.0, slide=True))


class TestStepSequencerCv:
    def test_returns_two_signals(self):
        val, gate = StepSequencer([Step(1.0)], bpm=BPM).cv(sample_rate=SR)
        assert isinstance(val, Signal)
        assert isinstance(gate, Signal)

    def test_duration(self):
        steps = [Step(1.0)] * 4
        val, gate = StepSequencer(steps, bpm=BPM).cv(sample_rate=SR)
        expected = 4 * STEP_LEN * 60.0 / BPM
        assert abs(val.duration - expected) < 0.01
        assert abs(gate.duration - expected) < 0.01

    def test_value_filled(self):
        val, _ = StepSequencer([Step(440.0)], bpm=BPM).cv(sample_rate=SR)
        mid = _step_samples() // 2
        assert val.data[mid] == pytest.approx(440.0)

    def test_gate_respects_gate_length(self):
        steps = [Step(440.0, gate_length=0.5)]
        _, gate = StepSequencer(steps, bpm=BPM).cv(sample_rate=SR)
        total = _step_samples()
        # First half should be gated
        quarter = total // 4
        assert gate.data[quarter] == pytest.approx(1.0)
        # Last quarter should be zero
        assert gate.data[total - 1] == pytest.approx(0.0)

    def test_rest_produces_zero(self):
        steps = [Step(440.0), Step.rest(), Step(880.0)]
        val, gate = StepSequencer(steps, bpm=BPM).cv(sample_rate=SR)
        rest_mid = _step_samples() + _step_samples() // 2
        assert val.data[rest_mid] == 0.0
        assert gate.data[rest_mid] == 0.0

    def test_tie_holds_value(self):
        steps = [Step(440.0), Step.tie(), Step(880.0)]
        val, gate = StepSequencer(steps, bpm=BPM).cv(sample_rate=SR)
        tie_mid = _step_samples() + _step_samples() // 2
        assert val.data[tie_mid] == pytest.approx(440.0)
        assert gate.data[tie_mid] == pytest.approx(1.0)

    def test_tie_no_gate_retrigger(self):
        steps = [Step(440.0, gate_length=1.0), Step.tie()]
        _, gate = StepSequencer(steps, bpm=BPM).cv(sample_rate=SR)
        # Gate should be 1.0 across the boundary
        boundary = _step_samples()
        assert gate.data[boundary - 1] == pytest.approx(1.0)
        assert gate.data[boundary] == pytest.approx(1.0)

    def test_repeats(self):
        steps = [Step(440.0)]
        val, _ = StepSequencer(steps, bpm=BPM).cv(repeats=3, sample_rate=SR)
        expected = 3 * STEP_LEN * 60.0 / BPM
        assert abs(val.duration - expected) < 0.01

    def test_repeats_values_cycle(self):
        steps = [Step(440.0), Step(880.0)]
        val, _ = StepSequencer(steps, bpm=BPM).cv(repeats=2, sample_rate=SR)
        # Third step (start of repeat) should be 440 again
        mid = _step_samples(2) + _step_samples() // 2
        assert val.data[mid] == pytest.approx(440.0)

    def test_slide_creates_ramp(self):
        steps = [Step(200.0, gate_length=1.0), Step(400.0, slide=True)]
        seq = StepSequencer(steps, bpm=BPM, slide_time=0.05)
        val, _ = seq.cv(sample_rate=SR)
        boundary = _step_samples()
        # Just before boundary: should be ramping (between 200 and 400)
        sample = boundary - 10
        assert 200.0 < val.data[sample] < 400.0

    def test_slide_endpoints(self):
        steps = [Step(200.0, gate_length=1.0), Step(400.0, slide=True)]
        seq = StepSequencer(steps, bpm=BPM, slide_time=0.05)
        val, _ = seq.cv(sample_rate=SR)
        boundary = _step_samples()
        slide_samples = int(0.05 * SR)
        ramp_start = boundary - slide_samples
        # Start of ramp should be close to 200
        assert val.data[ramp_start] == pytest.approx(200.0, abs=5.0)
        # At boundary, should be 400 (filled by the step itself)
        assert val.data[boundary] == pytest.approx(400.0)

    def test_empty_sequence(self):
        val, gate = StepSequencer([], bpm=BPM).cv(sample_rate=SR)
        assert len(val.data) == 0
        assert len(gate.data) == 0

    def test_sample_rate_respected(self):
        val, _ = StepSequencer([Step(1.0)], bpm=BPM).cv(sample_rate=2000)
        assert val.sample_rate == 2000


class TestStepSequencerRotate:
    def test_rotate_by_one(self):
        steps = [Step(1.0), Step(2.0), Step(3.0)]
        rotated = StepSequencer(steps, bpm=BPM).rotate(1)
        assert rotated.steps[0].value == 2.0
        assert rotated.steps[2].value == 1.0

    def test_rotate_empty(self):
        rotated = StepSequencer([], bpm=BPM).rotate(1)
        assert rotated.steps == []

    def test_rotate_preserves_params(self):
        seq = StepSequencer([Step(1.0)], bpm=140, step_length=0.5, slide_time=0.03)
        rotated = seq.rotate(0)
        assert rotated.bpm == 140
        assert rotated.step_length == 0.5
        assert rotated.slide_time == 0.03


class TestStepSequencerFromValues:
    def test_basic(self):
        seq = StepSequencer.from_values([440.0, 880.0], bpm=BPM)
        assert len(seq.steps) == 2
        assert seq.steps[0].value == 440.0

    def test_none_becomes_rest(self):
        seq = StepSequencer.from_values([440.0, None, 880.0])
        assert seq.steps[1].is_rest

    def test_kwargs_passed(self):
        seq = StepSequencer.from_values([1.0], bpm=140, step_length=0.5)
        assert seq.bpm == 140
        assert seq.step_length == 0.5
