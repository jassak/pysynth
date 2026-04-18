"""Tests for the presets.sounds module."""

import numpy as np

from pysynth import Signal, SAMPLE_RATE, Pitch, Oscillator
from pysynth.envelopes import adsr
from pysynth.music import Note
from pysynth.instruments import Sequencer

from presets.sounds import (
    Patch,
    organ, electric_piano,
    acid_bass, sub_bass, detuned_saw,
    ambient_pad, string_pad,
    pluck, bell, fm_bass, fm_metal,
    TR808, TR909,
)
from presets.scales import major
from presets.pitches import C


class TestPatchBasics:
    def test_returns_patch(self):
        assert isinstance(organ(), Patch)

    def test_render_simple(self):
        sig = organ().render(440, dur=0.5)
        assert isinstance(sig, Signal)
        assert len(sig.data) > 0

    def test_render_cv_gate(self):
        notes = [Note(Pitch(220), 0.5), Note(Pitch(330), 0.5)]
        pitch, gate = Sequencer(notes, bpm=120).cv()
        sig = organ().render(pitch, gate)
        assert isinstance(sig, Signal)
        assert len(sig.data) > 0

    def test_render_requires_dur_without_gate(self):
        import pytest
        with pytest.raises(ValueError, match="dur is required"):
            organ().render(440)

    def test_repr(self):
        r = repr(organ())
        assert "organ" in r
        assert r == "Patch('organ')"


class TestKeyPresets:
    def test_organ_defaults(self):
        sig = organ().render(220, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_organ_custom_harmonics(self):
        sig = organ(harmonics=(1.0, 0.0)).render(220, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_electric_piano(self):
        sig = electric_piano().render(440, dur=0.5)
        assert not np.all(sig.data == 0)


class TestBassPresets:
    def test_acid_bass(self):
        sig = acid_bass().render(55, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_sub_bass(self):
        sig = sub_bass().render(55, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_detuned_saw(self):
        sig = detuned_saw().render(110, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_detuned_saw_custom_detune(self):
        sig = detuned_saw(detune=0.01).render(110, dur=0.5)
        assert not np.all(sig.data == 0)


class TestPadPresets:
    def test_ambient_pad(self):
        sig = ambient_pad().render(220, dur=1.0)
        assert not np.all(sig.data == 0)

    def test_string_pad(self):
        sig = string_pad(n_layers=3).render(220, dur=1.0)
        assert not np.all(sig.data == 0)


class TestLeadPresets:
    def test_pluck(self):
        sig = pluck().render(440, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_bell(self):
        sig = bell().render(440, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_bell_custom_mod(self):
        sig = bell(mod_ratio=3.0, mod_index=5.0).render(440, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_fm_bass(self):
        sig = fm_bass().render(55, dur=0.5)
        assert not np.all(sig.data == 0)

    def test_fm_metal(self):
        sig = fm_metal().render(200, dur=0.5)
        assert not np.all(sig.data == 0)


class TestCVGateWorkflow:
    def test_organ_with_scale_and_sequencer(self):
        """Full workflow: scale -> notes -> sequencer -> patch.render()."""
        s = major(C)
        notes = [Note(s[i], 0.25) for i in [0, 2, 4]]
        pitch, gate = Sequencer(notes, bpm=120).cv()
        sig = organ().render(pitch, gate)
        assert isinstance(sig, Signal)
        assert len(sig.data) > 0

    def test_bell_with_cv_gate(self):
        """FM sounds work with CV/gate too."""
        notes = [Note(Pitch(440), 0.5), Note(Pitch(550), 0.5)]
        pitch, gate = Sequencer(notes, bpm=120).cv()
        sig = bell().render(pitch, gate)
        assert isinstance(sig, Signal)
        assert len(sig.data) > 0


class TestPercussion808:
    def test_kick(self):
        sig = TR808.kick()
        assert isinstance(sig, Signal)
        assert not np.all(sig.data == 0)

    def test_kick_custom_decay(self):
        short = TR808.kick(decay=0.3)
        long = TR808.kick(decay=1.0)
        assert len(short.data) < len(long.data)

    def test_snare(self):
        sig = TR808.snare()
        assert isinstance(sig, Signal)
        assert not np.all(sig.data == 0)

    def test_hihat_closed(self):
        sig = TR808.hihat(open=False)
        assert isinstance(sig, Signal)

    def test_hihat_open(self):
        sig = TR808.hihat(open=True)
        assert isinstance(sig, Signal)
        # open hat should be longer
        assert len(sig.data) > len(TR808.hihat(open=False).data)

    def test_clap(self):
        sig = TR808.clap()
        assert isinstance(sig, Signal)

    def test_cowbell(self):
        sig = TR808.cowbell()
        assert isinstance(sig, Signal)

    def test_tom(self):
        sig = TR808.tom()
        assert isinstance(sig, Signal)

    def test_rimshot(self):
        sig = TR808.rimshot()
        assert isinstance(sig, Signal)


class TestPercussion909:
    def test_kick(self):
        sig = TR909.kick()
        assert isinstance(sig, Signal)
        assert not np.all(sig.data == 0)

    def test_snare(self):
        sig = TR909.snare()
        assert isinstance(sig, Signal)

    def test_hihat_closed(self):
        sig = TR909.hihat(open=False)
        assert isinstance(sig, Signal)

    def test_hihat_open(self):
        sig = TR909.hihat(open=True)
        assert isinstance(sig, Signal)

    def test_clap(self):
        sig = TR909.clap()
        assert isinstance(sig, Signal)

    def test_ride(self):
        sig = TR909.ride()
        assert isinstance(sig, Signal)
