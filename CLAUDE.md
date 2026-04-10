# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/path/to/test_file.py::test_name

# Interactive development
uv run ipython
```

## Architecture

A Python audio synthesis library built around a few mathematical abstractions. Python 3.14+, `uv` for package management.

### Core types (`src/pysynth/_core.py`)

- **`Signal`** — a vector space over float32 audio arrays. Supports `+`, `*`, `-`, unary `-`; `Signal + Signal` zero-pads the shorter; `Signal + scalar` is a DC offset / FM carrier shift. `__radd__ = __add__` (commutative). `.play()`, `.plot()`, `.silence()`.
- **`Effect`** — base class; subclasses implement `__call__(signal) -> Signal`. The `|` operator chains effects left-to-right: `LowPassFilter(800) | Reverb(0.5)`.

### Generators (`src/pysynth/generators/`)

**`Oscillator`** is pitch-free and algebraically composable:
- `Oscillator("sine", ratio, phase)` — defines a waveform template; `ratio` is a relative frequency multiplier
- `osc1 + osc2` — additive synthesis (merges component lists)
- `osc * scalar` — scale amplitude
- `osc.at(hz)` — binds a frequency, returns a private `_Voice`
- `_Voice.render(dur, sample_rate) -> Signal` — performs synthesis; handles both constant `hz: float` and time-varying `hz: Signal` (via cumulative-sum phase integration)

**Generator protocol** (duck-typed): any object with `.at(hz) -> <voice>` where `<voice>` has `.render(dur, sample_rate) -> Signal`. Used by `Sequencer` and `Arpeggiator`. Custom generators (FM, etc.) implement this protocol without inheriting anything:

```python
class FMSynth:
    def at(self, hz):
        class _Voice:
            def render(self_, dur, sr=SAMPLE_RATE):
                mod = modulator.at(hz).render(dur, sr) * mod_index
                return carrier.at(hz + mod).render(dur, sr)
        return _Voice()
```

### Envelopes (`src/pysynth/envelopes/`)

- **`ADSR(attack, decay, sustain, sustain_level, release)`** — `.render(duration?) -> Signal`, `.apply(signal) -> Signal`
- **`LFO(waveform, rate, depth, offset)`** — `.render(duration) -> Signal` in `[offset-depth, offset+depth]`; used for vibrato, tremolo, FM modulation

### Effects (`src/pysynth/effects/`)

All inherit `Effect`, all chainable via `|`: `LowPassFilter`, `HighPassFilter`, `BandPassFilter`, `Gain`, `Compressor`, `Limiter`, `SimpleReverb`, `Delay`, `Echo`, `Tanh`, `Clip`, `Overdrive`.

### Music (`src/pysynth/music/`)

- **`Pitch(hz)`** — a torsor over the multiplicative group of positive reals: `pitch * ratio`, `pitch / pitch -> float`, `.up(cents)`
- **`Scale(tonic, intervals, unit="ratio"|"cents")`** — unbiased (no octave equivalence assumed); `scale[n] -> Pitch`
- **`Note(pitch, duration, velocity)`** — duration in **beats**; `Note.rest(duration)` for silence
- **`Sequencer(notes, bpm)`** — `.render(generator, envelope=None, repeats=1) -> Signal`; calls `generator.at(note.pitch.hz).render(dur_seconds, sample_rate)` per note
- **`Arpeggiator`** — delegates to Sequencer

### Mixing (`src/pysynth/mixing/`)

`Mixer`, `pan()`, `play()`, `render_to_wav()`. `int16` conversion happens only here (playback/export boundary).

### Design principles

- No in-place mutation — all operations return new instances
- `int16` only at the output boundary
- `Signal` arithmetic is the composition primitive; avoid special-casing
- Oscillator pitch is always supplied via `.at(hz)`, never stored on the oscillator
