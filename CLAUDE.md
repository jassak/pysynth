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

A Python audio synthesis library built around a few mathematical abstractions and a CV/gate signal flow model. Python 3.14+, `uv` for package management.

### Core types (`src/pysynth/_core.py`)

- **`Signal`** — a commutative ℝ-algebra over float32 audio arrays. `Signal + Signal` (mix, zero-pads shorter), `Signal * Signal` (pointwise product, truncates to shorter), `Signal + scalar` (DC offset), `Signal * scalar` (gain), `-Signal` (phase inversion). `.play()`, `.plot()`, `.silence()`.
- **`Effect`** — base class; subclasses implement `__call__(signal) -> Signal`. The `|` operator chains effects left-to-right: `LowPassFilter(800) | Reverb(0.5)`.

### Generators (`src/pysynth/generators/`)

**`Oscillator`** is pitch-free and algebraically composable:
- `Oscillator("sine", ratio, phase)` — defines a waveform template; `ratio` is a relative frequency multiplier
- `osc1 + osc2` — additive synthesis (merges component lists)
- `osc * scalar` — scale amplitude
- `osc.render(dur, hz) -> Signal` — renders at the given frequency; `hz` can be a float or a time-varying `Signal` (pitch CV, vibrato, FM)

### Envelopes (`src/pysynth/envelopes/`)

- **`Segment(duration, start, end, curve)`** — a single envelope segment with configurable curvature
- **`Envelope(segments, sustain_node)`** — ordered list of Segments.
  - `.render(duration?) -> Signal` — one-shot render
  - `.apply(signal) -> Signal` — multiply signal by envelope
  - `.trigger(gate) -> Signal` — gate-triggered render: re-triggers on rising edges, sustains while gate is high, releases when gate falls
- **`adsr(attack, decay, sustain, release) -> Envelope`** — convenience constructor with `sustain_node=1`; sustain is the level held while gate is high

Modulation signals (LFO-style) are expressed directly via Signal arithmetic: `Oscillator("sine").render(dur, rate) * depth + offset`.

### Effects (`src/pysynth/effects/`)

All inherit `Effect`, all chainable via `|`: `LowPassFilter`, `HighPassFilter`, `BandPassFilter`, `Gain`, `Compressor`, `Limiter`, `SimpleReverb`, `DatorroReverb`, `Delay`, `Echo`, `Tanh`, `Clip`, `Overdrive`.

### Music (`src/pysynth/music/`)

- **`Pitch(hz)`** — a torsor over the multiplicative group of positive reals: `pitch * ratio`, `pitch / pitch -> float`, `.up(cents)`
- **`Scale(tonic, intervals, unit="ratio"|"cents")`** — unbiased (no octave equivalence assumed); `scale[n] -> Pitch`
- **`Note(pitch, duration, velocity)`** — duration in **beats**; `Note.rest(duration)` for silence
- **`Sequencer(notes, bpm)`** — `.cv(repeats=1) -> (pitch: Signal, gate: Signal)` — produces pitch CV and gate control signals for the entire sequence
- **`Arpeggiator`** — builds a note sequence from a chord + pattern, delegates to Sequencer
- **`PolySequencer(events, n_parts, bpm)`** — polyphonic part-allocated sequencer
  - Takes piano-roll events: `list[(onset_beat, Note)]`
  - `.cv() -> list[(pitch: Signal, gate: Signal)]` — one CV/gate pair per part, all same duration
  - `.from_chords(chords)` — convenience for block chord progressions
  - Part allocation: first-free-part with most-recently-used preference; steals earliest-ending part when full

#### CV/gate signal flow

The Sequencer outputs control signals, not audio. Composition happens at the Signal level:

```python
# Monophonic
pitch, gate = Sequencer(notes, bpm=120).cv()
audio  = Oscillator("saw").render(pitch.duration, pitch)
amp    = adsr(0.01, 0.1, 0.7, 0.1).trigger(gate)
cutoff = adsr(0.005, 0.2, 0.0, 0.05).trigger(gate) * 4000 + 300
output = LowPassFilter(cutoff)(audio) * amp

# Polyphonic
parts = PolySequencer.from_chords(chords, n_parts=4, bpm=120).cv()
audio = sum(
    Oscillator("saw").render(p.duration, p) * adsr(0.01, 0.1, 0.7, 0.1).trigger(g)
    for p, g in parts
)
```

### Mixing (`src/pysynth/mixing/`)

`Mixer`, `pan()`, `play()`, `render_to_wav()`. `int16` conversion happens only here (playback/export boundary).

### Design principles

- No in-place mutation — all operations return new instances
- `int16` only at the output boundary
- Signal algebra is the composition primitive; avoid special-casing
- CV/gate signal flow: sequencers produce control signals, not audio
- Oscillator pitch is supplied at render time via `.render(dur, hz)`, never stored on the oscillator
