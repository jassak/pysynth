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

**`Generator`** (`base.py`) — abstract base class for all signal sources. All algebra lives here:
- `gen1 + gen2` — additive synthesis (sum rendered signals)
- `gen + scalar` / `scalar + gen` — DC offset
- `gen1 * gen2` — ring modulation (pointwise product)
- `gen * scalar` / `scalar * gen` — gain scaling
- `-gen` — phase inversion; `gen1 - gen2` — subtraction

Concrete generators: `Oscillator`, `WhiteNoise`, `PinkNoise`, `Wavetable`, `Sample`, `Granular`. All accept `**_kwargs` in `render()` so composites can forward unknown keyword args.

**`Oscillator`** is pitch-free:
- `Oscillator("sine", ratio, phase)` — defines a waveform template; `ratio` is a relative frequency multiplier
- `.render(dur, hz) -> Signal` — renders at the given frequency; `hz` can be a float or a time-varying `Signal` (pitch CV, vibrato, FM)

### Operators (`src/pysynth/operators.py`)

**`Operator(generator, envelope)`** — pairs a Generator with an Envelope. Same algebra as Generator (`+`, `*`, `-`) plus FM:
- `op1 << op2` — FM synthesis: op2's rendered output is added to op1's pitch
- `.render(pitch, gate) -> Signal` — renders the generator at `pitch`, applies envelope triggered by `gate`

FM topologies via explicit parenthesization:
- `carrier << (mod1 << mod2)` — cascade: mod2 → mod1 → carrier
- `carrier << (mod1 + mod2)` — parallel: both modulate carrier independently
- `<<` is non-associative: `(a << b) << c` differs from `a << (b << c)`

### Three-level algebra

`render` is a **homomorphism** at each level — composing then rendering equals rendering then composing:
- **Signal** — concrete audio, immediate operations
- **Generator** — deferred waveform templates, resolved by `.render(dur, hz)`
- **Operator** — generator + envelope pairs, resolved by `.render(pitch, gate)`

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

The Sequencer outputs control signals, not audio. Composition can happen at Signal or Operator level:

```python
# Signal-level (manual wiring)
pitch, gate = Sequencer(notes, bpm=120).cv()
audio  = Oscillator("saw").render(pitch.duration, pitch)
amp    = adsr(0.01, 0.1, 0.7, 0.1).trigger(gate)
cutoff = adsr(0.005, 0.2, 0.0, 0.05).trigger(gate) * 4000 + 300
output = LowPassFilter(cutoff)(audio) * amp

# Operator-level (each generator gets its own envelope)
pitch, gate = Sequencer(notes, bpm=120).cv()
bright = Operator(Oscillator("saw"), adsr(0.01, 0.1, 0.7, 0.3))
sub    = Operator(Oscillator("sine", ratio=0.5), adsr(0.005, 0.5, 0.9, 0.5))
output = (bright + sub).render(pitch, gate)

# FM synthesis
carrier = Operator(Oscillator("sine"), adsr(0.01, 0.1, 0.7, 0.3))
mod     = Operator(Oscillator("sine", ratio=2) * 200, adsr(0.01, 0.3, 0.0, 0.1))
output  = (carrier << mod).render(440.0, gate)

# Polyphonic
parts = PolySequencer.from_chords(chords, n_parts=4, bpm=120).cv()
voice = Operator(Oscillator("saw"), adsr(0.01, 0.1, 0.7, 0.1))
audio = sum(voice.render(p, g) for p, g in parts)
```

### Mixing (`src/pysynth/mixing/`)

`Mixer`, `pan()`, `play()`, `render_to_wav()`. `int16` conversion happens only here (playback/export boundary).

### Design principles

- No in-place mutation — all operations return new instances
- `int16` only at the output boundary
- Signal algebra is the composition primitive; avoid special-casing
- CV/gate signal flow: sequencers produce control signals, not audio
- Generator pitch is supplied at render time via `.render(dur, hz)`, never stored on the generator
- `render` is a homomorphism: `(a + b).render(...) == a.render(...) + b.render(...)` at both Generator and Operator level
