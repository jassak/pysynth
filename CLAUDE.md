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

# Run the module (plays audio)
uv run python -c "import pysynth"

# Interactive development
uv run ipython
```

## Architecture

This is an early-stage Python audio synthesis library. All source code lives in `src/pysynth/__init__.py`.

**Core building blocks:**
- `hammond_wave(hz, peak, n_samples)` — generates a Hammond organ-style waveform by summing sine waves at a fixed set of harmonic ratios with preset amplitudes, returning an `int16` numpy array
- `adsr(attack, decay, sustain, sustain_level, release)` — returns a float envelope array; all time parameters are in seconds and converted to samples using `SAMPLE_RATE = 44100`

**Audio pipeline pattern:** generate wave(s) → apply ADSR envelope → play via `sounddevice`. The module currently contains top-level script code that plays a sound on import — this is experimental/demo code, not library API.

Uses `uv` for package management (see `pyproject.toml`). Python 3.14+ required.
