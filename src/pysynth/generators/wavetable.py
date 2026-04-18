from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal, _as_array
from pysynth.generators.oscillators import Waveform, _shape


class Wavetable:
    """Wavetable oscillator with position morphing between stored waveforms.

    A Wavetable stores one or more single-cycle waveforms as fixed-size arrays.
    During rendering, phase accumulates based on pitch and samples are read via
    interpolated table lookup.  The ``position`` parameter selects which
    waveform (or blend of adjacent waveforms) to read from, enabling smooth
    timbral evolution when driven by a time-varying Signal.

    Examples::

        # Morph from sine through saw to square
        wt = Wavetable.from_waveforms(["sine", "saw", "square"])
        sig = wt.render(2.0, 440, position=1.0)   # pure saw

        # Time-varying position via an LFO
        lfo = Oscillator("triangle").render(2.0, 0.5) * 0.5 + 0.5
        sig = wt.render(2.0, 440, position=lfo)
    """

    _tables: np.ndarray   # shape (n_tables, table_size), float64
    _table_size: int
    _n_tables: int

    def __init__(
        self,
        tables: list[np.ndarray],
        table_size: int = 2048,
    ) -> None:
        if not tables:
            raise ValueError("tables must be a non-empty list of arrays")
        if table_size < 2:
            raise ValueError("table_size must be at least 2")

        resampled = []
        for i, tbl in enumerate(tables):
            arr = np.asarray(tbl, dtype=np.float64).ravel()
            if len(arr) == 0:
                raise ValueError(f"table {i} is empty")
            if len(arr) != table_size:
                x_old = np.linspace(0.0, 1.0, len(arr))
                x_new = np.linspace(0.0, 1.0, table_size)
                arr = np.interp(x_new, x_old, arr)
            resampled.append(arr)

        self._tables = np.array(resampled, dtype=np.float64)
        self._table_size = table_size
        self._n_tables = len(resampled)

    @classmethod
    def from_waveforms(
        cls,
        waveforms: list[Waveform],
        table_size: int = 2048,
    ) -> Wavetable:
        """Build a Wavetable from built-in waveform shapes."""
        phase = np.linspace(0.0, 2.0 * np.pi, table_size, endpoint=False)
        tables = [_shape(w, phase) for w in waveforms]
        return cls(tables, table_size=table_size)

    @classmethod
    def from_sample(
        cls,
        sample: Sample,
        n_frames: int,
        table_size: int = 2048,
    ) -> Wavetable:
        """Build a Wavetable by slicing a Sample into single-cycle frames.

        The sample is divided into *n_frames* equal segments, each resampled
        to *table_size*.  This allows loading wavetable ``.wav`` files
        (e.g. Serum / Vital format) directly.

        Parameters
        ----------
        sample:
            Source audio to slice.  Stereo samples are mixed to mono first.
        n_frames:
            Number of single-cycle frames to extract.
        table_size:
            Number of samples per frame after resampling.
        """
        from pysynth.generators.sample import Sample as _Sample  # avoid circular at module level

        data = sample.data.astype(np.float64)
        if data.ndim == 2:
            data = data.mean(axis=1)
        frame_len = len(data) // n_frames
        if frame_len == 0:
            raise ValueError(
                f"Sample has {len(data)} samples, too short for {n_frames} frames"
            )
        tables = []
        for i in range(n_frames):
            start = i * frame_len
            end = start + frame_len
            tables.append(data[start:end])
        return cls(tables, table_size=table_size)

    @property
    def n_tables(self) -> int:
        return self._n_tables

    @property
    def table_size(self) -> int:
        return self._table_size

    def render(
        self,
        dur: float,
        hz: float | Signal,
        sr: int = SAMPLE_RATE,
        *,
        position: float | Signal = 0.0,
    ) -> Signal:
        """Render the wavetable at the given frequency.

        Parameters
        ----------
        dur:
            Duration in seconds.
        hz:
            Frequency in Hz.  Accepts a constant float or a time-varying
            Signal (pitch CV, FM modulation).
        sr:
            Sample rate.
        position:
            Position in the wavetable, range ``[0, n_tables - 1]``.
            Integer values select an exact table; fractional values crossfade
            between adjacent tables.  Accepts a float or a time-varying Signal.
        """
        tables = self._tables
        tsize = self._table_size
        n_tables = self._n_tables
        n = int(dur * sr)

        # --- phase accumulation (same logic as _render_component) ---
        if isinstance(hz, Signal):
            freq_data = hz.data.astype(np.float64)
            if len(freq_data) < n:
                freq_data = np.pad(freq_data, (0, n - len(freq_data)))
            else:
                freq_data = freq_data[:n]
            phase_arr = 2.0 * np.pi * np.cumsum(freq_data) / sr
        else:
            t = np.arange(n, dtype=np.float64) / sr
            phase_arr = 2.0 * np.pi * float(hz) * t

        # --- normalize phase to float table index ---
        table_phase = (phase_arr / (2.0 * np.pi)) % 1.0
        table_idx = table_phase * tsize           # float in [0, tsize)

        # --- sample interpolation indices ---
        idx0 = np.floor(table_idx).astype(np.intp) % tsize
        idx1 = (idx0 + 1) % tsize
        frac = table_idx - np.floor(table_idx)

        # --- position (which table / crossfade) ---
        pos_arr = _as_array(position, n)
        pos_arr = np.clip(pos_arr, 0.0, n_tables - 1)

        tbl0 = np.floor(pos_arr).astype(np.intp)
        tbl1 = np.minimum(tbl0 + 1, n_tables - 1)
        pos_frac = pos_arr - np.floor(pos_arr)

        # --- bilinear interpolation (vectorized) ---
        v00 = tables[tbl0, idx0]
        v01 = tables[tbl0, idx1]
        v10 = tables[tbl1, idx0]
        v11 = tables[tbl1, idx1]

        interp_low = v00 + frac * (v01 - v00)
        interp_high = v10 + frac * (v11 - v10)
        output = interp_low + pos_frac * (interp_high - interp_low)

        return Signal(output.astype(np.float32), sr)

    def __repr__(self) -> str:
        return f"Wavetable(tables={self._n_tables}, size={self._table_size})"
