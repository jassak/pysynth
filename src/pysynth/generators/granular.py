from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from pysynth._core import SAMPLE_RATE, Signal, _as_array
from pysynth.generators.base import Generator
from pysynth.generators.sample import Sample


class Granular(Generator):
    """Granular synthesis engine that reads overlapping grains from a Sample.

    All parameters (except ``sample`` and ``window``) accept a float for a
    constant value or a :class:`Signal` for sample-rate modulation.

    Parameters
    ----------
    sample:
        The source audio to granulate.
    position:
        Read position in the sample, normalized to ``[0, 1]``.
    grain_size:
        Duration of each grain in seconds.
    density:
        Number of grains per second.
    pitch:
        Playback rate ratio for each grain (1.0 = original speed).
    spread:
        Random jitter applied to ``position``, in the range ``[0, 1]``.
    window:
        Window function applied to each grain.  Any name accepted by
        :func:`scipy.signal.get_window`.
    seed:
        Random seed for reproducible spread jitter.  ``None`` for
        non-deterministic output.

    Examples
    --------
    ::

        sample = Sample.from_file("texture.wav")
        # Slow scan through the sample with medium grain density
        pos = Oscillator("triangle").render(4.0, 0.1) * 0.5 + 0.5
        sig = Granular(sample, position=pos, grain_size=0.06, density=25).render(4.0)
    """

    def __init__(
        self,
        sample: Sample,
        position: float | Signal = 0.5,
        grain_size: float | Signal = 0.05,
        density: float | Signal = 20.0,
        pitch: float | Signal = 1.0,
        spread: float | Signal = 0.0,
        window: str = "hann",
        seed: int | None = None,
    ) -> None:
        self._sample = sample
        self._position = position
        self._grain_size = grain_size
        self._density = density
        self._pitch = pitch
        self._spread = spread
        self._window = window
        self._seed = seed

    def render(self, dur: float, hz: float | Signal | None = None, sr: int = SAMPLE_RATE, **_kwargs) -> Signal:
        """Render granular output.

        Parameters
        ----------
        dur:
            Duration in seconds.
        hz:
            When ``None``, grains play at the rate set by the ``pitch``
            constructor parameter.  When a float or Signal, the grain
            playback rate is set to ``hz / sample.root_pitch`` (requires
            ``root_pitch`` to be set on the sample).
        sr:
            Sample rate.
        """
        sample = self._sample
        position = self._position
        grain_size = self._grain_size
        density = self._density
        pitch = self._pitch
        spread = self._spread
        window_name = self._window
        seed = self._seed

        if hz is not None:
            if sample.root_pitch is None:
                raise ValueError(
                    "sample.root_pitch must be set to render with a frequency. "
                    "Use .render(dur) for default pitch, or set root_pitch on the sample."
                )

        sample_data = sample.data.astype(np.float64)
        n_src = len(sample)
        root = sample.root_pitch

        n_out = int(dur * sr)
        rng = np.random.default_rng(seed)

        # Resolve all parameters to arrays
        pos_arr = _as_array(position, n_out)
        gs_arr = _as_array(grain_size, n_out)
        dens_arr = _as_array(density, n_out)
        spread_arr = _as_array(spread, n_out)

        if hz is not None:
            hz_arr = _as_array(hz, n_out)
            pitch_arr = hz_arr / root
        else:
            pitch_arr = _as_array(pitch, n_out)

        # Schedule grain onsets
        onsets = []
        t = 0.0
        while t < dur:
            idx = min(int(t * sr), n_out - 1)
            onsets.append(idx)
            local_density = max(dens_arr[idx], 0.1)
            t += 1.0 / local_density

        # Window cache to avoid recomputing for same grain length
        window_cache: dict[int, np.ndarray] = {}

        # Overlap-add
        if sample_data.ndim == 1:
            output = np.zeros(n_out, dtype=np.float64)
        else:
            output = np.zeros((n_out, sample_data.shape[1]), dtype=np.float64)

        src_x = np.arange(n_src, dtype=np.float64)

        for onset_idx in onsets:
            grain_len = max(int(gs_arr[onset_idx] * sr), 1)
            local_pitch = pitch_arr[onset_idx]
            local_pos = pos_arr[onset_idx]
            local_spread = spread_arr[onset_idx]

            # Apply spread jitter
            if local_spread > 0:
                jitter = local_spread * (rng.random() * 2.0 - 1.0)
                local_pos = np.clip(local_pos + jitter, 0.0, 1.0)
            else:
                local_pos = np.clip(local_pos, 0.0, 1.0)

            # Compute read indices into the source sample
            read_start = local_pos * (n_src - 1)
            read_indices = read_start + np.arange(grain_len, dtype=np.float64) * local_pitch

            # Get or create window
            if grain_len not in window_cache:
                if grain_len == 1:
                    window_cache[grain_len] = np.ones(1)
                else:
                    window_cache[grain_len] = get_window(window_name, grain_len, fftbins=False)
            win = window_cache[grain_len]

            # How much of the grain fits in the output
            end_idx = min(onset_idx + grain_len, n_out)
            actual_len = end_idx - onset_idx
            if actual_len <= 0:
                continue

            if sample_data.ndim == 1:
                grain = np.interp(
                    read_indices[:actual_len], src_x, sample_data,
                    left=0.0, right=0.0,
                )
                grain *= win[:actual_len]
                output[onset_idx:end_idx] += grain
            else:
                for ch in range(sample_data.shape[1]):
                    grain = np.interp(
                        read_indices[:actual_len], src_x, sample_data[:, ch],
                        left=0.0, right=0.0,
                    )
                    grain *= win[:actual_len]
                    output[onset_idx:end_idx, ch] += grain

        return Signal(output.astype(np.float32), sr)

    def __repr__(self) -> str:
        return (
            f"Granular(sample={self._sample!r}, "
            f"grain_size={self._grain_size}, density={self._density})"
        )
