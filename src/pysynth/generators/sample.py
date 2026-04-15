from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pysynth._core import SAMPLE_RATE, Signal, Generator, _as_array


@dataclass
class Sample:
    """A recorded or synthesized audio snippet — the sample-based counterpart
    of :class:`Oscillator` and :class:`Wavetable`.

    Holds raw audio data together with optional metadata (root pitch, loop
    points) and exposes the standard ``.at(hz)`` interface for pitched
    playback through the CV/gate signal flow.

    Parameters
    ----------
    data:
        Audio samples as a float32 array.  Shape ``(n,)`` for mono or
        ``(n, 2)`` for stereo.
    sample_rate:
        Sample rate of the audio data.
    root_pitch:
        The pitch (in Hz) at which the sample was originally recorded.
        Required for pitched playback via ``.at(hz)``.
    loop_start:
        Start of the sustain loop region, in sample indices.
    loop_end:
        End of the sustain loop region, in sample indices.
    """

    data: np.ndarray
    sample_rate: int = SAMPLE_RATE
    root_pitch: float | None = None
    loop_start: int | None = None
    loop_end: int | None = None

    def __post_init__(self) -> None:
        self.data = np.array(self.data, dtype=np.float32, copy=True)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    @property
    def n_channels(self) -> int:
        return 1 if self.data.ndim == 1 else self.data.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_file(cls, path: str | Path, root_pitch: float | None = None) -> Sample:
        """Load a WAV file as a Sample.

        Parameters
        ----------
        path:
            Path to a ``.wav`` file.
        root_pitch:
            The pitch (Hz) at which the sample was recorded.
        """
        path = Path(path)
        if path.suffix.lower() not in (".wav",):
            raise ValueError(
                f"Unsupported audio format '{path.suffix}'. "
                "Only WAV files are supported."
            )
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
        else:
            data = np.asarray(data, dtype=np.float32)
        return cls(data, sr, root_pitch=root_pitch)

    @classmethod
    def from_signal(cls, signal: Signal, root_pitch: float | None = None) -> Sample:
        """Wrap an existing Signal as a Sample.

        Parameters
        ----------
        signal:
            The source signal.
        root_pitch:
            The pitch (Hz) the signal represents.
        """
        return cls(signal.data, signal.sample_rate, root_pitch=root_pitch)

    # ------------------------------------------------------------------ #
    # Conversion                                                           #
    # ------------------------------------------------------------------ #

    def as_signal(self) -> Signal:
        """Return the raw audio as a :class:`Signal` (copy of data)."""
        return Signal(self.data.copy(), self.sample_rate)

    # ------------------------------------------------------------------ #
    # Slicing                                                              #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: slice) -> Sample:
        """Slice by time in seconds: ``sample[0.5:1.2]``."""
        sliced = self.as_signal()[key]
        return Sample(
            sliced.data,
            self.sample_rate,
            root_pitch=self.root_pitch,
        )

    # ------------------------------------------------------------------ #
    # Processing                                                           #
    # ------------------------------------------------------------------ #

    def normalize(self, mode: str = "peak") -> Sample:
        """Return a peak-normalized copy.

        Parameters
        ----------
        mode:
            Normalization mode.  Currently only ``"peak"`` is supported.
        """
        normalized = self.as_signal().normalize(mode)
        return Sample(normalized.data, self.sample_rate, root_pitch=self.root_pitch,
                      loop_start=self.loop_start, loop_end=self.loop_end)

    # ------------------------------------------------------------------ #
    # Rendering — the .at() pattern                                        #
    # ------------------------------------------------------------------ #

    def at(self, hz: float | Signal | None = None) -> Generator:
        """Bind a playback pitch, returning a :class:`Generator`.

        Parameters
        ----------
        hz:
            Target frequency.  When ``None``, the sample plays back at its
            original rate.  When a float or Signal, the sample is resampled
            to transpose from ``root_pitch`` to ``hz`` (requires
            ``root_pitch`` to be set).
        """
        sample_data = self.data.astype(np.float64)
        sr_orig = self.sample_rate
        root = self.root_pitch
        loop_s = self.loop_start
        loop_e = self.loop_end
        n_src = len(self)

        if hz is not None and root is None:
            raise ValueError(
                "root_pitch must be set to use .at(hz) with a frequency. "
                "Use .at() for original-rate playback, or set root_pitch."
            )

        def render(dur: float, sr: int = SAMPLE_RATE) -> Signal:
            n_out = int(dur * sr)

            if hz is None:
                # Original-rate playback — resample if sample rates differ,
                # otherwise just copy/trim/pad.
                rate = sr_orig / sr
                indices = np.arange(n_out, dtype=np.float64) * rate
            elif isinstance(hz, Signal):
                hz_arr = _as_array(hz, n_out)
                ratio_arr = hz_arr / root
                # Phase accumulation: each sample advances by ratio * (sr_orig/sr)
                indices = np.cumsum(ratio_arr) * (sr_orig / sr)
            else:
                ratio = float(hz) / root
                indices = np.arange(n_out, dtype=np.float64) * ratio * (sr_orig / sr)

            # Loop-point wrapping or clamping
            if loop_s is not None and loop_e is not None and loop_e > loop_s:
                loop_len = loop_e - loop_s
                mask = indices >= loop_e
                if np.any(mask):
                    indices[mask] = loop_s + (indices[mask] - loop_s) % loop_len
            else:
                # Zero-pad beyond sample end
                pass

            if sample_data.ndim == 1:
                src_x = np.arange(n_src, dtype=np.float64)
                out = np.interp(indices, src_x, sample_data, left=0.0, right=0.0)
                return Signal(out.astype(np.float32), sr)
            else:
                # Stereo: resample each channel independently
                src_x = np.arange(n_src, dtype=np.float64)
                channels = []
                for ch in range(sample_data.shape[1]):
                    ch_data = np.interp(indices, src_x, sample_data[:, ch],
                                        left=0.0, right=0.0)
                    channels.append(ch_data)
                out = np.column_stack(channels)
                return Signal(out.astype(np.float32), sr)

        name = f"Sample.at({hz})" if hz is not None else "Sample.at()"
        return Generator(render, name=name)

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        channels = "mono" if self.n_channels == 1 else f"{self.n_channels}ch"
        pitch = f", root={self.root_pitch}Hz" if self.root_pitch else ""
        return f"Sample({self.duration:.3f}s, {channels}, {self.sample_rate}Hz{pitch})"
