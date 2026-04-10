from __future__ import annotations

from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from pysynth._core import Signal


def play(signal: Signal, blocking: bool = True) -> None:
    """Play a signal through the default audio output device.

    The signal is clipped to [-1, 1] and converted to float32 before playback.
    This is the only point in the pipeline where amplitude is bounded.
    """
    data = np.clip(signal.data, -1.0, 1.0).astype(np.float32)
    sd.play(data, samplerate=signal.sample_rate)
    if blocking:
        sd.wait()


def render_to_wav(signal: Signal, path: str | Path) -> None:
    """Write a signal to a WAV file as 16-bit PCM.

    The signal is normalised to [-1, 1] then quantised to int16. This is the
    only point in the pipeline where float32 is converted to int16.

    Parameters
    ----------
    signal:
        The signal to write.
    path:
        Output file path. The ``.wav`` extension is appended if absent.
    """
    path = Path(path)
    if path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")

    data = np.clip(signal.data, -1.0, 1.0)
    as_int16 = (data * 32767).astype(np.int16)
    wavfile.write(str(path), signal.sample_rate, as_int16)
