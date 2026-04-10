from __future__ import annotations

import numpy as np

from pysynth._core import Signal


def pan(signal: Signal, position: float) -> Signal:
    """Apply equal-power stereo panning to a signal.

    Converts a mono signal to stereo, or adjusts the balance of a stereo
    signal, using the constant-power (sine/cosine) pan law so that the
    perceived loudness stays constant across the stereo field.

    Parameters
    ----------
    signal:
        Input signal. Mono or stereo.
    position:
        Pan position in [-1, 1]. -1 = full left, 0 = centre, +1 = full right.

    Returns a new stereo Signal with shape (n_samples, 2).
    """
    position = float(np.clip(position, -1.0, 1.0))
    # Map [-1, 1] to [0, π/2]; equal-power law: L=cos(θ), R=sin(θ)
    angle = (position + 1.0) * np.pi / 4.0
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    if signal.n_channels == 1:
        mono = signal.data
        stereo = np.stack([mono * left_gain, mono * right_gain], axis=-1)
    else:
        # Stereo input: apply gain to each channel independently
        stereo = signal.data.copy()
        stereo[:, 0] *= left_gain
        stereo[:, 1] *= right_gain

    return Signal(stereo.astype(np.float32), signal.sample_rate)
