from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pysynth.spectral._spectrum import Spectrum


def freeze(spectrum: Spectrum, frame: int | None = None) -> Spectrum:
    """Spectral freeze: repeat a single frame across the entire duration.

    Parameters
    ----------
    frame:
        Index of the frame to freeze. If ``None``, uses the average
        magnitude across all frames (with phase from the first frame).
    """
    if frame is not None:
        frozen = spectrum.frames[frame]
    else:
        avg_mag = np.mean(spectrum.magnitude, axis=0)
        frozen = avg_mag * np.exp(1j * spectrum.phase[0])

    frames = np.tile(frozen, (spectrum.n_frames, 1))
    return Spectrum(frames, spectrum.window.copy(), spectrum.hop_size,
                    spectrum.sample_rate, spectrum.original_length)


def smear(spectrum: Spectrum, amount: float) -> Spectrum:
    """Spectral smear: blur magnitude across neighbouring bins per frame.

    The phase is preserved; only the magnitude envelope is smoothed.

    Parameters
    ----------
    amount:
        Standard deviation of the Gaussian kernel in bins. 0 = no change.
    """
    if amount <= 0:
        return Spectrum(spectrum.frames.copy(), spectrum.window.copy(),
                        spectrum.hop_size, spectrum.sample_rate,
                        spectrum.original_length)

    mag = spectrum.magnitude
    phase = spectrum.phase

    # Apply Gaussian blur along the frequency axis (axis=1) for each frame
    smoothed_mag = gaussian_filter1d(mag, sigma=amount, axis=1)

    frames = smoothed_mag * np.exp(1j * phase)
    return Spectrum(frames, spectrum.window.copy(), spectrum.hop_size,
                    spectrum.sample_rate, spectrum.original_length)


def shift_bins(spectrum: Spectrum, shift: int | float) -> Spectrum:
    """Shift all frequency bins up or down.

    Parameters
    ----------
    shift:
        Number of bins to shift. Positive = up in frequency, negative = down.
        Float values use linear interpolation between bins.
    """
    mag = spectrum.magnitude
    phase = spectrum.phase
    n_bins = spectrum.n_bins

    if isinstance(shift, float) and shift != int(shift):
        # Fractional shift via linear interpolation
        int_shift = int(np.floor(shift))
        frac = shift - int_shift
        shifted_mag = np.zeros_like(mag)
        shifted_phase = np.zeros_like(phase)

        for b in range(n_bins):
            src_lo = b - int_shift
            src_hi = src_lo - 1  # one bin lower (for interpolation)
            if 0 <= src_lo < n_bins and 0 <= src_hi < n_bins:
                shifted_mag[:, b] = (1 - frac) * mag[:, src_lo] + frac * mag[:, src_hi]
                shifted_phase[:, b] = phase[:, src_lo]
            elif 0 <= src_lo < n_bins:
                shifted_mag[:, b] = (1 - frac) * mag[:, src_lo]
                shifted_phase[:, b] = phase[:, src_lo]
            elif 0 <= src_hi < n_bins:
                shifted_mag[:, b] = frac * mag[:, src_hi]
                shifted_phase[:, b] = phase[:, src_hi]
    else:
        # Integer shift — simple roll with zero fill
        shift_int = int(shift)
        shifted_mag = np.zeros_like(mag)
        shifted_phase = np.zeros_like(phase)

        if shift_int >= 0:
            if shift_int < n_bins:
                shifted_mag[:, shift_int:] = mag[:, : n_bins - shift_int]
                shifted_phase[:, shift_int:] = phase[:, : n_bins - shift_int]
        else:
            if -shift_int < n_bins:
                shifted_mag[:, : n_bins + shift_int] = mag[:, -shift_int:]
                shifted_phase[:, : n_bins + shift_int] = phase[:, -shift_int:]

    frames = shifted_mag * np.exp(1j * shifted_phase)
    return Spectrum(frames, spectrum.window.copy(), spectrum.hop_size,
                    spectrum.sample_rate, spectrum.original_length)


def cross_synthesize(
    carrier: Spectrum,
    modulator: Spectrum,
    mix: float = 1.0,
) -> Spectrum:
    """Vocoder-style cross-synthesis: modulator magnitude, carrier phase.

    Parameters
    ----------
    carrier:
        Spectrum providing the phase (harmonic content).
    modulator:
        Spectrum providing the magnitude (spectral envelope).
    mix:
        Blend between carrier magnitude and modulator magnitude.
        0.0 = carrier unchanged, 1.0 = full cross-synthesis.
    """
    carrier._check_compatible(modulator)

    n = min(carrier.n_frames, modulator.n_frames)
    length = min(carrier.original_length, modulator.original_length)

    carrier_mag = carrier.magnitude[:n]
    modulator_mag = modulator.magnitude[:n]
    carrier_phase = carrier.phase[:n]

    blended_mag = (1.0 - mix) * carrier_mag + mix * modulator_mag
    frames = blended_mag * np.exp(1j * carrier_phase)

    return Spectrum(frames, carrier.window.copy(), carrier.hop_size,
                    carrier.sample_rate, length)


def pitch_shift(spectrum: Spectrum, semitones: float) -> Spectrum:
    """Shift pitch by resampling the magnitude spectrum per frame.

    Stretches or compresses the magnitude envelope along the frequency axis
    without changing duration. Phase is propagated from the nearest source bin.

    Parameters
    ----------
    semitones:
        Pitch shift in semitones. Positive = up, negative = down.
    """
    ratio = 2.0 ** (semitones / 12.0)
    n_bins = spectrum.n_bins
    mag = spectrum.magnitude
    phase = spectrum.phase

    # Source bin indices for each target bin
    src_bins = np.arange(n_bins) / ratio
    src_floor = np.floor(src_bins).astype(int)
    src_frac = src_bins - src_floor

    new_mag = np.zeros_like(mag)
    new_phase = np.zeros_like(phase)

    for b in range(n_bins):
        lo = src_floor[b]
        hi = lo + 1
        if 0 <= lo < n_bins and hi < n_bins:
            new_mag[:, b] = (1 - src_frac[b]) * mag[:, lo] + src_frac[b] * mag[:, hi]
            new_phase[:, b] = phase[:, lo]
        elif 0 <= lo < n_bins:
            new_mag[:, b] = mag[:, lo]
            new_phase[:, b] = phase[:, lo]

    frames = new_mag * np.exp(1j * new_phase)
    return Spectrum(frames, spectrum.window.copy(), spectrum.hop_size,
                    spectrum.sample_rate, spectrum.original_length)
