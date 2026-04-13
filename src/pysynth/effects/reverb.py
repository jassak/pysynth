from __future__ import annotations

import numba
import numpy as np

from pysynth._core import Effect, Signal


@numba.njit(cache=True)
def _comb_filter(x, out, buf, d, feedback, damp1, damp2):
    filt = 0.0
    for i in range(len(x)):
        buf_out = buf[i % d]
        filt = buf_out * damp2 + filt * damp1
        buf[i % d] = x[i] + filt * feedback
        out[i] = buf_out


@numba.njit(cache=True)
def _allpass_filter(inp, out, buf, d):
    for i in range(len(inp)):
        buf_out = buf[i % d]
        buf[i % d] = inp[i] + buf_out * 0.5
        out[i] = buf_out - inp[i] * 0.5


@numba.njit(cache=True)
def _iir_lowpass_inplace(y, coef):
    for i in range(1, len(y)):
        y[i] = coef * y[i - 1] + (1.0 - coef) * y[i]


@numba.njit(cache=True)
def _iir_lowpass_range(arr, coef, start, end):
    for k in range(start, end):
        arr[k] = coef * arr[k - 1] + (1.0 - coef) * arr[k]


def _allpass_comb(x: np.ndarray, d: int, start: int, diffusion: float) -> np.ndarray:
    """Vectorised Schroeder allpass-comb filter.

    Processes in chunks of ``d`` samples so the recurrence can be expressed as
    a vector operation. ``start`` is the first sample to process (samples
    before it are treated as silence / initial state).
    """
    y = np.zeros(len(x))
    i = start
    while i < len(x):
        s = min(d, len(x) - i)
        y[i:][:s] = diffusion * (x[i:][:s] - y[i - d :][:s]) + x[i - d :][:s]
        i += s
    return y


class SimpleReverb(Effect):
    """Schroeder-style reverb using a network of comb and allpass filters.

    Parameters
    ----------
    room_size:
        Controls comb filter delay lengths (0..1). Larger = longer decay.
    damping:
        High-frequency damping in the comb filters (0..1).
    wet:
        Mix ratio of processed signal (0 = dry, 1 = fully wet).
    """

    # Comb filter delay times in samples at 44100 Hz (prime-ish values)
    _COMB_DELAYS = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
    _ALLPASS_DELAYS = [225, 556, 441, 341]

    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet: float = 0.3,
    ) -> None:
        self.room_size = np.clip(room_size, 0.0, 1.0)
        self.damping = np.clip(damping, 0.0, 1.0)
        self.wet = np.clip(wet, 0.0, 1.0)

    def __call__(self, sig: Signal) -> Signal:
        x = sig.data.astype(np.float64)
        scale = sig.sample_rate / 44100.0

        feedback = 0.84 + self.room_size * 0.15
        damp1 = self.damping * 0.4
        damp2 = 1.0 - damp1

        # Parallel comb filters
        comb_out = np.zeros_like(x)
        for delay in self._COMB_DELAYS:
            d = max(1, int(delay * scale))
            buf = np.zeros(d)
            out = np.empty_like(x)
            _comb_filter(x, out, buf, d, feedback, damp1, damp2)
            comb_out += out

        comb_out /= len(self._COMB_DELAYS)

        # Series allpass filters
        ap_out = comb_out
        for delay in self._ALLPASS_DELAYS:
            d = max(1, int(delay * scale))
            buf = np.zeros(d)
            out = np.empty_like(ap_out)
            _allpass_filter(ap_out, out, buf, d)
            ap_out = out

        mixed = (1.0 - self.wet) * x + self.wet * ap_out
        return Signal(mixed.astype(np.float32), sig.sample_rate)


class DatorroReverb(Effect):
    """Dattorro figure-of-eight plate reverb (JAES 45(9), 1997).

    Two cross-coupled feedback loops form a "figure-of-eight" tank. An input
    diffuser (four allpass filters) spreads transients before they enter the
    tank. Multiple taps are read from both loops and mixed asymmetrically to
    produce decorrelated L/R output — the classic studio plate sound.

    Always outputs a stereo Signal regardless of whether the input is mono or
    stereo.

    All delay times are specified in the paper relative to a 29761 Hz
    reference sample rate and are scaled automatically to ``sig.sample_rate``.

    Parameters
    ----------
    decay:
        Reverb tail length (0..1). Higher = longer, denser tail.
    bandwidth:
        Pre-filter coefficient; rolls off high frequencies on the input
        (0..1, where 1 = no filtering).
    damping:
        High-frequency damping inside the tank (0..1). Small values = bright,
        large values = dark/muffled.
    decay_diffusion1, decay_diffusion2:
        Allpass coefficients in the tank (controls density of reflections).
    input_diffusion1, input_diffusion2:
        Allpass coefficients in the input section.
    wet:
        Level of the reverbed signal in the mix.
    dry:
        Level of the original (dry) signal in the mix.
    """

    _ORIGINAL_FS = 29761  # Reference sample rate from Dattorro's paper

    def __init__(
        self,
        decay: float = 0.5,
        bandwidth: float = 0.9995,
        damping: float = 0.0005,
        decay_diffusion1: float = 0.7,
        decay_diffusion2: float = 0.5,
        input_diffusion1: float = 0.75,
        input_diffusion2: float = 0.625,
        wet: float = 0.4,
        dry: float = 1.0,
    ) -> None:
        self.decay = decay
        self.bandwidth = bandwidth
        self.damping = damping
        self.decay_diffusion1 = decay_diffusion1
        self.decay_diffusion2 = decay_diffusion2
        self.input_diffusion1 = input_diffusion1
        self.input_diffusion2 = input_diffusion2
        self.wet = wet
        self.dry = dry

    def __call__(self, sig: Signal) -> Signal:
        fs = sig.sample_rate

        def c(n: int) -> int:
            """Scale a sample count from the paper's 29761 Hz reference."""
            return int(np.round(n * fs / self._ORIGINAL_FS))

        x = sig.data.astype(np.float64)
        mono = x if x.ndim == 1 else x.mean(axis=1)
        n = len(mono)

        # Tank delay line lengths (from Table 1 in the paper)
        dlt = [c(672), c(4453), c(1800), c(3720), c(908), c(4217), c(2656), c(3163)]
        # Pad must exceed the largest delay so index arithmetic never underflows
        pad = max(dlt) + 1

        y = np.concatenate([np.zeros(pad), mono])

        # Bandwidth pre-filter: first-order low-pass on input
        _iir_lowpass_inplace(y, self.bandwidth)

        # Input diffuser: four allpass filters in series
        y = _allpass_comb(y, c(142), pad, self.input_diffusion1)
        y = _allpass_comb(y, c(107), pad, self.input_diffusion1)
        y = _allpass_comb(y, c(379), pad, self.input_diffusion2)
        y = _allpass_comb(y, c(277), pad, self.input_diffusion2)

        # Tank: eight delay lines arranged as two cross-coupled loops
        dl = np.zeros((8, pad + n))
        i = pad
        while i < pad + n:
            s = min(min(dlt), pad + n - i)
            for j in range(0, 8, 4):
                # Cross-coupling: loop 0 feeds from loop 1's last line (dl[7])
                #                 loop 1 feeds from loop 0's last line (dl[3])
                prev = (j - 1) % 8
                dl[j, i:][:s] = (
                    y[i:][:s]
                    + self.decay * dl[prev, i - dlt[prev] :][:s]
                    + self.decay_diffusion1 * dl[j, i - dlt[j] :][:s]
                )
                dl[j + 1, i:][:s] = dl[j, i - dlt[j] :][:s] - self.decay_diffusion1 * dl[j, i:][:s]
                # High-frequency damping: first-order LP (numba-accelerated)
                _iir_lowpass_range(dl[j + 1], self.damping, i, i + s)
                dl[j + 2, i:][:s] = (
                    self.decay * dl[j + 1, i - dlt[j + 1] :][:s]
                    - self.decay_diffusion2 * dl[j + 2, i - dlt[j + 2] :][:s]
                )
                dl[j + 3, i:][:s] = dl[j + 2, i - dlt[j + 2] :][:s] + self.decay_diffusion2 * dl[j + 2, i:][:s]
            i += s

        # Output tap mix (from Table 2 in the paper)
        def tap(line: int, offset: int) -> np.ndarray:
            start = pad - dlt[line] + c(offset)
            return dl[line, start:][:n]

        yL = tap(5, 266) + tap(5, 2974) - tap(6, 1913) + tap(7, 1996) - tap(1, 1990) - tap(2, 187) - tap(3, 1066)
        yR = tap(1, 353) + tap(1, 3627) - tap(2, 1228) + tap(3, 2673) - tap(5, 2111) - tap(6, 335) - tap(7, 121)

        wet_stereo = np.column_stack([yL, yR]) * 0.6

        # Dry: expand mono to stereo so shapes match
        dry_stereo = np.column_stack([mono, mono]) if x.ndim == 1 else x

        out = self.dry * dry_stereo + self.wet * wet_stereo
        return Signal(out.astype(np.float32), sig.sample_rate)
