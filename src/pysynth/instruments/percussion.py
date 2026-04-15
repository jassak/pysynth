from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal


class Percussion:
    """Bridges a pre-rendered hit sound to the CV/gate world.

    Takes a single-hit Signal and exposes :meth:`trigger` which places
    the hit at every rising edge of a gate signal, scaled by the gate's
    pulse height (velocity).  This is the percussive counterpart to
    ``Envelope.trigger(gate)``::

        kick_hit = tr808_kick()          # pre-rendered single hit
        perc = Percussion(kick_hit)

        gates = drum_machine.cv()
        kick_audio = perc.trigger(gates["kick"])

    Parameters
    ----------
    hit:
        A pre-rendered single-hit Signal.
    """

    def __init__(self, hit: Signal) -> None:
        self.hit = hit

    def trigger(self, gate: Signal) -> Signal:
        """Place the hit at every rising edge in *gate*, scaled by velocity.

        Parameters
        ----------
        gate:
            A gate Signal (e.g. from :meth:`DrumMachine.cv`).  Rising
            edges trigger hit placement; the gate height at each edge
            scales the hit's amplitude.

        Returns
        -------
        Signal
            Audio spanning the full gate duration.
        """
        g = gate.data
        sr = gate.sample_rate

        # Detect rising edges: sample > 0 where previous sample was <= 0
        prev = np.empty_like(g)
        prev[0] = 0.0
        prev[1:] = g[:-1]
        edges = np.where((g > 0.0) & (prev <= 0.0))[0]

        if len(edges) == 0:
            return Signal.silence(gate.duration, sr)

        total = len(g)
        hit_data = self.hit.data
        hit_len = len(hit_data)
        if self.hit.n_channels == 1:
            out = np.zeros(total, dtype=np.float32)
        else:
            out = np.zeros((total, self.hit.n_channels), dtype=np.float32)

        for idx in edges:
            velocity = g[idx]
            end = min(idx + hit_len, total)
            out[idx:end] += hit_data[:end - idx] * velocity

        return Signal(out, sr)
