from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pysynth._core import Effect, Signal, SAMPLE_RATE
from pysynth.effects.dynamics import Limiter
from pysynth.mixing.panning import pan


@dataclass
class _Track:
    signal: Signal
    volume: float
    position: float        # pan: -1..+1
    effects: Effect | None


class Mixer:
    """Combine multiple Signals into a single stereo output.

    Each track has independent volume, pan position, and an optional effects
    chain applied before mixing. The final render sums all tracks and passes
    the result through a master limiter to prevent clipping.

    Usage::

        mixer = Mixer()
        mixer.add_track(bass,   volume=0.8, pan=0.0)
        mixer.add_track(lead,   volume=0.6, pan=-0.3, effects=Delay(0.3, 0.4))
        mixer.add_track(pads,   volume=0.4, pan=0.2,  effects=SimpleReverb(0.7))
        stereo = mixer.render()
        stereo.play()
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._tracks: list[_Track] = []

    def add_track(
        self,
        signal: Signal,
        *,
        volume: float = 1.0,
        position: float = 0.0,
        effects: Effect | None = None,
    ) -> None:
        """Add a signal as a new track.

        Parameters
        ----------
        signal:
            The audio content for this track.
        volume:
            Linear amplitude scale applied before mixing (1.0 = unity gain).
        position:
            Pan position in [-1, 1]. -1 = full left, 0 = centre, +1 = full right.
        effects:
            An Effect (or chained effects via ``|``) applied to this track before
            it is panned and mixed.
        """
        self._tracks.append(_Track(signal, volume, position, effects))

    def render(self, master_ceiling_db: float = -0.1) -> Signal:
        """Mix all tracks into a single stereo Signal.

        Each track is processed in order:
        1. Apply per-track effects chain (if any).
        2. Scale by track volume.
        3. Pan to stereo.

        The resulting stereo signals are summed sample-by-sample (shorter tracks
        are zero-padded). A master Limiter is applied last to prevent clipping.

        Parameters
        ----------
        master_ceiling_db:
            Hard ceiling applied by the master limiter, in dBFS. Default -0.1 dBFS.

        Returns a stereo Signal with shape (n_samples, 2).
        """
        if not self._tracks:
            return Signal(np.zeros((0, 2), dtype=np.float32), self.sample_rate)

        # Find the longest track to set buffer length
        stereo_signals: list[Signal] = []
        for track in self._tracks:
            sig = track.signal
            if track.effects is not None:
                sig = track.effects(sig)
            sig = sig * track.volume
            sig = pan(sig, track.position)
            stereo_signals.append(sig)

        max_len = max(len(s.data) for s in stereo_signals)
        buf = np.zeros((max_len, 2), dtype=np.float32)

        for sig in stereo_signals:
            n = len(sig.data)
            buf[:n] += sig.data

        mixed = Signal(buf, self.sample_rate)
        return Limiter(master_ceiling_db)(mixed)
