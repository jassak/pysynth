"""Sound presets: ready-to-use instrument patches and drum sounds.

Pitched instruments are functions returning ``Patch`` objects::

    from presets.sounds import organ, bell
    organ().preview()
    bell(mod_index=5.0).render(440, dur=2.0)

Percussion functions return ``Signal`` directly::

    from presets.sounds import tr808_kick, tr909_hihat
    tr808_kick().play()
"""

from presets.sounds.patch import Patch
from presets.sounds.synths import Synth, FMSynth
from presets.sounds.keys import organ, electric_piano
from presets.sounds.bass import acid_bass, sub_bass, detuned_saw
from presets.sounds.pads import ambient_pad, string_pad
from presets.sounds.leads import pluck, bell, fm_bass, fm_metal
from presets.sounds.percussion import (
    tr808_kick, tr808_snare, tr808_hihat, tr808_clap,
    tr808_cowbell, tr808_tom, tr808_rimshot,
    tr909_kick, tr909_snare, tr909_hihat, tr909_clap, tr909_ride,
)
