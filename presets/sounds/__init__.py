"""Sound presets: ready-to-use instrument patches and drum sounds.

Pitched instruments are functions returning ``Patch`` objects::

    from presets.sounds import organ, bell
    organ().preview()
    bell(mod_index=5.0).render(440, dur=2.0)

Percussion functions return ``Signal`` directly::

    from presets.sounds import TR808, TR909
    TR808.kick().play()
"""

from presets.sounds.patch import Patch
from presets.sounds.synths import Synth, FMSynth
from presets.sounds.keys import organ, electric_piano
from presets.sounds.bass import acid_bass, sub_bass, detuned_saw
from presets.sounds.pads import ambient_pad, string_pad
from presets.sounds.leads import pluck, bell, fm_bass, fm_metal
from presets.sounds.percussion import TR808, TR909
