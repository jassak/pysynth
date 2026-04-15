"""Pitch constants for all 12 pitch classes in octave 3 (A3 = 220 Hz).

These make scale transposition immediate::

    from presets.pitches import C, A
    from presets.scales import major

    major()     # A major (default)
    major(C)    # C major
"""

from pysynth import Pitch

A  = Pitch(220.000)
Bb = Pitch(233.082)
B  = Pitch(246.942)
C  = Pitch(261.626)
Db = Pitch(277.183)
D  = Pitch(293.665)
Eb = Pitch(311.127)
E  = Pitch(329.628)
F  = Pitch(349.228)
Gb = Pitch(369.994)
G  = Pitch(391.995)
Ab = Pitch(415.305)

# Enharmonic aliases
Asharp = Bb
Csharp = Db
Dsharp = Eb
Fsharp = Gb
Gsharp = Ab
