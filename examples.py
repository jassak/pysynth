"""
examples.py — pysynth sound library
Load in the REPL and call .play() on anything that catches your eye.

  $ uv run ipython
  >>> from examples import *
  >>> drone.play()
  >>> bass_seq.play()

Everything is a function to keep import fast. Call it to get a Signal:
  >>> bell().play()
  >>> melody().play()
"""

from pysynth import ( SAMPLE_RATE, Signal, Generator, Oscillator, WhiteNoise,
    PinkNoise, Wavetable, Segment, Envelope, adsr, LowPassFilter, HighPassFilter,
    BandPassFilter, Gain, Compressor, Limiter, SimpleReverb, DatorroReverb, Delay,
    Echo, Tanh, Clip, Overdrive, Pitch, Note, Scale, Sequencer, Step, StepSequencer,
    Arpeggiator, Mixer, pan, stft, freeze, smear, shift_bins, cross_synthesize,
    pitch_shift, SpectralFreeze, SpectralSmear, PitchShift, Vocoder, ConvolutionReverb,
)

DUR = 3.0


# ---------------------------------------------------------------------------
# Shared scales / envelopes (cheap to construct, not rendered yet)
# ---------------------------------------------------------------------------

ji_major = Scale(220, [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2])
pentatonic = Scale(220, [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 2])

note_env = adsr(attack=0.01, decay=0.05, sustain=0.35, sustain_level=0.7, release=0.08)
short_env = adsr(attack=0.005, decay=0.08, sustain=0.1, sustain_level=0.5, release=0.06)
pluck_env = adsr(attack=0.005, decay=0.2, sustain=0.0, sustain_level=0.0, release=0.01)
bell_env = adsr(attack=0.01, decay=0.4, sustain=0.0, sustain_level=0.0, release=0.01)
bass_env = adsr(attack=0.005, decay=0.1, sustain=0.4, sustain_level=0.6, release=0.15)
pad_env = adsr(attack=0.3, decay=0.2, sustain=1.5, sustain_level=0.8, release=0.5)


# ---------------------------------------------------------------------------
# 1. Basic tones
# ---------------------------------------------------------------------------


def sine_tone(hz=440, dur=DUR):
    return Oscillator("sine").at(hz).render(dur)


def square_tone(hz=220, dur=DUR):
    return Oscillator("square").at(hz).render(dur)


def saw_tone(hz=110, dur=DUR):
    return Oscillator("saw").at(hz).render(dur)


def pluck(hz=440):
    """Short plucked sine — good for testing envelopes."""
    return pluck_env.apply(Oscillator("sine").at(hz).render(0.6))


# ---------------------------------------------------------------------------
# 2. Additive synthesis
# ---------------------------------------------------------------------------


def organ(hz=220, dur=DUR):
    """Hammond-style drawbar organ: 8 harmonic partials."""
    osc = (
        Oscillator("sine", ratio=1) * 0.80
        + Oscillator("sine", ratio=2) * 0.60
        + Oscillator("sine", ratio=3) * 0.40
        + Oscillator("sine", ratio=4) * 0.30
        + Oscillator("sine", ratio=5) * 0.20
        + Oscillator("sine", ratio=6) * 0.15
        + Oscillator("sine", ratio=8) * 0.10
        + Oscillator("sine", ratio=10) * 0.05
    )
    return osc.at(hz).render(dur) * 0.25


def detuned_saw(hz=110, dur=DUR):
    """Two saws slightly detuned — fat unison sound."""
    a = Oscillator("saw").at(hz * 1.004).render(dur)
    b = Oscillator("saw").at(hz * 0.996).render(dur)
    return (a + b) * 0.35


# ---------------------------------------------------------------------------
# 3. FM synthesis
# ---------------------------------------------------------------------------


def bell(hz=440, dur=DUR):
    """FM bell: carrier + 2x modulator, high index."""
    mod = Oscillator("sine").at(hz * 2.0).render(dur) * (hz * 3.5)
    sig = Oscillator("sine").at(hz + mod).render(dur)
    return bell_env.apply(sig)


def fm_bass(hz=55, dur=DUR):
    """Wobbly FM sub-bass."""
    mod = Oscillator("sine").at(hz * 0.5).render(dur) * (hz * 2.0)
    return Oscillator("sine").at(hz + mod).render(dur)


def fm_metal(hz=200, dur=DUR):
    """Inharmonic FM: irrational ratio makes it metallic."""
    mod = Oscillator("sine").at(hz * 3.14).render(dur) * (hz * 8.0)
    return Oscillator("sine").at(hz + mod).render(dur)


# ---------------------------------------------------------------------------
# 4. Vibrato & modulation
# ---------------------------------------------------------------------------


def vibrato(hz=440, rate=5.0, depth=15.0, dur=DUR):
    """Pitch LFO vibrato."""
    lfo = Oscillator("sine").at(rate).render(dur) * depth
    return Oscillator("sine").at(hz + lfo).render(dur)


def tremolo(hz=440, rate=7.0, depth=0.4, dur=DUR):
    """Amplitude LFO tremolo."""
    lfo = Oscillator("sine").at(rate).render(dur) * depth + (1.0 - depth)
    return Oscillator("sine").at(hz).render(dur) * lfo


def pitch_sweep(start_hz=200, end_hz=800, dur=DUR):
    """Slow sinusoidal pitch sweep between two frequencies."""
    mid = (start_hz + end_hz) / 2
    depth = (end_hz - start_hz) / 2
    lfo = Oscillator("sine").at(0.5 / dur * 2).render(dur) * depth + mid
    return Oscillator("sine").at(lfo).render(dur)


# ---------------------------------------------------------------------------
# 5. Filters
# ---------------------------------------------------------------------------


def lp_sweep(hz=110, dur=DUR):
    """Saw wave with LFO-swept low-pass filter (classic synth filter sweep)."""
    src = Oscillator("saw").at(hz).render(dur) * 0.5
    cutoff = Oscillator("sine").at(0.4).render(dur) * 1200 + 1400
    return LowPassFilter(cutoff)(src)


def wah(hz=110, rate=2.0, dur=DUR):
    """Band-pass wah-wah effect."""
    src = Oscillator("saw").at(hz).render(dur) * 0.5
    center = Oscillator("sine").at(rate).render(dur) * 600 + 900
    return BandPassFilter(center - 200, center + 200)(src)


def resonant_sweep(hz=110, dur=DUR):
    """High-order low-pass with fast sweep — nasal / resonant character."""
    src = Oscillator("saw").at(hz).render(dur) * 0.5
    cutoff = Oscillator("sine").at(0.25).render(dur) * 2000 + 2200
    return LowPassFilter(cutoff, order=8)(src)


# ---------------------------------------------------------------------------
# 6. Distortion & saturation
# ---------------------------------------------------------------------------


def soft_clip(hz=110, dur=DUR):
    return Tanh(drive=4.0)(Oscillator("saw").at(hz).render(dur) * 0.4)


def hard_clip(hz=110, dur=DUR):
    return Clip(threshold=0.3)(Oscillator("saw").at(hz).render(dur) * 0.6)


def overdrive_tone(hz=110, dur=DUR):
    """Overdrive + low-pass to tame harshness."""
    sig = Oscillator("saw").at(hz).render(dur) * 0.4
    return (Overdrive(gain=6.0, bias=0.1) | LowPassFilter(3000))(sig)


def dynamic_drive(hz=110, dur=DUR):
    """Drive amount modulated by slow LFO."""
    sig = Oscillator("saw").at(hz).render(dur) * 0.4
    drive = Oscillator("sine").at(0.3).render(dur) * 3.0 + 4.0
    return Tanh(drive=drive)(sig)


# ---------------------------------------------------------------------------
# 7. Reverb & delay
# ---------------------------------------------------------------------------


def with_reverb(signal: Signal, room=0.6, wet=0.4) -> Signal:
    return DatorroReverb(decay=room, wet=wet)(signal)


def with_echo(signal: Signal, time=0.35, repeats=5) -> Signal:
    return Echo(delay_time=time, repeats=repeats, decay=0.55, wet=0.5)(signal)


def flanger(hz=440, dur=DUR):
    """LFO-modulated short delay — classic flange sweep."""
    src = Oscillator("sine").at(hz).render(dur)
    lfo = Oscillator("sine").at(0.4).render(dur) * 0.003 + 0.005
    return Delay(delay_time=lfo, feedback=0.6, wet=0.5)(src)


def space_bell():
    """FM bell with plate reverb and echo — very ambient."""
    sig = bell(523, dur=1.5)
    sig = DatorroReverb(decay=0.85, wet=0.5)(sig)
    return Echo(delay_time=0.4, repeats=4, decay=0.7, wet=0.4)(sig)


# ---------------------------------------------------------------------------
# 8. Noise & percussion
# ---------------------------------------------------------------------------


def hihat(open=False):
    dur = 0.4 if open else 0.08
    env = adsr(0.001, dur * 0.8, 0.0, 0.0, 0.01)
    return env.apply(HighPassFilter(8000)(WhiteNoise().render(dur + 0.02)))


def snare():
    env = adsr(0.001, 0.12, 0.0, 0.0, 0.01)
    return env.apply(WhiteNoise().render(0.15) * 0.6 + Oscillator("sine").at(180).render(0.15) * 0.4)


def kick():
    env = adsr(0.001, 0.08, 0.0, 0.0, 0.01)
    pitch = Oscillator("sine").at(80).render(0.15) * 0.8
    body = Oscillator("sine").at(40).render(0.15) * 0.2
    return env.apply(pitch + body)


def drum_pattern():
    """One bar of kick, snare, hi-hat rendered to a single Signal."""
    k, s, h = kick(), snare(), hihat()
    bar = Signal.silence(2.0)
    beats = [0.0, 0.5, 1.0, 1.5]  # 4/4 at 120 bpm → 0.5s per beat
    result = Signal.silence(2.0)
    for t in beats:
        n = int(t * SAMPLE_RATE)
        # hi-hat on every beat
        end = min(len(result.data), n + len(h.data))
        result = Signal(result.data.copy())
        result.data[n:end] += h.data[: end - n] * 0.5
    # kick on 1, 3; snare on 2, 4
    for t, src in [(0.0, k), (1.0, k), (0.5, s), (1.5, s)]:
        n = int(t * SAMPLE_RATE)
        end = min(len(result.data), n + len(src.data))
        result.data[n:end] += src.data[: end - n]
    return result


# ---------------------------------------------------------------------------
# 9. Melodic sequences (CV/gate)
# ---------------------------------------------------------------------------


def melody():
    """Ascending/descending ji-major scale on a sine oscillator."""
    notes = [Note(ji_major[i], 0.5) for i in [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0]]
    pitch, gate = Sequencer(notes, bpm=100).cv()
    audio = Oscillator("sine").at(pitch).render(pitch.duration)
    return audio * note_env.trigger(gate)


def bass_line(repeats=2):
    """Simple bass line with rests, rendered on a triangle+harmonic voice."""
    notes = [
        Note(ji_major[0], 1.0),
        Note.rest(0.5),
        Note(ji_major[4], 0.5),
        Note(ji_major[2], 1.0),
        Note.rest(0.5),
        Note(ji_major[0] * 0.5, 0.5),  # sub-octave
    ]
    gen = Oscillator("triangle") + Oscillator("triangle", ratio=2) * 0.3
    pitch, gate = Sequencer(notes, bpm=90).cv(repeats=repeats)
    audio = gen.at(pitch).render(pitch.duration)
    return audio * bass_env.trigger(gate)


def pentatonic_groove(repeats=4):
    """Pentatonic phrase with velocity shaping on a square wave."""
    notes = [
        Note(pentatonic[0], 0.25, velocity=1.0),
        Note(pentatonic[2], 0.25, velocity=0.7),
        Note(pentatonic[4], 0.25, velocity=0.9),
        Note(pentatonic[3], 0.25, velocity=0.6),
        Note(pentatonic[1], 0.50, velocity=0.8),
        Note(pentatonic[5], 0.50, velocity=1.0),
    ]
    gen = Oscillator("square") + Oscillator("square", ratio=2) * 0.15
    pitch, gate = Sequencer(notes, bpm=120).cv(repeats=repeats)
    audio = gen.at(pitch).render(pitch.duration)
    return audio * short_env.trigger(gate)


# ---------------------------------------------------------------------------
# 10. Arpeggiators
# ---------------------------------------------------------------------------


def _major_chord():
    return [Note(ji_major[i], 0.25) for i in [0, 2, 4, 7]]


def arp_up(bars=4, bpm=140):
    pitch, gate = Arpeggiator(_major_chord(), pattern="up", note_duration=0.25, bpm=bpm).cv(bars=bars)
    audio = Oscillator("sine").at(pitch).render(pitch.duration)
    return audio * note_env.trigger(gate)


def arp_updown(bars=4, bpm=160):
    pitch, gate = Arpeggiator(_major_chord(), pattern="up_down", note_duration=0.125, bpm=bpm).cv(bars=bars)
    audio = Oscillator("triangle").at(pitch).render(pitch.duration)
    return audio * short_env.trigger(gate)


def arp_random(bars=4, bpm=120):
    gen = Oscillator("sine") + Oscillator("sine", ratio=3) * 0.3
    pitch, gate = Arpeggiator(_major_chord(), pattern="random", note_duration=0.25, bpm=bpm, octaves=2).cv(bars=bars)
    audio = gen.at(pitch).render(pitch.duration)
    return audio * note_env.trigger(gate)


def arp_fm(bars=4, bpm=80):
    """Arpeggiator using FM bells."""
    pitch, gate = Arpeggiator(_major_chord(), pattern="up", note_duration=0.5, bpm=bpm).cv(bars=bars)
    mod = Oscillator("sine").at(pitch * 2.0).render(pitch.duration) * (pitch * 8.0)
    audio = Oscillator("sine").at(pitch + mod).render(pitch.duration)
    return audio * bell_env.trigger(gate)


# ---------------------------------------------------------------------------
# 11. 303-style acid bass (CV/gate showcase)
# ---------------------------------------------------------------------------


def acid_bass(bpm=130):
    """TB-303 style: saw + filter envelope + amp envelope via CV/gate."""
    notes = [
        Note(Pitch(55), 0.5),
        Note(Pitch(55), 0.25),
        Note(Pitch(82.4), 0.25),
        Note(Pitch(73.4), 0.5),
        Note.rest(0.25),
        Note(Pitch(55), 0.25),
    ]
    pitch, gate = Sequencer(notes, bpm=bpm).cv(repeats=4)
    dur = pitch.duration

    audio = Oscillator("saw").at(pitch).render(dur) * 0.5
    amp = adsr(0.005, 0.1, 0.3, 0.8, 0.05).trigger(gate)
    cutoff = adsr(0.005, 0.15, 0.0, 0.0, 0.03).trigger(gate) * 5000 + 300
    output = LowPassFilter(cutoff)(audio) * amp

    return (Overdrive(gain=3.0) | LowPassFilter(4000))(output)


def acid_303(bpm=138):
    """TB-303 acid line built with StepSequencer — separate lanes for
    pitch, filter cutoff, and accent drive modular-synth style.

    16 steps of classic acid: slides, ties, rests, per-step filter and
    accent control all wired together via Signal algebra."""
    A1, C2, D2, E2, G1 = 55.0, 65.4, 73.4, 82.4, 49.0

    # -- Pitch lane: note pattern with slides, ties, and rests ----------
    pitch_steps = [
        Step(A1),                             # 1
        Step(A1, gate_length=0.5),            # 2  staccato
        Step(C2, slide=True),                 # 3  slide up
        Step.tie(),                           # 4  hold C2
        Step(D2),                             # 5
        Step.rest(),                          # 6  silence
        Step(A1, gate_length=0.9),            # 7  long gate
        Step(E2, slide=True),                 # 8  slide up
        Step(D2, slide=True),                 # 9  slide back down
        Step(D2, gate_length=0.4),            # 10 staccato
        Step.rest(),                          # 11
        Step(A1),                             # 12
        Step(G1, slide=True),                 # 13 slide to sub
        Step.tie(),                           # 14 hold G1
        Step(A1, slide=True),                 # 15 slide back up
        Step(A1, gate_length=0.3),            # 16 short blip
    ]
    pitch, gate = StepSequencer(
        pitch_steps, bpm=bpm, step_length=0.25, slide_time=0.03
    ).cv(repeats=4)
    dur = pitch.duration

    # -- Cutoff lane: per-step filter brightness -------------------------
    lo, hi = 400.0, 4500.0
    cutoff_steps = [
        Step(hi),  Step(lo),  Step(hi),  Step(hi),
        Step(hi),  Step(lo),  Step(lo),  Step(hi),
        Step(hi),  Step(lo),  Step(lo),  Step(hi),
        Step(hi),  Step(hi),  Step(hi, slide=True),  Step(lo),
    ]
    cutoff_base, _ = StepSequencer(
        cutoff_steps, bpm=bpm, step_length=0.25, slide_time=0.015
    ).cv(repeats=4)

    # -- Accent lane: drive amount per step ------------------------------
    soft, hard = 0.5, 1.0
    accent_steps = [
        Step(hard), Step(soft), Step(hard), Step(soft),
        Step(hard), Step(soft), Step(soft), Step(hard),
        Step(hard), Step(soft), Step(soft), Step(soft),
        Step(hard), Step(soft), Step(hard), Step(soft),
    ]
    accent, _ = StepSequencer(
        accent_steps, bpm=bpm, step_length=0.25
    ).cv(repeats=4)

    # -- Synthesis -------------------------------------------------------
    audio = Oscillator("saw").at(pitch).render(dur) * 0.5

    # Filter envelope opens on each gate, scaled by the cutoff lane
    filt_env = adsr(0.003, 0.12, 0.0, 0.0, 0.02).trigger(gate)
    cutoff = filt_env * cutoff_base + 250

    amp = adsr(0.003, 0.08, 0.25, 0.7, 0.04).trigger(gate)
    filtered = LowPassFilter(cutoff)(audio) * amp

    # Accent lane drives a tanh saturator — louder steps get more grit
    output = Tanh(drive=accent)(filtered)
    return (LowPassFilter(5500) | Echo(delay_time=0.375, repeats=3, decay=0.35, wet=0.2))(output)


# ---------------------------------------------------------------------------
# 12. Produced pieces
# ---------------------------------------------------------------------------


def ambient_pad(dur=6.0):
    """Slow stereo pad: detuned saws, plate reverb, panned wide."""
    env = adsr(0.5, 0.3, dur - 1.5, 0.75, 0.7)
    sig_l = env.apply(Oscillator("saw").at(220 * 1.003).render(dur)) * 0.3
    sig_r = env.apply(Oscillator("saw").at(220 * 0.997).render(dur)) * 0.3
    plate = DatorroReverb(decay=0.85, bandwidth=0.9995, wet=0.5)
    mx = Mixer()
    mx.add_track(plate(sig_l), volume=0.8, position=-0.7)
    mx.add_track(plate(sig_r), volume=0.8, position=+0.7)
    return mx.render()


# ---------------------------------------------------------------------------
# 13. Wavetable synthesis
# ---------------------------------------------------------------------------


def wt_sweep(hz=220, dur=DUR):
    """Slow morph from sine through saw to square via LFO-driven position."""
    wt = Wavetable.from_waveforms(["sine", "saw", "square"])
    lfo = Oscillator("triangle").at(0.3).render(dur) * 1.0 + 1.0  # sweeps 0→2
    return wt.at(hz, position=lfo).render(dur) * 0.5


def wt_pad(dur=6.0):
    """Evolving stereo pad: detuned wavetables with slow position drift."""
    wt = Wavetable.from_waveforms(["sine", "triangle", "saw"])
    env = adsr(0.4, 0.3, dur - 1.3, 0.8, 0.6)
    # slow, slightly different position LFOs for left and right
    pos_l = Oscillator("sine").at(0.08).render(dur) * 0.8 + 1.0
    pos_r = Oscillator("sine").at(0.11).render(dur) * 0.8 + 1.0
    sig_l = env.apply(wt.at(220 * 1.003, position=pos_l).render(dur)) * 0.3
    sig_r = env.apply(wt.at(220 * 0.997, position=pos_r).render(dur)) * 0.3
    plate = DatorroReverb(decay=0.8, wet=0.45)
    mx = Mixer()
    mx.add_track(plate(sig_l), volume=0.8, position=-0.6)
    mx.add_track(plate(sig_r), volume=0.8, position=+0.6)
    return mx.render()


def wt_bass(bpm=110):
    """Sequenced bass with filter-envelope-driven wavetable position."""
    wt = Wavetable.from_waveforms(["triangle", "saw", "square"])
    notes = [
        Note(Pitch(55), 0.5),
        Note(Pitch(55), 0.25),
        Note(Pitch(82.4), 0.25),
        Note(Pitch(73.4), 0.5),
        Note.rest(0.25),
        Note(Pitch(55), 0.25),
    ]
    pitch, gate = Sequencer(notes, bpm=bpm).cv(repeats=4)
    dur = pitch.duration

    # position follows the filter envelope — brighter on attack
    pos_env = adsr(0.005, 0.2, 0.0, 0.0, 0.05).trigger(gate) * 2.0
    audio = wt.at(pitch, position=pos_env).render(dur) * 0.5
    amp = adsr(0.005, 0.1, 0.3, 0.7, 0.1).trigger(gate)
    cutoff = adsr(0.005, 0.15, 0.0, 0.0, 0.03).trigger(gate) * 4000 + 300

    return (LowPassFilter(cutoff) | Overdrive(gain=2.0) | LowPassFilter(3500))(audio) * amp


def wt_pluck(hz=440):
    """Wavetable pluck: position decays from bright to mellow."""
    import numpy as np

    wt = Wavetable.from_waveforms(["square", "saw", "triangle", "sine"])
    dur = 0.8
    n = int(dur * SAMPLE_RATE)
    # fast exponential decay from 0 (square) to 3 (sine)
    pos = Signal(np.float32(3.0 * (1.0 - np.exp(-np.linspace(0, 8, n)))))
    return pluck_env.apply(wt.at(hz, position=pos).render(dur))


def lead_over_bass(bpm=90):
    """Lead melody + bass line mixed together."""
    lead_notes = [Note(ji_major[i], 0.5) for i in [4, 5, 7, 6, 5, 4, 2, 0]]
    bass_notes = [
        Note(ji_major[0], 1.0),
        Note(ji_major[4], 1.0),
        Note(ji_major[2], 1.0),
        Note(ji_major[5], 1.0),
    ]
    lead_gen = Oscillator("sine") + Oscillator("sine", ratio=2) * 0.2
    bass_gen = Oscillator("triangle") + Oscillator("triangle", ratio=2) * 0.4

    lead_p, lead_g = Sequencer(lead_notes, bpm=bpm).cv()
    bass_p, bass_g = Sequencer(bass_notes, bpm=bpm // 2).cv(repeats=2)

    lead_sig = lead_gen.at(lead_p).render(lead_p.duration) * note_env.trigger(lead_g)
    bass_sig = bass_gen.at(bass_p).render(bass_p.duration) * bass_env.trigger(bass_g)

    bass_fx = (Overdrive(gain=2.0) | LowPassFilter(600))(bass_sig)
    lead_fx = DatorroReverb(decay=0.4, wet=0.25)(lead_sig)

    mx = Mixer()
    mx.add_track(lead_fx, volume=0.55, position=0.0)
    mx.add_track(bass_fx, volume=0.70, position=0.0)
    return mx.render()


# ---------------------------------------------------------------------------
# 14. Spectral processing
# ---------------------------------------------------------------------------


def frozen_bell():
    """FM bell frozen at the attack transient — infinite sustain drone."""
    sig = bell(523, dur=2.0)
    return SpectralFreeze(freeze_time=0.02)(sig)


def smeared_pad(dur=5.0):
    """Saw pad with heavy spectral smear — blurs harmonics into a wash."""
    env = adsr(0.3, 0.2, dur - 1.0, 0.8, 0.5)
    sig = env.apply(Oscillator("saw").at(220).render(dur)) * 0.3
    return SpectralSmear(amount=12.0)(sig)


def pitch_shifted_melody():
    """Pentatonic melody harmonised a fifth up via spectral pitch shift."""
    notes = [Note(pentatonic[i], 0.5) for i in [0, 2, 4, 3, 1, 0]]
    pitch, gate = Sequencer(notes, bpm=100).cv(repeats=2)
    audio = Oscillator("sine").at(pitch).render(pitch.duration)
    dry = audio * note_env.trigger(gate)
    shifted = PitchShift(semitones=7)(dry)
    return dry * 0.6 + shifted * 0.4


def robot_voice():
    """Vocoder: white noise carrier shaped by a synthetic 'voice' modulator."""
    dur = 2.0
    # Modulator — a buzzy low tone that mimics vocal formants
    mod_env = adsr(0.05, 0.1, dur - 0.5, 0.7, 0.3)
    modulator = mod_env.apply(Oscillator("saw").at(120).render(dur))
    # Carrier — noise, which the vocoder reshapes
    carrier = WhiteNoise().render(dur)
    return Vocoder(modulator, n_fft=1024, mix=0.9)(carrier) * 0.3


def spectral_freeze_pad(dur=6.0):
    """Evolving pad frozen mid-morph, then reverbed for ambience."""
    wt = Wavetable.from_waveforms(["sine", "saw", "square"])
    # Render a morphing wavetable, then freeze it mid-sweep
    pos = Oscillator("triangle").at(0.5).render(2.0) * 1.0 + 1.0
    source = wt.at(220, position=pos).render(2.0) * 0.4
    spec = stft(source, n_fft=2048)
    # Freeze the frame where the morph is between saw and square
    frozen = freeze(spec, frame=spec.n_frames * 3 // 4)
    sig = frozen.to_signal()
    return DatorroReverb(decay=0.9, wet=0.55)(sig)


def spectral_chord():
    """Three frozen tones mixed in the frequency domain."""
    tones = [
        Oscillator("saw").at(ji_major[0].hz).render(1.0),
        Oscillator("saw").at(ji_major[2].hz).render(1.0),
        Oscillator("saw").at(ji_major[4].hz).render(1.0),
    ]
    specs = [stft(t, n_fft=2048) for t in tones]
    frozen = [freeze(s, frame=s.n_frames // 2) for s in specs]
    combined = frozen[0] * 0.33 + frozen[1] * 0.33 + frozen[2] * 0.33
    return combined.to_signal()


def bin_shift_riser(dur=4.0):
    """Noise swept upward through the spectrum — a tension riser."""
    sig = PinkNoise().render(dur) * 0.4
    spec = stft(sig, n_fft=2048)
    # Shift bins progressively higher across frames
    import numpy as np
    shifts = np.linspace(0, 80, spec.n_frames)
    new_frames = np.zeros_like(spec.frames)
    for i in range(spec.n_frames):
        shifted = shift_bins(
            type(spec)(
                spec.frames[i:i+1], spec.window, spec.hop_size,
                spec.sample_rate, spec.original_length,
            ),
            shift=int(shifts[i]),
        )
        new_frames[i] = shifted.frames[0]
    result = type(spec)(
        new_frames, spec.window, spec.hop_size,
        spec.sample_rate, spec.original_length,
    )
    return result.to_signal()


def convolution_clap():
    """Noise burst convolved with a short impulse — percussive clap."""
    import numpy as np
    burst = WhiteNoise().render(0.01)
    # Synthetic impulse response: three early reflections + decay tail
    n_ir = int(0.08 * SAMPLE_RATE)
    ir_data = np.zeros(n_ir, dtype=np.float32)
    ir_data[0] = 1.0
    ir_data[int(0.005 * SAMPLE_RATE)] = 0.7
    ir_data[int(0.012 * SAMPLE_RATE)] = 0.5
    ir_data[int(0.020 * SAMPLE_RATE)] = 0.3
    # Exponential decay tail
    t = np.arange(n_ir) / SAMPLE_RATE
    ir_data += np.float32(0.15 * np.exp(-t * 60) * np.random.default_rng(0).standard_normal(n_ir))
    ir = Signal(ir_data)
    return ConvolutionReverb(ir, wet=1.0)(burst)


def cross_synth_textures():
    """Cross-synthesis: bell magnitude stamped onto noise phase — glassy texture."""
    dur = 3.0
    bell_sig = bell(880, dur=dur)
    noise = PinkNoise().render(dur) * 0.5
    bell_spec = stft(bell_sig, n_fft=2048)
    noise_spec = stft(noise, n_fft=2048)
    result = cross_synthesize(noise_spec, bell_spec, mix=0.85)
    return DatorroReverb(decay=0.7, wet=0.4)(result.to_signal()) * 0.5


def spectral_smear_drone(dur=8.0):
    """Extreme spectral smear on a chord — blurs into a shimmering drone."""
    chord = (
        Oscillator("saw").at(ji_major[0].hz).render(dur) * 0.2
        + Oscillator("saw").at(ji_major[2].hz).render(dur) * 0.2
        + Oscillator("saw").at(ji_major[4].hz).render(dur) * 0.2
    )
    env = adsr(0.5, 0.3, dur - 1.5, 0.8, 0.7)
    sig = env.apply(chord)
    spec = stft(sig, n_fft=4096)
    smeared = smear(spec, amount=30.0)
    return DatorroReverb(decay=0.85, wet=0.5)(smeared.to_signal()) * 0.5


def vocoder_arp(bpm=120):
    """Arpeggiated carrier processed through a vocoder with a saw modulator."""
    dur = 4.0
    # Carrier: arpeggiated sine
    notes = [Note(ji_major[i], 0.25) for i in [0, 2, 4, 7]]
    pitch, gate = Arpeggiator(notes, pattern="up", note_duration=0.125, bpm=bpm).cv(bars=4)
    carrier = Oscillator("sine").at(pitch).render(pitch.duration) * note_env.trigger(gate)
    # Modulator: slow saw sweep — gives rhythmic vowel-like filtering
    mod = Oscillator("saw").at(80).render(pitch.duration) * 0.5
    mod_env = adsr(0.01, 0.3, pitch.duration - 0.8, 0.6, 0.4)
    modulator = mod_env.apply(mod)
    return Vocoder(modulator, n_fft=1024, mix=0.8)(carrier) * 0.4


# ---------------------------------------------------------------------------
# 15. TR-808 drum machine
# ---------------------------------------------------------------------------


def tr808_kick(decay=0.7):
    """808 kick: sine with fast pitch sweep from ~160 Hz down to ~45 Hz,
    plus a short click transient for the attack."""
    import numpy as np

    dur = 0.05 + decay
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    # exponential pitch sweep: starts at ~160 Hz, decays to ~45 Hz
    pitch_sig = Signal(np.float32(45.0 + 115.0 * np.exp(-t * 35.0)))
    body = Oscillator("sine").at(pitch_sig).render(dur)
    # amplitude envelope: fast attack, long exponential decay
    amp = Envelope([
        Segment(0.004, 0.0, 1.0, curve=-4),
        Segment(decay, 1.0, 0.0, curve=4),
    ])
    # click transient from a high-pitched burst
    click_env = Envelope([Segment(0.003, 1.0, 0.0, curve=-6)])
    click = click_env.apply(Oscillator("sine").at(160).render(dur)) * 0.6
    return amp.apply(body) + click


def tr808_snare():
    """808 snare: pitched sine body + bandpass-filtered noise rattle."""
    dur = 0.25
    # body: two sine partials with fast pitch drop
    import numpy as np
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pitch1 = Signal(np.float32(180.0 + 40.0 * np.exp(-t * 50.0)))
    pitch2 = Signal(np.float32(330.0 + 40.0 * np.exp(-t * 50.0)))
    body_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur - 0.001, 1.0, 0.0, curve=3),
    ])
    body = body_env.apply(
        Oscillator("sine").at(pitch1).render(dur) * 0.5
        + Oscillator("sine").at(pitch2).render(dur) * 0.3
    )
    # noise rattle
    noise_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur - 0.001, 1.0, 0.0, curve=2),
    ])
    noise = noise_env.apply(
        BandPassFilter(2000, 9000)(WhiteNoise().render(dur))
    )
    return body * 0.6 + noise * 0.5


def tr808_clap():
    """808 clap: multiple noise bursts spread ~5 ms apart, then a tail."""
    import numpy as np

    burst_dur = 0.012
    gap = 0.005
    tail_dur = 0.18
    total = burst_dur * 4 + gap * 3 + tail_dur
    result = Signal.silence(total)
    burst_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(burst_dur - 0.001, 1.0, 0.0, curve=-2),
    ])
    # 4 rapid bursts
    for i in range(4):
        offset = int((burst_dur + gap) * i * SAMPLE_RATE)
        burst = burst_env.apply(
            BandPassFilter(1200, 3500)(WhiteNoise().render(burst_dur))
        )
        end = min(len(result.data), offset + len(burst.data))
        result.data[offset:end] += burst.data[:end - offset] * 0.7
    # reverberant noise tail
    tail_start = int((burst_dur + gap) * 3 * SAMPLE_RATE)
    tail_env = Envelope([
        Segment(0.005, 0.8, 1.0, curve=-2),
        Segment(tail_dur - 0.005, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(800, 3500)(WhiteNoise().render(tail_dur))
    )
    end = min(len(result.data), tail_start + len(tail.data))
    result.data[tail_start:end] += tail.data[:end - tail_start] * 0.5
    return result


def tr808_hihat(open=False):
    """808 hi-hat: six square oscillators at metallic ratios, highpass filtered."""
    dur = 0.3 if open else 0.05
    # the original 808 uses six metal-square oscillators at these frequencies
    freqs = [204.5, 298.5, 366.5, 522.7, 540.5, 800.6]
    metal = sum(
        Oscillator("square").at(f).render(dur + 0.01) * (1.0 / len(freqs))
        for f in freqs
    )
    filtered = (HighPassFilter(7000) | BandPassFilter(7500, 12000))(metal)
    if open:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-4),
            Segment(dur - 0.001, 1.0, 0.0, curve=2),
        ])
    else:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-4),
            Segment(dur - 0.001, 1.0, 0.0, curve=-3),
        ])
    return env.apply(filtered) * 0.5


def tr808_cowbell():
    """808 cowbell: two square oscillators at 540 Hz and 800 Hz."""
    dur = 0.12
    sig = (
        Oscillator("square").at(540).render(dur) * 0.5
        + Oscillator("square").at(800).render(dur) * 0.5
    )
    filtered = BandPassFilter(500, 3000)(sig)
    env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(0.03, 1.0, 0.6, curve=1),
        Segment(dur - 0.031, 0.6, 0.0, curve=2),
    ])
    return env.apply(filtered) * 0.4


def tr808_tom(pitch=100, decay=0.3):
    """808 tom: sine with pitch sweep, variable tuning."""
    import numpy as np
    dur = decay + 0.01
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pitch_sig = Signal(np.float32(pitch + pitch * 0.5 * np.exp(-t * 30.0)))
    env = Envelope([
        Segment(0.002, 0.0, 1.0, curve=-4),
        Segment(decay, 1.0, 0.0, curve=3),
    ])
    return env.apply(Oscillator("sine").at(pitch_sig).render(dur))


def tr808_rimshot():
    """808 rimshot: short pitched triangle + noise click."""
    dur = 0.03
    tone_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-6),
        Segment(dur - 0.001, 1.0, 0.0, curve=-3),
    ])
    tone = tone_env.apply(Oscillator("triangle").at(500).render(dur))
    noise = tone_env.apply(
        HighPassFilter(2000)(WhiteNoise().render(dur))
    )
    return tone * 0.6 + noise * 0.4


def tr808_pattern(bpm=126):
    """Classic 808 boom-bap pattern: kick, snare, hats, with echo."""
    step = 60.0 / bpm / 2  # 16th note duration
    bars = 2
    total = step * 16 * bars
    result = Signal.silence(total)

    k = tr808_kick()
    s = tr808_snare()
    ch = tr808_hihat(open=False)
    oh = tr808_hihat(open=True)
    cb = tr808_cowbell()

    def place(sig, beat_16th):
        n = int(beat_16th * step * SAMPLE_RATE)
        end = min(len(result.data), n + len(sig.data))
        result.data[n:end] += sig.data[:end - n]

    for bar in range(bars):
        off = bar * 16
        # kick: 1, 4, 11 (syncopated)
        for b in [0, 3, 10]:
            place(k, off + b)
        # snare: 4, 12 (beats 2 and 4)
        for b in [4, 12]:
            place(s * 0.8, off + b)
        # closed hat on every other 16th
        for b in range(0, 16, 2):
            place(ch * 0.4, off + b)
        # open hat on offbeats
        for b in [2, 6, 14]:
            place(oh * 0.3, off + b)
        # cowbell accent
        place(cb * 0.3, off + 8)

    return Echo(delay_time=step * 3, repeats=2, decay=0.3, wet=0.15)(result)


# ---------------------------------------------------------------------------
# 16. TR-909 drum machine
# ---------------------------------------------------------------------------


def tr909_kick():
    """909 kick: punchier than 808 — shorter pitch sweep, harder attack."""
    import numpy as np
    dur = 0.35
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    # faster sweep, starts higher for more punch
    pitch_sig = Signal(np.float32(50.0 + 200.0 * np.exp(-t * 55.0)))
    body = Oscillator("sine").at(pitch_sig).render(dur)
    amp = Envelope([
        Segment(0.002, 0.0, 1.0, curve=-6),
        Segment(0.30, 1.0, 0.0, curve=3),
    ])
    # harder click transient
    click_env = Envelope([Segment(0.002, 1.0, 0.0, curve=-8)])
    click = click_env.apply(WhiteNoise().render(dur)) * 0.3
    return Clip(threshold=0.9)(amp.apply(body) + click)


def tr909_snare():
    """909 snare: sine body + aggressive noise — more bite than 808."""
    dur = 0.2
    import numpy as np
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pitch_sig = Signal(np.float32(200.0 + 60.0 * np.exp(-t * 60.0)))
    body_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-6),
        Segment(dur - 0.001, 1.0, 0.0, curve=2),
    ])
    body = body_env.apply(Oscillator("sine").at(pitch_sig).render(dur))
    # the 909 noise is brighter and more prominent
    noise_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur * 0.7, 1.0, 0.15, curve=2),
        Segment(dur * 0.3 - 0.001, 0.15, 0.0, curve=1),
    ])
    noise = noise_env.apply(
        HighPassFilter(3000)(WhiteNoise().render(dur))
    )
    return body * 0.5 + noise * 0.6


def tr909_hihat(open=False):
    """909 hi-hat: six square oscillators at metallic ratios — brighter than 808."""
    dur = 0.4 if open else 0.04
    # 909 uses similar metallic oscillator approach but different tuning
    freqs = [205.3, 304.4, 369.6, 522.7, 540.5, 811.2]
    metal = sum(
        Oscillator("square").at(f).render(dur + 0.01) * (1.0 / len(freqs))
        for f in freqs
    )
    # 909 hats are brighter — higher filter cutoff
    filtered = (HighPassFilter(9000) | BandPassFilter(9000, 14000))(metal)
    if open:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-4),
            Segment(dur * 0.6, 1.0, 0.3, curve=1),
            Segment(dur * 0.4 - 0.001, 0.3, 0.0, curve=2),
        ])
    else:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-6),
            Segment(dur - 0.001, 1.0, 0.0, curve=-4),
        ])
    return env.apply(filtered) * 0.5


def tr909_clap():
    """909 clap: tighter bursts than 808 with built-in room ambience."""
    import numpy as np
    burst_dur = 0.008
    gap = 0.003
    tail_dur = 0.14
    total = burst_dur * 4 + gap * 3 + tail_dur
    result = Signal.silence(total)
    burst_env = Envelope([
        Segment(0.0005, 0.0, 1.0, curve=-6),
        Segment(burst_dur - 0.0005, 1.0, 0.0, curve=-3),
    ])
    for i in range(4):
        offset = int((burst_dur + gap) * i * SAMPLE_RATE)
        burst = burst_env.apply(
            BandPassFilter(1000, 4000)(WhiteNoise().render(burst_dur))
        )
        end = min(len(result.data), offset + len(burst.data))
        result.data[offset:end] += burst.data[:end - offset] * 0.8
    tail_start = int((burst_dur + gap) * 3 * SAMPLE_RATE)
    tail_env = Envelope([
        Segment(0.003, 0.7, 1.0, curve=-2),
        Segment(tail_dur - 0.003, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(1000, 5000)(WhiteNoise().render(tail_dur))
    )
    end = min(len(result.data), tail_start + len(tail.data))
    result.data[tail_start:end] += tail.data[:end - tail_start] * 0.5
    return SimpleReverb(room_size=0.2, wet=0.3)(result)


def tr909_ride(dur=0.6):
    """909 ride cymbal: metallic oscillators with longer decay."""
    freqs = [205.3, 304.4, 369.6, 522.7, 540.5, 811.2, 1043.0]
    metal = sum(
        Oscillator("square").at(f).render(dur + 0.01) * (1.0 / len(freqs))
        for f in freqs
    )
    filtered = (HighPassFilter(5000) | BandPassFilter(6000, 11000))(metal)
    env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur * 0.3, 1.0, 0.4, curve=1),
        Segment(dur * 0.7 - 0.001, 0.4, 0.0, curve=2),
    ])
    return env.apply(filtered) * 0.35


def tr909_pattern(bpm=130):
    """Classic four-on-the-floor 909 pattern: kick every beat, snare on 2/4,
    hats driving 16ths."""
    step = 60.0 / bpm / 4  # 16th note
    bars = 2
    total = step * 16 * bars
    result = Signal.silence(total)

    k = tr909_kick()
    s = tr909_snare()
    ch = tr909_hihat(open=False)
    oh = tr909_hihat(open=True)
    cl = tr909_clap()
    ride = tr909_ride()

    def place(sig, beat_16th, vol=1.0):
        n = int(beat_16th * step * SAMPLE_RATE)
        end = min(len(result.data), n + len(sig.data))
        result.data[n:end] += sig.data[:end - n] * vol

    for bar in range(bars):
        off = bar * 16
        # four-on-the-floor kick
        for b in [0, 4, 8, 12]:
            place(k, off + b)
        # clap on 2 and 4
        for b in [4, 12]:
            place(cl, off + b, 0.7)
        # snare ghost notes
        for b in [7, 15]:
            place(s, off + b, 0.35)
        # closed hats on 16ths, open hat for offbeat accents
        for b in range(16):
            if b in [2, 6, 10, 14]:
                place(oh, off + b, 0.25)
            else:
                place(ch, off + b, 0.4)
        # ride accent
        place(ride, off + 0, 0.25)
        place(ride, off + 8, 0.2)

    return Limiter(ceiling_db=-1.5)(result)


# ---------------------------------------------------------------------------
# 17. 808/909 v2 — pure pysynth (no numpy), for A/B comparison
# ---------------------------------------------------------------------------
# The v1 functions use raw numpy for exponential pitch sweeps and manual
# data[] splicing.  These v2 variants use only Envelope, Segment, and Signal
# arithmetic so you can hear whether the Segment curve parameter is a
# sufficient substitute.


def tr808_kick_v2(decay=0.7):
    """v2: pitch sweep via Envelope instead of np.exp."""
    dur = 0.05 + decay
    # Segment(dur, start=1, end=0, curve=5) gives a fast-decaying convex curve.
    # Scale it: * 115 + 45 → sweeps from 160 Hz down to 45 Hz.
    pitch_sig = Envelope([Segment(dur, 1.0, 0.0, curve=-20)]).render(dur) * 115.0 + 45.0
    body = Oscillator("sine").at(pitch_sig).render(dur)
    amp = Envelope([
        Segment(0.004, 0.0, 1.0, curve=-4),
        Segment(decay, 1.0, 0.0, curve=4),
    ])
    click_env = Envelope([Segment(0.003, 1.0, 0.0, curve=-6)])
    click = click_env.apply(Oscillator("sine").at(160).render(dur)) * 0.6
    return amp.apply(body) + click


def tr808_snare_v2():
    """v2: pitch drop via Envelope instead of np.exp."""
    dur = 0.25
    pitch1 = Envelope([Segment(dur, 1.0, 0.0, curve=-6)]).render(dur) * 40.0 + 180.0
    pitch2 = Envelope([Segment(dur, 1.0, 0.0, curve=-6)]).render(dur) * 40.0 + 330.0
    body_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur - 0.001, 1.0, 0.0, curve=3),
    ])
    body = body_env.apply(
        Oscillator("sine").at(pitch1).render(dur) * 0.5
        + Oscillator("sine").at(pitch2).render(dur) * 0.3
    )
    noise_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur - 0.001, 1.0, 0.0, curve=2),
    ])
    noise = noise_env.apply(
        BandPassFilter(2000, 9000)(WhiteNoise().render(dur))
    )
    return body * 0.6 + noise * 0.5


def tr808_clap_v2():
    """v2: clap bursts via Sequencer gate pulses instead of data[] splicing.

    Uses a StepSequencer with very short steps to generate the burst timing,
    then triggers noise through an envelope.  The tail is a separate layer
    mixed on top."""
    burst_dur = 0.012
    gap = 0.005
    tail_dur = 0.18
    step_len = burst_dur + gap
    n_bursts = 4

    # 4 gate pulses via StepSequencer, each burst_dur long with a gap
    burst_steps = [Step(1.0, gate_length=burst_dur / step_len)] * n_bursts
    bpm = 60.0 / step_len  # one step = step_len seconds
    _, gate = StepSequencer(burst_steps, bpm=bpm, step_length=1.0).cv()

    # trigger filtered noise bursts from the gate
    burst_env = adsr(attack=0.001, decay=burst_dur - 0.002, sustain=0.0,
                     sustain_level=0.0, release=0.001)
    bursts_dur = gate.duration
    noise_src = BandPassFilter(1200, 3500)(WhiteNoise().render(bursts_dur))
    bursts = noise_src * burst_env.trigger(gate) * 0.7

    # reverberant noise tail, rendered separately and mixed
    tail_env = Envelope([
        Segment(0.005, 0.8, 1.0, curve=-2),
        Segment(tail_dur - 0.005, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(800, 3500)(WhiteNoise().render(tail_dur))
    ) * 0.5

    # mix: bursts then tail starts at the last burst
    total = bursts_dur + tail_dur
    result = Signal.silence(total)
    # layer bursts at start
    result = result + bursts
    # layer tail starting near the last burst — use silence padding
    tail_offset = step_len * (n_bursts - 1)
    tail_padded = Signal.silence(tail_offset) + tail
    result = result + tail_padded
    return result


def tr808_tom_v2(pitch=100, decay=0.3):
    """v2: pitch sweep via Envelope instead of np.exp."""
    dur = decay + 0.01
    # sweep from pitch*1.5 down to pitch
    pitch_sig = Envelope([Segment(dur, 1.0, 0.0, curve=-8)]).render(dur) * (pitch * 0.5) + pitch
    env = Envelope([
        Segment(0.002, 0.0, 1.0, curve=-4),
        Segment(decay, 1.0, 0.0, curve=3),
    ])
    return env.apply(Oscillator("sine").at(pitch_sig).render(dur))


def tr909_kick_v2():
    """v2: pitch sweep via Envelope instead of np.exp."""
    dur = 0.35
    pitch_sig = Envelope([Segment(dur, 1.0, 0.0, curve=-15)]).render(dur) * 200.0 + 50.0
    body = Oscillator("sine").at(pitch_sig).render(dur)
    amp = Envelope([
        Segment(0.002, 0.0, 1.0, curve=-6),
        Segment(0.30, 1.0, 0.0, curve=3),
    ])
    click_env = Envelope([Segment(0.002, 1.0, 0.0, curve=-8)])
    click = click_env.apply(WhiteNoise().render(dur)) * 0.3
    return Clip(threshold=0.9)(amp.apply(body) + click)


def tr909_snare_v2():
    """v2: pitch drop via Envelope instead of np.exp."""
    dur = 0.2
    pitch_sig = Envelope([Segment(dur, 1.0, 0.0, curve=-10)]).render(dur) * 60.0 + 200.0
    body_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-6),
        Segment(dur - 0.001, 1.0, 0.0, curve=2),
    ])
    body = body_env.apply(Oscillator("sine").at(pitch_sig).render(dur))
    noise_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur * 0.7, 1.0, 0.15, curve=2),
        Segment(dur * 0.3 - 0.001, 0.15, 0.0, curve=1),
    ])
    noise = noise_env.apply(
        HighPassFilter(3000)(WhiteNoise().render(dur))
    )
    return body * 0.5 + noise * 0.6


def tr909_clap_v2():
    """v2: clap bursts via StepSequencer instead of data[] splicing."""
    burst_dur = 0.008
    gap = 0.003
    tail_dur = 0.14
    step_len = burst_dur + gap
    n_bursts = 4

    burst_steps = [Step(1.0, gate_length=burst_dur / step_len)] * n_bursts
    bpm = 60.0 / step_len
    _, gate = StepSequencer(burst_steps, bpm=bpm, step_length=1.0).cv()

    burst_env = adsr(attack=0.0005, decay=burst_dur - 0.001, sustain=0.0,
                     sustain_level=0.0, release=0.0005)
    bursts_dur = gate.duration
    noise_src = BandPassFilter(1000, 4000)(WhiteNoise().render(bursts_dur))
    bursts = noise_src * burst_env.trigger(gate) * 0.8

    tail_env = Envelope([
        Segment(0.003, 0.7, 1.0, curve=-2),
        Segment(tail_dur - 0.003, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(1000, 5000)(WhiteNoise().render(tail_dur))
    ) * 0.5

    total = bursts_dur + tail_dur
    result = Signal.silence(total)
    result = result + bursts
    tail_offset = step_len * (n_bursts - 1)
    tail_padded = Signal.silence(tail_offset) + tail
    result = result + tail_padded
    return SimpleReverb(room_size=0.2, wet=0.3)(result)


# ---------------------------------------------------------------------------
# 18. 808/909 v3 — using Signal.shift() for event placement
# ---------------------------------------------------------------------------
# v1 used data[] splicing, v2 used StepSequencer (verbose).
# v3 uses Signal.shift(seconds) + addition — the natural Signal algebra approach.


def tr808_clap_v3():
    """v3: clap bursts placed with shift() + addition."""
    burst_dur = 0.012
    gap = 0.005
    tail_dur = 0.18
    step = burst_dur + gap
    burst_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(burst_dur - 0.001, 1.0, 0.0, curve=-2),
    ])
    burst = burst_env.apply(
        BandPassFilter(1200, 3500)(WhiteNoise().render(burst_dur))
    ) * 0.7
    tail_env = Envelope([
        Segment(0.005, 0.8, 1.0, curve=-2),
        Segment(tail_dur - 0.005, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(800, 3500)(WhiteNoise().render(tail_dur))
    ) * 0.5
    return sum(burst.shift(i * step) for i in range(4)) + tail.shift(3 * step)


def tr909_clap_v3():
    """v3: 909 clap with shift()."""
    burst_dur = 0.008
    gap = 0.003
    tail_dur = 0.14
    step = burst_dur + gap
    burst_env = Envelope([
        Segment(0.0005, 0.0, 1.0, curve=-6),
        Segment(burst_dur - 0.0005, 1.0, 0.0, curve=-3),
    ])
    burst = burst_env.apply(
        BandPassFilter(1000, 4000)(WhiteNoise().render(burst_dur))
    ) * 0.8
    tail_env = Envelope([
        Segment(0.003, 0.7, 1.0, curve=-2),
        Segment(tail_dur - 0.003, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(1000, 5000)(WhiteNoise().render(tail_dur))
    ) * 0.5
    return SimpleReverb(room_size=0.2, wet=0.3)(
        sum(burst.shift(i * step) for i in range(4)) + tail.shift(3 * step)
    )


def tr808_pattern_v3(bpm=126):
    """v3: 808 pattern using shift() instead of data[] splicing."""
    step = 60.0 / bpm / 2
    k = tr808_kick()
    s = tr808_snare_v2()
    ch = tr808_hihat(open=False)
    oh = tr808_hihat(open=True)
    cb = tr808_cowbell()

    def at(sig, *beats):
        return sum(sig.shift(b * step) for b in beats)

    result = (
        at(k, 0, 3, 10, 16, 19, 26)
        + at(s * 0.8, 4, 12, 20, 28)
        + at(ch * 0.4, *range(0, 32, 2))
        + at(oh * 0.3, 2, 6, 14, 18, 22, 30)
        + at(cb * 0.3, 8, 24)
    )
    return Echo(delay_time=step * 3, repeats=2, decay=0.3, wet=0.15)(result)


def tr909_pattern_v3(bpm=130):
    """v3: 909 pattern using shift() instead of data[] splicing."""
    step = 60.0 / bpm / 4
    k = tr909_kick_v2()
    s = tr909_snare_v2()
    ch = tr909_hihat(open=False)
    oh = tr909_hihat(open=True)
    cl = tr909_clap_v3()
    ride = tr909_ride()

    def at(sig, *beats):
        return sum(sig.shift(b * step) for b in beats)

    offbeat = [2, 6, 10, 14, 18, 22, 26, 30]
    onbeat = [b for b in range(32) if b not in offbeat]
    result = (
        at(k, 0, 4, 8, 12, 16, 20, 24, 28)
        + at(cl * 0.7, 4, 12, 20, 28)
        + at(s * 0.35, 7, 15, 23, 31)
        + at(ch * 0.4, *onbeat)
        + at(oh * 0.25, *offbeat)
        + at(ride * 0.25, 0, 16)
        + at(ride * 0.2, 8, 24)
    )
    return Limiter(ceiling_db=-1.5)(result)


def drum_pattern_v3():
    """v3 of drum_pattern (section 8) — rewritten with shift()."""
    k, s, h = kick(), snare(), hihat()
    step = 0.5  # 120 bpm quarter note

    def at(sig, *beats):
        return sum(sig.shift(b * step) for b in beats)

    return (
        at(h * 0.5, 0, 1, 2, 3)
        + at(k, 0, 2)
        + at(s, 1, 3)
    )
