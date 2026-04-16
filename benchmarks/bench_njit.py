"""Benchmarks for numba @njit optimization of sample-level loops."""

import time

import numpy as np

from pysynth._core import Signal


def bench(name, fn, repeats=5):
    """Time fn() over repeats after a warmup call. Print min/mean."""
    fn()  # warmup (triggers JIT if applicable)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mn, avg = min(times), sum(times) / len(times)
    print(f"{name}: min={mn*1000:.1f}ms  mean={avg*1000:.1f}ms")


SR = 44100
DUR = 5.0
N = int(SR * DUR)
rng = np.random.default_rng(42)
NOISE = Signal(rng.uniform(-0.5, 0.5, N).astype(np.float32), SR)


def bench_compressor():
    from pysynth.effects.dynamics import Compressor

    comp = Compressor(threshold_db=-12.0, ratio=4.0, attack=0.005, release=0.1)
    bench("Compressor", lambda: comp(NOISE))


def bench_delay_fixed():
    from pysynth.effects.delay import Delay

    delay = Delay(delay_time=0.25, feedback=0.5, wet=0.5)
    bench("Delay (fixed)", lambda: delay(NOISE))


def bench_delay_modulated():
    from pysynth.effects.delay import Delay

    lfo = Signal(
        (np.sin(2 * np.pi * 0.5 * np.arange(N) / SR) * 0.01 + 0.02).astype(np.float32),
        SR,
    )
    delay = Delay(delay_time=lfo, feedback=0.4, wet=0.5)
    bench("Delay (modulated)", lambda: delay(NOISE))


def bench_simple_reverb():
    from pysynth.effects.reverb import SimpleReverb

    rev = SimpleReverb(room_size=0.5, damping=0.5, wet=0.3)
    bench("SimpleReverb", lambda: rev(NOISE))


def bench_dattorro_reverb():
    from pysynth.effects.reverb import DatorroReverb

    rev = DatorroReverb(decay=0.5, wet=0.4)
    bench("DatorroReverb", lambda: rev(NOISE))


def bench_envelope():
    from pysynth.envelopes.envelope import adsr

    env = adsr(0.01, 0.1, 0.7, 0.1)
    # Gate: 120 bpm, 50% duty cycle = note every 0.5s, gate high for 0.25s
    gate_data = np.zeros(N, dtype=np.float32)
    period = int(SR * 0.5)
    on_time = int(SR * 0.25)
    for start in range(0, N, period):
        gate_data[start : start + on_time] = 1.0
    gate = Signal(gate_data, SR)
    bench("Envelope trigger", lambda: env.trigger(gate))


def bench_filter_lowpass():
    from pysynth.effects.filters import LowPassFilter

    # Modulated cutoff: sweep 200-4000 Hz
    cutoff = Signal(
        (np.sin(2 * np.pi * 1.0 * np.arange(N) / SR) * 1900 + 2100).astype(np.float32),
        SR,
    )
    lpf = LowPassFilter(cutoff_hz=cutoff, order=4)
    bench("LowPassFilter (modulated)", lambda: lpf(NOISE))


def bench_filter_highpass():
    from pysynth.effects.filters import HighPassFilter

    cutoff = Signal(
        (np.sin(2 * np.pi * 0.5 * np.arange(N) / SR) * 400 + 500).astype(np.float32),
        SR,
    )
    hpf = HighPassFilter(cutoff_hz=cutoff, order=4)
    bench("HighPassFilter (modulated)", lambda: hpf(NOISE))


def bench_filter_bandpass():
    from pysynth.effects.filters import BandPassFilter

    low = Signal(
        (np.sin(2 * np.pi * 0.3 * np.arange(N) / SR) * 200 + 400).astype(np.float32),
        SR,
    )
    high = Signal(
        (np.sin(2 * np.pi * 0.3 * np.arange(N) / SR) * 500 + 2000).astype(np.float32),
        SR,
    )
    bpf = BandPassFilter(low_hz=low, high_hz=high, order=4)
    bench("BandPassFilter (modulated)", lambda: bpf(NOISE))


if __name__ == "__main__":
    import sys

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    benchmarks = {
        "compressor": bench_compressor,
        "delay_fixed": bench_delay_fixed,
        "delay_modulated": bench_delay_modulated,
        "simple_reverb": bench_simple_reverb,
        "dattorro_reverb": bench_dattorro_reverb,
        "envelope": bench_envelope,
        "filter_lowpass": bench_filter_lowpass,
        "filter_highpass": bench_filter_highpass,
        "filter_bandpass": bench_filter_bandpass,
    }

    if "all" in targets:
        for name, fn in benchmarks.items():
            fn()
    else:
        for t in targets:
            if t in benchmarks:
                benchmarks[t]()
            else:
                print(f"Unknown benchmark: {t}")
