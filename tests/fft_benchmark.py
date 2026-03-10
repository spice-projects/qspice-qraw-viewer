from unittest import TestCase
import time
import json
from pathlib import Path

import numpy as np

from viewer.fft import compute_fft_many, resample_uniform, WindowFunction, FftOutput


class TestFftExtras(TestCase):

    def test_micro_benchmark_record_runtime(self):
        """Run a small FFT workload and record median runtime to build/bench.json.

        This test is intentionally non-failing; it records a runtime value that
        CI or developers can inspect to detect regressions over time.
        """
        n_signals = 4
        n_samples = 16384
        fs = 1000.0
        x = np.linspace(0.0, n_samples / fs, n_samples, endpoint=False)
        t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        y = np.vstack([np.sin(2 * np.pi * (50 + i * 10) * t) for i in range(n_signals)])

        # warmup
        compute_fft_many(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)

        runs = 3
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            compute_fft_many(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)
            times.append(time.perf_counter() - t0)

        median = float(sorted(times)[len(times) // 2])

        # ensure build directory exists and write runtime metadata for external inspection
        out = Path("build")
        out.mkdir(exist_ok=True)
        meta = {
            "micro_bench_fft_median_seconds": median,
            "n_signals": n_signals,
            "n_samples": n_samples,
        }
        (out / "bench_fft.json").write_text(json.dumps(meta))

    def test_resample_uniform_equivalence(self):
        """Verify that internal resampling matches explicit `resample_uniform`.

        We generate a non-uniform axis, compute the FFT using the public API
        (which resamples internally), and compare to resampling explicitly
        then calling the FFT on the uniform grid.
        """
        n = 2048
        # create a mildly non-uniform grid
        rng = np.random.default_rng(1)
        base = np.linspace(0.0, 1.0, n)
        jitter = (rng.random(n) - 0.5) * 0.001
        x_nonuniform = base + jitter
        x_nonuniform = np.sort(x_nonuniform)

        # single-tone signal
        f = 60.0
        y = np.sin(2 * np.pi * f * x_nonuniform)

        freqs_internal, values_internal = compute_fft_many(x_nonuniform, np.asarray([y]), window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)

        # explicit resample to uniform grid and recompute
        x_uniform, y_uniform = resample_uniform(x_nonuniform, y)
        freqs_explicit, values_explicit = compute_fft_many(x_uniform, np.asarray([y_uniform]), window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)

        # frequencies should match and magnitudes be numerically close
        np.testing.assert_allclose(freqs_internal, freqs_explicit, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(values_internal[0], values_explicit[0], rtol=1e-8, atol=1e-10)
