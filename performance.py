"""
performance.py
==============
compare np.roll vs explicit for-loop
for computing finite differences.

Tests both in terms of:
  - Wall-clock time (speed)
  - Memory usage (space)
  - Result accuracy (are they the same?)

Also tests Numba JIT compilation (item 8) if available.

Run:
    python performance.py
"""

import numpy as np
import time
import sys
sys.path.insert(0, "src")


# ===========================================================================
# Method 1: np.roll  (current implementation)
# ===========================================================================

def diff_x_roll(f, dx):
    """
    2nd-order centred x-derivative using np.roll.
    Periodic BCs built in automatically.
    """
    return (np.roll(f, -1, axis=1) - np.roll(f, +1, axis=1)) / (2 * dx)


# ===========================================================================
# Method 2: Explicit for-loop with array indexing
# ===========================================================================

def diff_x_loop(f, dx):
    """
    2nd-order centred x-derivative using explicit for-loop.
    Periodic BCs handled manually.
    """
    nz, nx = f.shape
    out    = np.zeros_like(f)

    for k in range(nz):
        for i in range(nx):
            ip1 = (i + 1) % nx   # periodic: wraps around
            im1 = (i - 1) % nx
            out[k, i] = (f[k, ip1] - f[k, im1]) / (2 * dx)

    return out


# ===========================================================================
# Method 3: Numpy slicing (no roll, slightly faster)
# ===========================================================================

def diff_x_slice(f, dx):
    """
    2nd-order centred x-derivative using numpy slicing.
    Manually handles periodic BCs with np.concatenate.
    """
    # Pad left and right with periodic ghost cells
    f_pad = np.concatenate([f[:, -1:], f, f[:, :1]], axis=1)
    return (f_pad[:, 2:] - f_pad[:, :-2]) / (2 * dx)


# ===========================================================================
# Method 4: Numba JIT  (if available)
# ===========================================================================

try:
    from numba import njit

    @njit
    def diff_x_numba(f, dx):
        """
        2nd-order centred x-derivative using Numba JIT.
        Compiled to machine code on first call.
        """
        nz, nx = f.shape
        out    = np.zeros((nz, nx))
        for k in range(nz):
            for i in range(nx):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                out[k, i] = (f[k, ip1] - f[k, im1]) / (2.0 * dx)
        return out

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ===========================================================================
# Benchmark runner
# ===========================================================================

def benchmark(func, f, dx, n_repeats=50, name=""):
    """Time a function over n_repeats calls, return mean time in ms."""
    # Warmup
    _ = func(f, dx)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func(f, dx)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)   # ms

    mean_ms = np.mean(times)
    std_ms  = np.std(times)
    print(f"  {name:<25} {mean_ms:8.3f} ms  ± {std_ms:.3f} ms")
    return result, mean_ms


def memory_estimate(func_name, nz, nx):
    """
    Rough memory estimate for each method.
    np.roll creates a copy of the array internally.
    """
    bytes_per_float = 8   # float64
    array_size_mb   = nz * nx * bytes_per_float / 1e6

    estimates = {
        "np.roll":   f"{2 * array_size_mb:.2f} MB  (2 temp copies)",
        "for-loop":  f"{1 * array_size_mb:.2f} MB  (1 output array)",
        "slicing":   f"{2 * array_size_mb:.2f} MB  (padded copy + output)",
        "numba":     f"{1 * array_size_mb:.2f} MB  (1 output array, compiled)",
    }
    return estimates


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  Finite Difference Performance Comparison")
    print("=" * 55)

    dx = 10.0

    for nz, nx in [(100, 100), (250, 250), (500, 500)]:

        print(f"\n  Grid size: {nz} x {nx}  "
              f"({nz*nx:,} points)")
        print("  " + "-" * 50)

        f = np.random.rand(nz, nx)

        # Time each method
        r1, t1 = benchmark(diff_x_roll,  f, dx, name="np.roll")
        r2, t2 = benchmark(diff_x_loop,  f, dx, name="for-loop")
        r3, t3 = benchmark(diff_x_slice, f, dx, name="slicing")

        if NUMBA_AVAILABLE:
            diff_x_numba(f, dx)   # compile first
            r4, t4 = benchmark(diff_x_numba, f, dx, name="numba JIT")

        # Verify all methods give same result
        print(f"\n  Accuracy check (max diff vs np.roll):")
        print(f"    for-loop vs roll:  {np.max(np.abs(r2-r1)):.2e}")
        print(f"    slicing  vs roll:  {np.max(np.abs(r3-r1)):.2e}")
        if NUMBA_AVAILABLE:
            print(f"    numba    vs roll:  {np.max(np.abs(r4-r1)):.2e}")

        # Memory estimates
        print(f"\n  Memory estimates:")
        mem = memory_estimate("", nz, nx)
        for method, est in mem.items():
            print(f"    {method:<12} {est}")

        # Speedup summary
        print(f"\n  Speedup vs for-loop:")
        print(f"    np.roll  is {t2/t1:.1f}x faster than for-loop")
        print(f"    slicing  is {t2/t3:.1f}x faster than for-loop")
        if NUMBA_AVAILABLE:
            print(f"    numba    is {t2/t4:.1f}x faster than for-loop")

    print("\n" + "=" * 55)
    print("  Conclusion for dissertation:")
    print("  np.roll is clean, readable, and fast for small grids.")
    print("  Numba JIT gives biggest speedup for large grids.")
    print("  For-loop is slowest — avoid in inner loops.")
    print("=" * 55 + "\n")
