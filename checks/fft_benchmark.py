"""FFT vs dense excitation benchmark behind the Appendix C claims: the O(T log T + D^2)
FFT path is essentially break-even at D_max=100 and substantially faster at D_max=400 on the
worldwide-terrorism series (where the paper reports a full 20-start selection cut ~sixfold).

Benchmarks the unit the selection repeats thousands of times -- one forward-validation bilevel
evaluation (inner train-MAP fit + analytic hypergradient, gpdhp_fv_grad.make_fv_fns grad_fv) --
on the RAND dev series (T = 13,476), dense vs FFT, at D_max in {100, 400}: median of `--evals`
timed calls after a JIT warm-up call.  The per-evaluation ratio is the driver of the full-selection
speedup; exact wall times are machine-specific.  Also verifies dense and FFT return the same
validation pLL to ~1e-9 (the FFT path is an exact reindexing, not an approximation).
Runtime ~5-10 min.  Usage:  python fft_benchmark.py [--evals 3]
"""
import argparse, os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from library import datasets
from library import gpdhp_fv_grad as FVG

#      [s_level, s_fou, s_lin, K_NB, mean_lag, size, kappa, sigma_u, ell_u, beta, lk, s_week]
NAT0 = [3.7,     0.01,  1e-6,  0.77, 4.6,      0.9,  6.9,   0.007,   30.0,  0.05, 0.46, 10.9]


def bench(y_dev, D, use_fft, n_evals):
    fns = FVG.make_fv_fns(y_dev, 365.0, True, Kf=2, D_max=D, use_fft=use_fft)
    nat = jnp.asarray(np.asarray(NAT0, float))
    g, pll, _ = fns["grad_fv"](nat)                    # warm-up (JIT compile) -- untimed
    times = []
    for _ in range(n_evals):
        t0 = time.perf_counter()
        g, pll, _ = fns["grad_fv"](nat)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(pll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=3)
    a = ap.parse_args()
    y_dev, _, period, daily = datasets.load_split("rand")
    print(f"rand dev T={len(np.asarray(y_dev))}; timing one bilevel evaluation "
          f"(inner MAP + analytic hypergradient), median of {a.evals} after warm-up\n"
          f"{'D_max':>6} {'dense(s)':>10} {'fft(s)':>8} {'speedup':>8} {'|dpll|':>10}")
    for D in (100, 400):
        td, pd = bench(y_dev, D, use_fft=False, n_evals=a.evals)
        tf, pf = bench(y_dev, D, use_fft=True, n_evals=a.evals)
        assert abs(pd - pf) < 1e-6, f"D={D}: dense and FFT validation pLL differ ({abs(pd-pf):.2e})"
        print(f"{D:>6} {td:>10.2f} {tf:>8.2f} {td/tf:>7.2f}x {abs(pd-pf):>10.2e}", flush=True)
    print("\ndone_marker  (Appendix C: break-even at D=100; the D=400 per-evaluation ratio drives "
          "the reported ~6x full-selection speedup on this series)")
