"""Conceptual / illustrative paper figures, reproduced in Python.

These seven PNGs depend on NO fitted data -- they are analytic illustrations whose parameters are
fully specified in the manuscript captions, so nothing here reads a dataset or an npz:

  * discrete_hawkes_intensity_mu05.png       -- Fig 1: a toy discrete-time Hawkes intensity path
  * gp_realisations_beta{0.00,0.03,0.10}.png -- prior draws u(d) of the warped-RBF GP lag component
  * Kf_heatmap_beta{0.00,0.03,0.10}.png      -- the corresponding K_g covariance heatmaps

The GP kernel is the SAME warped-RBF the production model uses
(_gpdhp_common.warped_rbf_cholesky), re-expressed in numpy so the appendix
illustration and the fitted model share one kernel definition:
    a(d)   = sigma_u * exp(-beta * d / 2)                 (beta=0 -> a = sigma_u)
    w(d)   = (1 - exp(-beta * d)) / (beta * ell)          (beta=0 -> w = d / ell)
    K[d,d']= a(d) a(d') exp(-1/2 (w(d) - w(d'))^2) + eps^2 [d=d'],   d = 1..D.
Increasing beta at fixed ell damps the amplitude and warps time toward long lags, pulling the prior
excitation to zero at large d (see the manuscript's Appendix prior-illustration figure).

Usage:  python illustration_figures.py            # writes all 7 PNGs into ../../paper
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "paper")   # papercode/../paper
os.makedirs(OUT, exist_ok=True)

SIGMA_U, ELL, D = 1.0, 10.0, 100          # prior scale, lag length-scale, number of lags
BETAS = [0.00, 0.03, 0.10]
N_DRAWS = 6
GRID = dict(ls="--", color="0.85", lw=0.6)
plt.rcParams.update({"font.size": 12, "figure.facecolor": "white", "savefig.facecolor": "white"})


# ----------------------------------------------------- Fig 1: discrete Hawkes intensity path
def discrete_hawkes_intensity():
    mu, jump, decay = 0.5, 0.75, 0.5           # baseline, jump size phi(1), geometric decay
    events = [(3, 2), (7, 4), (10, 3)]         # (time, multiplicity)
    T = 20
    t = np.arange(0, T + 1)
    # excitation from an event at s is felt from t=s+1 on: phi(d)=jump*decay^(d-1), d=t-s>=1
    lam = np.full_like(t, mu, dtype=float)
    for s, m in events:
        d = t - s
        lam += np.where(d >= 1, m * jump * decay ** np.clip(d - 1, 0, None), 0.0)
    top = lam.max() * 1.15
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    ax.step(t, lam, where="post", color="black", lw=1.8, label="Intensity")
    ax.axhline(mu, color="red", ls="--", lw=1.4, label="Baseline")
    for i, (s, m) in enumerate(events):
        ax.axvline(s, color="black", ls=":", lw=1.2, label="Events" if i == 0 else None)
        ax.text(s, top * 0.965, f"× {m}", ha="center", va="top", fontsize=11)
    ax.set_xlim(0, T); ax.set_ylim(0, top)
    ax.set_xlabel(r"time index $t$"); ax.set_ylabel(r"intensity $\lambda(t)$")
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "discrete_hawkes_intensity_mu05.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------- warped-RBF kernel (numpy mirror of the model)
def warped_rbf_K(beta, sigma_u=SIGMA_U, ell=ELL, n=D, eps=1e-4):
    d = np.arange(1, n + 1, dtype=float)
    if abs(beta) < 1e-10:
        a = sigma_u * np.ones(n); w = d / ell
    else:
        a = sigma_u * np.exp(-beta * d / 2.0); w = (1.0 - np.exp(-beta * d)) / (beta * ell)
    K = np.outer(a, a) * np.exp(-0.5 * (w[:, None] - w[None, :]) ** 2)
    return K + (eps ** 2) * np.eye(n)


# ----------------------------------------------------- GP prior draws + covariance heatmaps
def gp_realisations(beta, z):
    K = warped_rbf_K(beta)
    L = np.linalg.cholesky(K + 1e-12 * np.eye(D))
    draws = (L @ z).T                                   # (N_DRAWS, D); shared whitened z across betas
    d = np.arange(1, D + 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    for u in draws:
        ax.plot(d, u, lw=1.4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlim(1, D); ax.set_xlabel(r"lag $d$"); ax.set_ylabel(r"$u(d)$")
    ax.set_title(rf"GP realisations ($n={N_DRAWS}$), $\beta={beta:.2f}$, $\ell_g={ELL:.1f}$")
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"gp_realisations_beta{beta:.2f}.png"), dpi=200)
    plt.close(fig)


def kf_heatmap(beta):
    K = warped_rbf_K(beta)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(K, origin="lower", extent=[1, D, 1, D], cmap="viridis", vmin=0.0, vmax=1.0,
                   aspect="auto")
    ax.set_xlabel(r"lag $d'$"); ax.set_ylabel(r"lag $d$")
    ax.set_title(rf"Covariance $K_g$, $\beta={beta:.2f}$, $\ell_g={ELL:.1f}$")
    ax.grid(True, **GRID)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"Kf_heatmap_beta{beta:.2f}.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    discrete_hawkes_intensity()
    z = np.random.default_rng(0).standard_normal((D, N_DRAWS))   # one whitened draw set, warped per beta
    for b in BETAS:
        gp_realisations(b, z)
        kf_heatmap(b)
    print("wrote 7 illustrative figures to", OUT)
