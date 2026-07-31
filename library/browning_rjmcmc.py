"""Random-histogram discrete-time Hawkes process with a negative-binomial observation
layer (Browning, Rousseau & Mengersen 2022, "A flexible, random histogram kernel for
discrete-time Hawkes processes", arXiv:2208.02921), fitted by reversible-jump MCMC.

Model (faithful Browning; baseline="intercept"):

    lambda_t = mu + sum_{d=1..D} N_{t-d} f(d),     N_t | H_{t-1} ~ NegBin(mean=lambda_t, size=kappa)

with mu > 0 a CONSTANT baseline and f(d) >= 0 a piecewise-constant excitation kernel on a
RANDOM partition of the lag grid {1,...,D} into contiguous bins (the "random histogram").
The identity link keeps lambda_t > 0 by construction.  Browning uses Poisson counts; we use
the negative-binomial observation layer of the paper so the comparison to GP-DHP is on the
same observation model (as done for the fixed-bin Histogram DHP-NB benchmark).

Optional baseline="seasonal": eta_t = beta0 + sum_j beta_j B_j(t) + sum_d N_{t-d} f(d), with
B the intercept+Fourier design used by the fixed-bin Histogram DHP-NB and lambda_t=max(eta_t,0);
this pairs Browning's random-histogram *kernel* with the paper's seasonal baseline, isolating
the effect of the baseline from that of the kernel.

Inference: reversible-jump MCMC (Green 1995).  Each sweep does fixed-dimension RWM updates of
log mu (or the Fourier betas), log kappa, and the per-bin log heights, and one birth/death move
that adds/removes a single histogram break on the discrete interior grid {1,...,D-1}.  Heights
are split/merged by a length-weighted, log-mean-preserving map with Jacobian |h_L h_R / h|; the
uniform break-position prior and the uniform birth/death proposal cancel, leaving only the
Poisson prior on the number of breaks and the birth/death selection probabilities.

Prediction: the one-step-ahead posterior-predictive NB log-score on the held-out test set,
evaluated by log-sum-exp over the retained draws -- exactly the GP-DHP Bayesian predictive.
"""
import os
import sys
import numpy as np
from scipy.special import gammaln, logsumexp

from .jax_baselines import (dhp_lag_matrix, dhp_lag_matrix_future,   # noqa: E402  (shared conventions)
                           hist_baseline_design, HIST_CFG)

FLOOR = 1e-6                                                       # matches jax_baselines.FLOOR


# --------------------------------------------------------------------------- #
#  design: cumulative lag sums so any contiguous bin-sum is a column difference #
# --------------------------------------------------------------------------- #
def _cum_lag(y, D):
    """cumX[t,j] = sum_{d=1..j} y[t-d]  (cumX[:,0]=0);  bin {a+1..b} sum = cumX[:,b]-cumX[:,a]."""
    X = dhp_lag_matrix(np.asarray(y, float), D)                   # (T,D): X[t,d-1]=y[t-d]
    return np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, axis=1)], axis=1)   # (T,D+1)


def _cum_lag_future(y_dev, y_test, D):
    Xf = dhp_lag_matrix_future(np.asarray(y_dev, float), np.asarray(y_test, float), D)  # (n_test,D)
    return np.concatenate([np.zeros((Xf.shape[0], 1)), np.cumsum(Xf, axis=1)], axis=1)


def _seasonal_design(period, n, K_fourier=2):
    t = np.arange(1, n + 1, dtype=float)
    cols = [np.ones(n)]
    for k in range(1, K_fourier + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols)                                  # (n, 1+2K)


# --------------------------------------------------------------------------- #
#  priors (log densities)                                                       #
# --------------------------------------------------------------------------- #
def _lgamma_pdf(x, a, b):
    """log Gamma(shape=a, rate=b) density (full, with normalizer -- needed for RJ moves)."""
    return a * np.log(b) - gammaln(a) + (a - 1.0) * np.log(x) - b * x


def _lnorm_pdf(x, s):
    return -0.5 * np.log(2 * np.pi * s * s) - 0.5 * (x / s) ** 2


class Prior:
    def __init__(self, ybar, g_mean=None, s_g=2.5, mu_mean=None, kappa_mean=50.0,
                 lam_breaks=3.0, beta_sd=1.0e3):    # near-flat seasonal baseline (unregularized, count scale)
        # log-height prior g=log f(d) ~ N(m_g, s_g^2); the linear split g_L=g+(nR/n)xi keeps
        # the reversible-jump Jacobian exactly 1 (no small-height blow-up in raw height space).
        self.m_g = float(np.log(0.05) if g_mean is None else g_mean)
        self.s_g = float(s_g)
        mu_mean = ybar if mu_mean is None else mu_mean
        self.a_mu, self.b_mu = 1.0, 1.0 / mu_mean                 # mu ~ Exp(mean ybar)
        self.a_k, self.b_k = 1.0, 1.0 / kappa_mean               # kappa ~ Exp(mean 50)
        self.lam_breaks = lam_breaks                              # #breaks m ~ Poisson(lam_breaks)
        self.beta_sd = beta_sd                                    # seasonal betas ~ N(0, beta_sd^2)

    def lg(self, g):      return _lnorm_pdf(g - self.m_g, self.s_g)   # log-height prior N(m_g, s_g^2)
    def lmu(self, mu):    return _lgamma_pdf(mu, self.a_mu, self.b_mu)
    def lkap(self, k):    return _lgamma_pdf(k, self.a_k, self.b_k)


# --------------------------------------------------------------------------- #
#  NB log-likelihood (numpy; caches the kappa-dependent gamma terms)            #
# --------------------------------------------------------------------------- #
def _ll(lam, y, kappa, gyk, gk, gy1):
    lam = np.maximum(lam, FLOOR)
    s = kappa + lam
    return float(np.sum(gyk - gk - gy1 + kappa * (np.log(kappa) - np.log(s))
                        + y * (np.log(lam) - np.log(s))))


def _bd_probs(m, D):
    """Birth/death selection probs at m interior breaks on a grid with D-1 candidate positions."""
    if m <= 0:
        return 1.0, 0.0
    if m >= D - 1:
        return 0.0, 1.0
    return 0.5, 0.5


# --------------------------------------------------------------------------- #
#  one reversible-jump chain                                                    #
# --------------------------------------------------------------------------- #
def run_chain(cumX, y, prior, baseline="intercept", B=None, link="identity",
              D=100, n_burn=8000, n_samp=40000, thin=20, seed=0,
              init_breaks=2, adapt=True):
    """Returns dict with retained draws (edges/heights/mu/kappa[/beta]) and diagnostics."""
    rng = np.random.default_rng(seed)
    T = len(y)
    gy1 = gammaln(y + 1.0)
    seasonal = (baseline != "intercept")                         # intercept=faithful Browning; else B@beta baseline
    nb_cols = B.shape[1] if seasonal else 0

    def link_lam(eta):
        return eta if link == "identity" else np.maximum(eta, 0.0)

    # ---- initial state -----------------------------------------------------
    edges = np.unique(np.concatenate([[0], np.linspace(0, D, init_breaks + 2)[1:-1].round(),
                                       [D]])).astype(int)
    edges = np.unique(np.clip(edges, 0, D)); edges[0], edges[-1] = 0, D
    K = len(edges) - 1
    g = np.full(K, prior.m_g)                                     # log-heights g=log f(d)
    mu = float(max(np.mean(y), 1.0))
    kappa = 50.0
    beta = np.zeros(nb_cols)
    if seasonal:
        beta[0] = 0.5 * max(np.mean(y), 1.0)                      # count-scale intercept start (relu link)
        base = B @ beta
    else:
        base = np.full(T, mu)

    def bin_col(a, b):                                            # sum over lags {a+1..b}
        return cumX[:, b] - cumX[:, a]

    exc = np.zeros(T)
    for k in range(K):
        exc += np.exp(g[k]) * bin_col(edges[k], edges[k + 1])
    gyk = gammaln(y + kappa); gk = gammaln(kappa)
    cur = _ll(link_lam(base + exc), y, kappa, gyk, gk, gy1)

    # ---- step sizes (adapted in burn-in) -----------------------------------
    s_mu, s_h, s_k, s_beta, s_split = 0.15, 0.35, 0.15, 0.10, 0.6
    acc = dict(mu=[0, 0], h=[0, 0], k=[0, 0], beta=[0, 0], move=[0, 0], birth=[0, 0], death=[0, 0])

    draws = []
    Ktrace = []
    total = n_burn + n_samp
    for it in range(total):
        burn = it < n_burn

        # --- baseline: log mu (intercept) or block betas (seasonal) ---------
        if seasonal:
            j = rng.integers(nb_cols)
            prop = beta.copy(); prop[j] += s_beta * rng.standard_normal()
            base_p = base + (prop[j] - beta[j]) * B[:, j]
            new = _ll(link_lam(base_p + exc), y, kappa, gyk, gk, gy1)
            lpri = _lnorm_pdf(prop[j], prior.beta_sd) - _lnorm_pdf(beta[j], prior.beta_sd)
            if np.log(rng.random()) < new - cur + lpri:
                beta, base, cur = prop, base_p, new; acc["beta"][0] += 1
            acc["beta"][1] += 1
        else:
            lmp = np.log(mu) + s_mu * rng.standard_normal(); mup = np.exp(lmp)
            base_p = base + (mup - mu)
            new = _ll(link_lam(base_p + exc), y, kappa, gyk, gk, gy1)
            # proposal on log mu => +log(mup/mu) Jacobian; prior on mu
            if np.log(rng.random()) < new - cur + prior.lmu(mup) - prior.lmu(mu) + (lmp - np.log(mu)):
                mu, base, cur = mup, base_p, new; acc["mu"][0] += 1
            acc["mu"][1] += 1

        # --- kappa (log RWM) ------------------------------------------------
        lkp = np.log(kappa) + s_k * rng.standard_normal(); kap_p = np.exp(lkp)
        gyk_p, gk_p = gammaln(y + kap_p), gammaln(kap_p)
        new = _ll(link_lam(base + exc), y, kap_p, gyk_p, gk_p, gy1)
        if np.log(rng.random()) < new - cur + prior.lkap(kap_p) - prior.lkap(kappa) + (lkp - np.log(kappa)):
            kappa, gyk, gk, cur = kap_p, gyk_p, gk_p, new; acc["k"][0] += 1
        acc["k"][1] += 1

        # --- per-bin log-heights (Gaussian RWM on g=log f) ------------------
        for k in range(len(g)):
            col = bin_col(edges[k], edges[k + 1])
            gp = g[k] + s_h * rng.standard_normal()
            exc_p = exc + (np.exp(gp) - np.exp(g[k])) * col
            new = _ll(link_lam(base + exc_p), y, kappa, gyk, gk, gy1)
            if np.log(rng.random()) < new - cur + prior.lg(gp) - prior.lg(g[k]):
                g[k], exc, cur = gp, exc_p, new; acc["h"][0] += 1
            acc["h"][1] += 1

        # --- move a break: dimension-preserving resize (symmetric) ----------
        m = len(edges) - 2
        if m >= 1:
            j = int(rng.integers(m)) + 1
            lo, hi = edges[j - 1] + 1, edges[j + 1] - 1           # keep both adjacent bins non-empty
            if hi > lo:
                e_new = int(rng.integers(lo, hi + 1))
                if e_new != edges[j]:
                    dcol = (np.exp(g[j - 1]) - np.exp(g[j])) * (cumX[:, e_new] - cumX[:, edges[j]])
                    new = _ll(link_lam(base + exc + dcol), y, kappa, gyk, gk, gy1)
                    if np.log(rng.random()) < new - cur:         # uniform position prior + symmetric proposal
                        edges = edges.copy(); edges[j] = e_new
                        exc, cur = exc + dcol, new; acc["move"][0] += 1
                    acc["move"][1] += 1

        # --- birth / death (reversible jump), a few attempts per sweep -------
        for _bd in range(2):
            m = len(edges) - 2                                    # interior breaks
            b_cur, d_cur = _bd_probs(m, D)
            if rng.random() < b_cur:                             # BIRTH
                acc["birth"][1] += 1
                free = np.setdiff1d(np.arange(1, D), edges[1:-1])
                if free.size:
                    p = int(rng.choice(free))
                    k = int(np.searchsorted(edges, p) - 1)        # bin containing lag p (edges[k]<p<edges[k+1])
                    nL, nR = p - edges[k], edges[k + 1] - p
                    n = nL + nR
                    xi = s_split * rng.standard_normal()
                    gL = g[k] + (nR / n) * xi                     # length-weighted log-mean-preserving
                    gR = g[k] - (nL / n) * xi                     # linear split in g  =>  |Jacobian| = 1
                    SL, SR = bin_col(edges[k], p), bin_col(p, edges[k + 1])
                    exc_p = exc + np.exp(gL) * SL + np.exp(gR) * SR - np.exp(g[k]) * (SL + SR)
                    new = _ll(link_lam(base + exc_p), y, kappa, gyk, gk, gy1)
                    _, d_new = _bd_probs(m + 1, D)
                    log_alpha = (new - cur
                                 + prior.lg(gL) + prior.lg(gR) - prior.lg(g[k])
                                 + (np.log(prior.lam_breaks) - np.log(m + 1))   # Poisson p(m+1)/p(m)
                                 + (np.log(d_new) - np.log(b_cur))              # positions+choice cancel to d_new/b_cur
                                 - _lnorm_pdf(xi, s_split))                     # 1/q(xi); |Jacobian|=1 in g-space
                    if np.log(rng.random()) < log_alpha:
                        edges = np.insert(edges, k + 1, p)
                        g = np.insert(g, k + 1, gR); g[k] = gL
                        exc, cur = exc_p, new; acc["birth"][0] += 1
            else:                                                 # DEATH
                acc["death"][1] += 1
                if m >= 1:
                    j = int(rng.integers(m)) + 1                  # remove interior edge edges[j]
                    ga, gb = g[j - 1], g[j]
                    na, nb = edges[j] - edges[j - 1], edges[j + 1] - edges[j]
                    n = na + nb
                    gm = (na * ga + nb * gb) / n                  # inverse of the linear split
                    xi = ga - gb                                  # auxiliary the reverse birth would draw
                    Sa, Sb = bin_col(edges[j - 1], edges[j]), bin_col(edges[j], edges[j + 1])
                    exc_p = exc + np.exp(gm) * (Sa + Sb) - np.exp(ga) * Sa - np.exp(gb) * Sb
                    new = _ll(link_lam(base + exc_p), y, kappa, gyk, gk, gy1)
                    b_new, _ = _bd_probs(m - 1, D)
                    log_alpha = (new - cur
                                 + prior.lg(gm) - prior.lg(ga) - prior.lg(gb)
                                 + (np.log(m) - np.log(prior.lam_breaks))       # Poisson p(m-1)/p(m)
                                 + (np.log(b_new) - np.log(d_cur))
                                 + _lnorm_pdf(xi, s_split))                     # q(xi); |Jacobian|=1
                    if np.log(rng.random()) < log_alpha:
                        edges = np.delete(edges, j)
                        g = np.delete(g, j); g[j - 1] = gm
                        exc, cur = exc_p, new; acc["death"][0] += 1

        # --- light step adaptation during burn-in ---------------------------
        if adapt and burn and (it + 1) % 200 == 0:
            def rate(key):
                return acc[key][0] / max(acc[key][1], 1)
            s_mu *= np.exp(0.6 * (rate("mu") - 0.30)) if not seasonal else 1.0
            s_beta *= np.exp(0.6 * (rate("beta") - 0.30)) if seasonal else 1.0
            s_h *= np.exp(0.4 * (rate("h") - 0.30))
            s_k *= np.exp(0.6 * (rate("k") - 0.30))
            s_split *= np.exp(0.3 * ((rate("birth") + rate("death")) / 2 - 0.30))
            s_split = float(np.clip(s_split, 0.03, 4.0))
            for v in acc.values():
                v[0] = v[1] = 0

        # --- retain -----------------------------------------------------------
        if not burn and ((it - n_burn) % thin == 0):
            draws.append((edges.copy(), np.exp(g), float(mu), float(kappa),
                          beta.copy() if seasonal else None))
        if not burn:
            Ktrace.append(len(g))

    rates = {k: acc[k][0] / max(acc[k][1], 1) for k in acc}       # post-burn-in rates
    return dict(draws=draws, Ktrace=np.array(Ktrace), rates=rates,
                seasonal=seasonal, link=link, nb_cols=nb_cols)


# --------------------------------------------------------------------------- #
#  posterior-predictive one-step-ahead NB log-score on the test set             #
# --------------------------------------------------------------------------- #
def predict_logscores(chains, cumXf, y_test, period, B_test=None, link="identity"):
    """logscore_i = logsumexp_s NB(y_i | lambda_i^(s), kappa^(s)) - log S  (pooled over chains)."""
    y = np.asarray(y_test, float)
    n = len(y); gy1 = gammaln(y + 1.0)
    per_draw = []
    for ch in chains:
        for (edges, h, mu, kappa, beta) in ch["draws"]:
            exc = np.zeros(n)
            for k in range(len(h)):
                exc += h[k] * (cumXf[:, edges[k + 1]] - cumXf[:, edges[k]])
            base = (B_test @ beta) if ch["seasonal"] else np.full(n, mu)
            eta = base + exc
            lam = np.maximum(eta if link == "identity" else np.maximum(eta, 0.0), FLOOR)
            s = kappa + lam
            per_draw.append(gammaln(y + kappa) - gammaln(kappa) - gy1
                            + kappa * (np.log(kappa) - np.log(s)) + y * (np.log(lam) - np.log(s)))
    M = np.vstack(per_draw)                                       # (S, n)
    return logsumexp(M, axis=0) - np.log(M.shape[0])              # (n,)


def _matched_baseline(y_dev, y_test, period, daily, cal, dates, Zfull):
    """Baseline design IDENTICAL to the fixed-bin Histogram DHP-NB (intercept + one annual harmonic,
    plus a weekly harmonic for daily series) with the external-covariate block appended for cov cells.
    Non-intercept columns are z-scored by fit-period statistics so a single RWM step mixes all
    coefficients; returns (B_fit, B_test)."""
    T = len(y_dev)
    Bfull = np.asarray(hist_baseline_design(cal, period, HIST_CFG[bool(daily)]["base"], dates), float)
    if Zfull is not None:
        Bfull = np.column_stack([Bfull, np.asarray(Zfull, float)])
    m = Bfull[:T].mean(0); sd = Bfull[:T].std(0, ddof=0)
    for j in range(Bfull.shape[1]):
        if sd[j] > 1e-8 and not np.allclose(Bfull[:, j], Bfull[0, j]):    # non-constant -> standardize
            Bfull[:, j] = (Bfull[:, j] - m[j]) / sd[j]
    return Bfull[:T], Bfull[T:]


def fit_score_browning(y_dev, y_test, period, daily=False, baseline="intercept",
                       cal=None, dates=None, Zfull=None,
                       D_max=100, n_chains=4, n_burn=8000, n_samp=40000, thin=20,
                       K_fourier=2, seed=0, verbose=True):
    """Fit the random-histogram DHP-NB by RJMCMC on y_dev and return (test logscores, info).

    baseline="intercept"  : faithful Browning (constant baseline, identity link).
    baseline="matched"    : random-histogram kernel with the SAME seasonal[+dow]+covariate baseline as
                            the fixed-bin Histogram DHP-NB (requires cal/dates; Zfull for cov cells).
    baseline="seasonal"   : intercept + K annual harmonics, no covariates (standalone convenience)."""
    y_dev = np.asarray(y_dev, float); y_test = np.asarray(y_test, float)
    T = len(y_dev)
    cumX = _cum_lag(y_dev, D_max)
    cumXf = _cum_lag_future(y_dev, y_test, D_max)
    prior = Prior(ybar=float(np.mean(y_dev)))
    seasonal = (baseline != "intercept")
    link = "identity" if not seasonal else "relu"
    B = B_test = None
    if baseline == "matched":
        B, B_test = _matched_baseline(y_dev, y_test, period, daily, cal, dates, Zfull)
    elif baseline == "seasonal":
        B = _seasonal_design(period, T, K_fourier)
        t_test = np.arange(T + 1, T + len(y_test) + 1, dtype=float)
        cols = [np.ones(len(y_test))]
        for k in range(1, K_fourier + 1):
            cols += [np.sin(2 * np.pi * k * t_test / period), np.cos(2 * np.pi * k * t_test / period)]
        B_test = np.column_stack(cols)

    chains = []
    for c in range(n_chains):
        ch = run_chain(cumX, y_dev, prior, baseline=baseline, B=B, link=link, D=D_max,
                       n_burn=n_burn, n_samp=n_samp, thin=thin, seed=seed + c,
                       init_breaks=[1, 2, 4, 7][c % 4])
        chains.append(ch)
        if verbose:
            print(f"  [chain {c}] accept {'beta' if seasonal else 'mu'}={ch['rates'].get('beta' if seasonal else 'mu'):.2f} "
                  f"h={ch['rates']['h']:.2f} kap={ch['rates']['k']:.2f} move={ch['rates']['move']:.2f} "
                  f"birth={ch['rates']['birth']:.2f} death={ch['rates']['death']:.2f} | "
                  f"K mean={ch['Ktrace'].mean():.2f} [{ch['Ktrace'].min()},{ch['Ktrace'].max()}] "
                  f"| n_draws={len(ch['draws'])}", flush=True)
    ls = predict_logscores(chains, cumXf, y_test, period, B_test=B_test, link=link)
    Kall = np.concatenate([ch["Ktrace"] for ch in chains])
    info = dict(K_mean=float(Kall.mean()), K_med=float(np.median(Kall)),
                K_range=(int(Kall.min()), int(Kall.max())),
                n_draws=sum(len(ch["draws"]) for ch in chains),
                rates={k: float(np.mean([ch["rates"][k] for ch in chains])) for k in chains[0]["rates"]},
                baseline=baseline)
    return np.asarray(ls, float), info
