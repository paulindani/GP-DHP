"""Shared machinery for the GP-DHP (softplus link) collapsed-latent posteriors.

Implements, in pure JAX, the design of the Gaussian-Process Discrete Hawkes
Process used in the paper: everything is built from raw counts + the
selected hyperparameters.

The sampled parameter is the *whitened* latent coefficient vector theta with an
exact N(0, I) prior (the paper's collapsed parameterisation), so the
unconstrained space IS theta and the constrain/unconstrain maps are identity.
The negative log posterior is

    nlp(theta) = -sum_t logNB2(N_t | mu_t, kappa) + 0.5 * ||theta||^2,
    mu_t       = softplus_s(offset_t + (A theta)_t),        (+ floor)
    A          = [B | X L],   offset = X (K_NB q_NB),

with B the scaled baseline design, L the Cholesky of the warped-RBF lag
covariance K_g, X the lagged-count matrix, and softplus_s(x) = s*log(1+exp(x/s)).
A and offset are constants (hyperparameters are fixed), so nlp is a cheap
matvec + NB likelihood, fully vmap/grad-safe (no pure_callback needed).
"""
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)

DATA_ROOT = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Design pieces (rebuilt in JAX from raw counts + hyperparameters)
# --------------------------------------------------------------------------- #
def _fourier_design(t_idx, period, K):
    """Columns [sin1, cos1, sin2, cos2, ...] -- finite-Fourier harmonic design."""
    cols = []
    for k in range(1, K + 1):
        cols.append(jnp.sin(2 * jnp.pi * k * t_idx / period))
        cols.append(jnp.cos(2 * jnp.pi * k * t_idx / period))
    return jnp.stack(cols, axis=1)


def _baseline_cols(t_idx, period, h):
    """Scaled baseline columns at explicit time indices (level, harm, weekly, lin)."""
    n = t_idx.shape[0]
    parts = [h["sigma_level"] * jnp.ones((n, 1))]
    parts.append(h["sigma_fourier"] * _fourier_design(t_idx, period, int(h["K_fourier"])))
    if h["sigma_weekly"] > 0:
        parts.append(h["sigma_weekly"]
                     * _fourier_design(t_idx, h["period_weekly"], int(h["K_weekly"])))
    if h["sigma_lin"] > 0:
        parts.append((h["sigma_lin"] * t_idx)[:, None])
    return jnp.concatenate(parts, axis=1)


def baseline_design(T, period, h):
    """Scaled baseline design B (T x q_b) over time indices 1..T."""
    return _baseline_cols(jnp.arange(1, T + 1, dtype=jnp.float64), period, h)


def lag_matrix_future(y_train, y_test, D):
    """Future lag design X_test (n_test x D): X[i,d] = combined-history count at
    global test day (T_train + i) minus lag d, using [y_train, y_test] history."""
    yall = np.concatenate([np.asarray(y_train, float), np.asarray(y_test, float)])
    T = len(y_train); n_test = len(y_test)
    Xt = np.zeros((n_test, D), dtype=np.float64)
    for i in range(n_test):
        t = T + i                              # global 0-indexed day of test row i
        for j in range(D):
            s = t - (j + 1)
            if s >= 0:
                Xt[i, j] = yall[s]
    return jnp.asarray(Xt)


def warped_rbf_cholesky(D, beta, sigma_u, ell_u, epsilon_f=1e-4):
    """Lower-Cholesky L of the warped-RBF lag covariance K_g (D x D)."""
    d = jnp.arange(1, D + 1, dtype=jnp.float64)
    if abs(beta) < 1e-10:
        a = sigma_u * jnp.ones(D)
        g = d / ell_u
    else:
        a = sigma_u * jnp.exp(-beta * d / 2)
        g = (1.0 - jnp.exp(-beta * d)) / (beta * ell_u)
    K = jnp.outer(a, a) * jnp.exp(-0.5 * (g[:, None] - g[None, :]) ** 2)
    K = K + (epsilon_f ** 2) * jnp.eye(D)
    K = 0.5 * (K + K.T)
    return jnp.linalg.cholesky(K + 1e-12 * jnp.eye(D))   # lower factor, L L^T = K


def nb_kernel(D, mean_lag, size):
    """Normalised negative-binomial excitation shape q_NB (length D)."""
    k = jnp.arange(0, D, dtype=jnp.float64)
    p = size / (size + mean_lag)
    logpmf = (gammaln(k + size) - gammaln(size) - gammaln(k + 1)
              + size * jnp.log(p) + k * jnp.log(1 - p))
    q = jnp.exp(logpmf)
    return q / jnp.sum(q)


def lag_matrix(y, D):
    """Lagged-count design X (T x D): X[t,d] = y[t-d] for d=1..D (0 before start)."""
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[0]
    X = np.zeros((T, D), dtype=np.float64)
    for j in range(D):                      # column j is lag d = j+1
        s = j + 1
        if s < T:
            X[s:, j] = y[:T - s]
    return jnp.asarray(X)


def build_design(y, h):
    """Rebuild (A, offset, B, L, m_NB, X, q_NB) from raw counts + hyperparameters."""
    y = jnp.asarray(y, dtype=jnp.float64)
    T = y.shape[0]
    D = int(h["D_max"])
    B = baseline_design(T, h["period"], h)
    X = lag_matrix(y, D)
    q_NB = nb_kernel(D, h["mean_lag"], h["size"])
    m_NB = h["K_NB"] * q_NB
    offset = X @ m_NB
    if h["sigma_u"] > 0:
        L = warped_rbf_cholesky(D, h["beta"], h["sigma_u"], h["ell_u"])
        A = jnp.concatenate([B, X @ L], axis=1)
    else:
        L = jnp.zeros((D, 0))
        A = B
    return dict(A=A, offset=offset, B=B, L=L, m_NB=m_NB, X=X, q_NB=q_NB,
                q_b=B.shape[1], q_g=L.shape[1])


# --------------------------------------------------------------------------- #
# Likelihood + posterior
# --------------------------------------------------------------------------- #
def softplus_link(eta, link_scale, floor):
    return link_scale * jax.nn.softplus(eta / link_scale) + floor


def nb2_loglik(y, mu, kappa):
    """NB2 log-likelihood parameterised by mean mu and size kappa (mean parameterization)."""
    return jnp.sum(gammaln(y + kappa) - gammaln(kappa) - gammaln(y + 1)
                   + kappa * jnp.log(kappa / (kappa + mu))
                   + y * jnp.log(mu / (kappa + mu)))


def make_nlp(A, offset, y, kappa, link_scale, floor):
    """Return nlp(theta) for the whitened latent (prior N(0, I))."""
    A = jnp.asarray(A); offset = jnp.asarray(offset); y = jnp.asarray(y)

    def nlp(theta):
        eta = offset + A @ theta
        mu = softplus_link(eta, link_scale, floor)
        return -nb2_loglik(y, mu, kappa) + 0.5 * jnp.sum(theta ** 2)
    return nlp


def nb2_logpmf_vec(y, mu, kappa):
    """Per-observation NB2 log-pmf (mean mu, size kappa)."""
    return (gammaln(y + kappa) - gammaln(kappa) - gammaln(y + 1)
            + kappa * jnp.log(kappa / (kappa + mu)) + y * jnp.log(mu / (kappa + mu)))


def make_test_logscore(B_test, X_test, L, m_NB, y_test, kappa, link_scale, floor, q_b):
    """Return theta -> per-observation held-out log score on the test period.

    One-step-ahead intensity: lambda(t) = softplus( B_test(t) theta_b
    + X_test(t) (m_NB + L theta_g) ), scored against the NB2 test counts.
    """
    B_test = jnp.asarray(B_test); X_test = jnp.asarray(X_test)
    L = jnp.asarray(L); m_NB = jnp.asarray(m_NB); y_test = jnp.asarray(y_test)

    def test_logscore(theta):
        tb = theta[:q_b]; tg = theta[q_b:]
        f = m_NB + (L @ tg if L.shape[1] else 0.0)      # excitation kernel over lags
        eta = B_test @ tb + X_test @ f
        mu = softplus_link(eta, link_scale, floor)
        return nb2_logpmf_vec(y_test, mu, kappa)        # (n_test,)
    return test_logscore


# --------------------------------------------------------------------------- #
# Export loader (for validation and for the model modules)
# --------------------------------------------------------------------------- #
def build_fold_model(y_train, y_test, h, kappa, link_scale, floor=1e-6):
    """Build a rolling-origin fold posterior from raw counts + fixed hyperparameters.

    Returns nlp(theta) (whitened latent, N(0,I) prior), the one-step-ahead
    test log score, and the projection to (baseline, excitation).
    """
    y_tr = np.asarray(y_train, float); y_te = np.asarray(y_test, float)
    T = len(y_tr); n_test = len(y_te); D = int(h["D_max"])
    des = build_design(y_tr, h)
    nlp = make_nlp(des["A"], des["offset"], jnp.asarray(y_tr), kappa, link_scale, floor)
    tt = jnp.arange(T + 1, T + n_test + 1, dtype=jnp.float64)
    B_test = _baseline_cols(tt, h["period"], h)
    X_test = lag_matrix_future(y_tr, y_te, D)
    tls = make_test_logscore(B_test, X_test, des["L"], des["m_NB"], jnp.asarray(y_te),
                             kappa, link_scale, floor, des["q_b"])
    return dict(nlp=jax.jit(nlp), test_logscore=jax.jit(tls),
                q=des["A"].shape[1], q_b=des["q_b"], q_g=des["q_g"], n_test=n_test)


def _load_bin(path, shape):
    a = np.fromfile(path, dtype="<f8")
    return a.reshape(shape, order="F")           # column-major (Fortran order)


def load_model(name):
    """Load a model: hyperparameters, counts, and the JAX-rebuilt design.

    Returns a dict with the hyperparameters `h`, counts `y`, the rebuilt design
    (`A`, `offset`, ...), a jitted `nlp`, `q`, and (if present) the R-exported
    validation arrays under keys prefixed `R_`.
    """
    d = DATA_ROOT / name
    man = json.loads((d / "manifest.json").read_text())
    T, D, q = int(man["T"]), int(man["D_max"]), int(man["q"])
    y = _load_bin(d / "y.bin", (T,))
    h = dict(D_max=D, period=float(man["period"]), K_fourier=int(man["K_fourier"]),
             sigma_level=float(man["sigma_level"]), sigma_fourier=float(man["sigma_fourier"]),
             sigma_lin=float(man["sigma_lin"]), sigma_weekly=float(man["sigma_weekly"]),
             K_weekly=int(man["K_weekly"]), period_weekly=float(man["period_weekly"]),
             K_NB=float(man["K_NB"]), mean_lag=float(man["mean_lag"]), size=float(man["size"]),
             beta=float(man["beta"]), sigma_u=float(man["sigma_u"]), ell_u=float(man["ell_u"]))
    des = build_design(y, h)
    kappa = float(man["kappa"]); link_scale = float(man["link_scale"]); floor = float(man["floor"])
    nlp = make_nlp(des["A"], des["offset"], y, kappa, link_scale, floor)
    out = dict(name=name, man=man, h=h, y=jnp.asarray(y), kappa=kappa,
               link_scale=link_scale, floor=floor, q=q, nlp=jax.jit(nlp), **des)
    # test-period design + predictive log-score (if exported)
    n_test = int(man.get("n_test", 0))
    if n_test and (d / "y_test.bin").exists():
        y_test = _load_bin(d / "y_test.bin", (n_test,))
        B_test = _load_bin(d / "B_test.bin", (n_test, des["q_b"]))
        X_test = _load_bin(d / "X_test.bin", (n_test, D))
        out.update(n_test=n_test, y_test=jnp.asarray(y_test),
                   B_test=jnp.asarray(B_test), X_test=jnp.asarray(X_test),
                   test_logscore=jax.jit(make_test_logscore(
                       B_test, X_test, des["L"], des["m_NB"], y_test,
                       kappa, link_scale, floor, des["q_b"])))
    # attach R-exported validation arrays if present
    for key, shape in [("A", (T, q)), ("offset", (T,)), ("B", (T, int(man["q_b"]))),
                       ("L", (D, int(man["q_g"]))), ("m_NB", (D,)), ("X", (T, D)),
                       ("theta_map", (q,)), ("grad_map", (q,)), ("H_laplace", (q, q))]:
        p = d / f"{key}.bin"
        if p.exists():
            out["R_" + key] = _load_bin(p, shape)
    return out
