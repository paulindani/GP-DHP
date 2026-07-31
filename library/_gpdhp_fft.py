"""FFT-accelerated excitation for GP-DHP.

The GP-DHP linear predictor is
    eta(t) = b(t) + sum_{d=1}^{D} y[t-d] f[d-1],   f = m_NB + L theta_g            (dense build_design)
where the second term is a *causal convolution* of the count series y with the length-D excitation
kernel f.  The dense design stores X @ L (T x D) and evaluates A @ theta at O(T D) per likelihood/
gradient call.  Here we compute the same convolution with the real FFT at O(T log T), keeping the
kernel matvec  L theta_g  (O(D^2), dense warped-RBF Cholesky) — total  O(T log T + D^2)  per eval.

The forward conv is written with jnp.fft.rfft/irfft; JAX autodiff produces the exact adjoint
(a correlation, itself an O(T log T) FFT), so jax.grad(nlp_fft) matches the dense gradient to
machine precision.  This module is a drop-in inner-objective builder; it is verified against the
dense make_nlp in scratchpad/fft_verify_bench.py.
"""
import numpy as np
import jax
import jax.numpy as jnp

from ._gpdhp_common import (softplus_link, nb2_loglik, nb2_logpmf_vec, nb_kernel,
                            baseline_design, _baseline_cols, warped_rbf_cholesky)


def _next_fast_len(n):
    """Smallest 5-smooth integer >= n (fast rfft length).  Host-side, so it is a compile constant."""
    try:
        from scipy.fft import next_fast_len
        return int(next_fast_len(int(n)))
    except Exception:
        m = int(n)
        while True:
            r = m
            for p in (2, 3, 5):
                while r % p == 0:
                    r //= p
            if r == 1:
                return m
            m += 1


def _causal_conv(y_rfft, f, Lfft, T):
    """Linear causal convolution  e[t] = sum_{d=1}^{len(f)} y[t-d] f[d-1]  via rfft.

    y_rfft = jnp.fft.rfft(y, Lfft) is precomputed once (y fixed).  f is the length-D kernel; we
    prepend a zero (lag 0 has no self-excitation) so e[t] uses only strictly past counts.
    """
    f_full = jnp.concatenate([jnp.zeros(1, f.dtype), f])          # lag 0..D
    F = jnp.fft.rfft(f_full, n=Lfft)
    return jnp.fft.irfft(y_rfft * F, n=Lfft)[:T]                  # linear conv, truncated to [0,T)


def make_nlp_fft(y, B, L, m_NB, kappa, link_scale, floor):
    """FFT analogue of _gpdhp_common.make_nlp.

    Returns nlp(theta) with theta = [theta_b (q_b) | theta_g (q_g=D)], whitened prior N(0, I).
    B is the baseline design (T x q_b); L the warped-RBF Cholesky (D x D); m_NB the parametric NB
    excitation kernel (length D).  Everything that does not depend on theta (y_rfft, the NB offset)
    is precomputed once, outside nlp, so the per-eval cost is one FFT-conv + one L @ theta_g matvec."""
    y = jnp.asarray(y, jnp.float64); B = jnp.asarray(B, jnp.float64)
    L = jnp.asarray(L, jnp.float64); m_NB = jnp.asarray(m_NB, jnp.float64)
    T = int(y.shape[0]); D = int(L.shape[0]); q_b = int(B.shape[1]); q_g = int(L.shape[1])
    Lfft = _next_fast_len(T + D + 1)
    y_rfft = jnp.fft.rfft(y, n=Lfft)                              # precomputed once (y fixed)
    offset = _causal_conv(y_rfft, m_NB, Lfft, T)                 # fixed NB excitation X @ m_NB

    def nlp(theta):
        tb, tg = theta[:q_b], theta[q_b:]
        pen = 0.5 * jnp.sum(theta ** 2)
        e_g = _causal_conv(y_rfft, L @ tg, Lfft, T)              # GP excitation X @ (L theta_g)
        eta = offset + B @ tb + e_g
        mu = softplus_link(eta, link_scale, floor)
        return -nb2_loglik(y, mu, kappa) + pen
    return nlp


def build_fold_model_fft(y_train, y_test, h, kappa, link_scale, floor=1e-6):
    """FFT analogue of _gpdhp_common.build_fold_model (no-covariate GP-DHP fold).

    Same whitened-latent train MAP objective and one-step-ahead held-out log score, but the
    excitation is a causal FFT convolution rather than a dense X / X@L design: nothing of size
    O(T D) or O(T D^2) is materialised, so the one-time final fit is O((T+n) log(T+n) + D^2) too.
    The test excitation X_test(t)(m_NB + L theta_g) is exactly the causal convolution of the
    COMBINED [train, test] counts with the length-D kernel, sliced to the test window (the same
    combined-history one-step lags as lag_matrix_future).  Verified against the dense
    build_fold_model to machine precision.  Returns the same dict shape."""
    y_tr = np.asarray(y_train, float); y_te = np.asarray(y_test, float)
    T = int(len(y_tr)); n_test = int(len(y_te)); D = int(h["D_max"])
    B = baseline_design(T, h["period"], h)                          # (T, q_b), no X / X@L
    m_NB = h["K_NB"] * nb_kernel(D, h["mean_lag"], h["size"])
    L = warped_rbf_cholesky(D, h["beta"], h["sigma_u"], h["ell_u"]) if h["sigma_u"] > 0 \
        else jnp.zeros((D, 0))
    q_b = int(B.shape[1]); q_g = int(L.shape[1])
    nlp = make_nlp_fft(y_tr, B, L, m_NB, kappa, link_scale, floor)  # train MAP (FFT)

    # one-step test score: causal conv over the COMBINED series, sliced to the test window
    Nall = T + n_test
    Lfft = _next_fast_len(Nall + D + 1)
    yall_rfft = jnp.fft.rfft(jnp.asarray(np.concatenate([y_tr, y_te])), n=Lfft)
    tt = jnp.arange(T + 1, T + n_test + 1, dtype=jnp.float64)
    B_test = _baseline_cols(tt, h["period"], h)                    # (n_test, q_b)
    yte_j = jnp.asarray(y_te)

    def test_logscore(theta):
        tb, tg = theta[:q_b], theta[q_b:]
        f = m_NB + (L @ tg if q_g else 0.0)                        # excitation kernel over lags
        e_all = _causal_conv(yall_rfft, f, Lfft, Nall)            # X_full @ f over combined series
        eta = B_test @ tb + e_all[T:T + n_test]
        mu = softplus_link(eta, link_scale, floor)
        return nb2_logpmf_vec(yte_j, mu, kappa)

    return dict(nlp=jax.jit(nlp), test_logscore=jax.jit(test_logscore),
                q=q_b + q_g, q_b=q_b, q_g=q_g, n_test=n_test)


def build_cov_model_fft(y_train, C_train, C_score, y_score, h, kappa, link_scale, col_sigmas,
                        floor=1e-6):
    """FFT analogue of the dense covariate GP-DHP fold: same whitened-latent
    train MAP objective and one-step test score with the covariate columns C scaled by col_sigmas
    entering the baseline, but the excitation X(m_NB + L theta_g) is a causal FFT convolution rather
    than a dense X / X@L design.  theta = [theta_b (q_b) | theta_cov (ncol) | theta_g (q_g)], prior N(0,I).
    Verified against build_cov_model to machine precision.  Returns the same dict shape."""
    y_tr = np.asarray(y_train, float); y_sc = np.asarray(y_score, float)
    T = int(len(y_tr)); n_sc = int(len(y_sc)); D = int(h["D_max"])
    B = baseline_design(T, h["period"], h)
    m_NB = h["K_NB"] * nb_kernel(D, h["mean_lag"], h["size"])
    L = warped_rbf_cholesky(D, h["beta"], h["sigma_u"], h["ell_u"]) if h["sigma_u"] > 0 \
        else jnp.zeros((D, 0))
    q_b = int(B.shape[1]); q_g = int(L.shape[1])
    cs = jnp.asarray(col_sigmas, float); ncol = int(cs.shape[0])
    Ctr = jnp.asarray(C_train) * cs                                # (T, ncol) scaled
    y_tr_j = jnp.asarray(y_tr)
    Lfft_tr = _next_fast_len(T + D + 1)
    ytr_rfft = jnp.fft.rfft(y_tr_j, n=Lfft_tr)

    def nlp(theta):
        tb = theta[:q_b]; tc = theta[q_b:q_b + ncol]; tg = theta[q_b + ncol:]
        f = m_NB + ((L @ tg) if q_g else 0.0)                      # excitation kernel over lags
        pen = 0.5 * jnp.sum(theta ** 2)
        eta = B @ tb + Ctr @ tc + _causal_conv(ytr_rfft, f, Lfft_tr, T)
        mu = softplus_link(eta, link_scale, floor)
        return -nb2_loglik(y_tr_j, mu, kappa) + pen

    Nall = T + n_sc                                                # test: combined-history conv, sliced
    Lfft = _next_fast_len(Nall + D + 1)
    yall_rfft = jnp.fft.rfft(jnp.asarray(np.concatenate([y_tr, y_sc])), n=Lfft)
    tt = jnp.arange(T + 1, T + n_sc + 1, dtype=jnp.float64)
    B_sc = _baseline_cols(tt, h["period"], h)
    Csc = jnp.asarray(C_score) * cs
    y_sc_j = jnp.asarray(y_sc)

    def test_logscore(theta):
        tb = theta[:q_b]; tc = theta[q_b:q_b + ncol]; tg = theta[q_b + ncol:]
        f = m_NB + ((L @ tg) if q_g else 0.0)
        e_all = _causal_conv(yall_rfft, f, Lfft, Nall)
        eta = B_sc @ tb + Csc @ tc + e_all[T:T + n_sc]
        mu = softplus_link(eta, link_scale, floor)
        return nb2_logpmf_vec(y_sc_j, mu, kappa)

    return dict(nlp=jax.jit(nlp), test_logscore=jax.jit(test_logscore),
                dim=q_b + ncol + q_g, q_b=q_b, ncol=ncol, q_g=q_g)
