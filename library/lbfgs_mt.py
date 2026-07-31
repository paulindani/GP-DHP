"""Standalone L-BFGS with Moré-Thuente line search in JAX.

Adapted from https://github.com/jjmorenog620/jax/tree/fix-lbfgs-initial-step-size
(PR discussion: https://github.com/jax-ml/jax/pull/36794)

The fork replaces the JAX zoom-style Wolfe line search with the Moré-Thuente
line search used by scipy L-BFGS-B (the dcsrch algorithm from MINPACK-2),
adds adaptive initial step sizing, numerical safeguards in the rho/gamma
updates, and falls back to a small gradient step on line search failure.

Public API:
  minimize_lbfgs_mt(fun, x0, ...)

Status codes in the returned object:
  0 = nominal / converged
  1 = max iterations reached
  2 = max function evaluations reached
  3 = max gradient evaluations reached
  4 = insufficient progress (ftol)
  5 = NaN loss or gradient — line search and fallback both failed.
      x_k is reverted to the last valid iterate on this status.

Notes on this patch:
  * Line search `ls_ok` requires a finite `f_k`, not just finite `a_k`.
  * After line search + fallback, if f_kp1 or g_kp1 is non-finite, revert
    (x_k, f_k, g_k) to the previous iterate and set status=5, failed=True.
    This prevents NaN from polluting the L-BFGS history.
  * `maxiter` may be passed as a Python int OR a JAX scalar. When passed as
    a JAX scalar, a single JIT cache entry handles any value — no recompile
    when switching between cold (500) and warm (50) budgets.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.numpy import linalg as jnp_linalg
import numpy as np


Array = jax.Array

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)


class _LineSearchResults(NamedTuple):
    failed: bool | Array
    nit: int | Array
    nfev: int | Array
    ngev: int | Array
    k: int | Array
    a_k: int | Array
    f_k: Array
    g_k: Array
    status: bool | Array


class LBFGSResults(NamedTuple):
    converged: Array
    failed: Array
    k: int | Array
    nfev: int | Array
    ngev: int | Array
    x_k: Array
    f_k: Array
    g_k: Array
    s_history: Array
    y_history: Array
    rho_history: Array
    gamma: float | Array
    status: int | Array
    ls_status: int | Array
    old_old_fval: float | Array


# ============================================================
# Moré-Thuente line search (dcsrch)
# ============================================================
class _DCStepResult(NamedTuple):
    stx: Array
    fx: Array
    dx: Array
    sty: Array
    fy: Array
    dy: Array
    stp: Array
    brackt: Array


def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """Safeguarded step computation for the Moré-Thuente line search.

    Dispatches to one of four cases via lax.switch so only the selected
    case executes.
    """
    sgnd = jnp.sign(dp) * jnp.sign(dx)

    case1 = fp > fx
    case2 = (~case1) & (sgnd < 0)
    case3 = (~case1) & (~case2) & (jnp.abs(dp) < jnp.abs(dx))
    index = jnp.where(
        case1, 0, jnp.where(case2, 1, jnp.where(case3, 2, 3))
    ).astype(jnp.int32)

    operands = (stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax)

    def _case1(ops):
        stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
        theta = 3.0 * (fx - fp) / (stp - stx + 1e-30) + dx + dp
        s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
        disc = (theta / s) ** 2 - (dx / s) * (dp / s)
        gamma = s * jnp.sqrt(jnp.maximum(disc, 0.0))
        gamma = jnp.where(stp < stx, -gamma, gamma)
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / (q + 1e-30)
        stpc = stx + r * (stp - stx)
        denom = (fx - fp) / (stp - stx + 1e-30) + dx
        stpq = stx + (dx / (denom + jnp.sign(denom) * 1e-30)) / 2.0 * (stp - stx)
        stpf = jnp.where(
            jnp.abs(stpc - stx) <= jnp.abs(stpq - stx),
            stpc, stpc + (stpq - stpc) / 2.0,
        )
        stpf = jnp.where(disc < 0, stx + 0.5 * (stp - stx), stpf)
        stpf = jnp.where(jnp.isfinite(stpf), stpf, stx + 0.5 * (stp - stx))
        return stpf, jnp.array(True)

    def _case2(ops):
        stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
        theta = 3.0 * (fx - fp) / (stp - stx + 1e-30) + dx + dp
        s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
        disc = (theta / s) ** 2 - (dx / s) * (dp / s)
        gamma = s * jnp.sqrt(jnp.maximum(disc, 0.0))
        gamma = jnp.where(stp > stx, -gamma, gamma)
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / (q + 1e-30)
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx + 1e-30)) * (stx - stp)
        stpf = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)
        stpf = jnp.where(disc < 0, (stx + stp) / 2.0, stpf)
        stpf = jnp.where(jnp.isfinite(stpf), stpf, (stx + stp) / 2.0)
        return stpf, jnp.array(True)

    def _case3(ops):
        stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
        theta = 3.0 * (fx - fp) / (stp - stx + 1e-30) + dx + dp
        s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
        gamma = s * jnp.sqrt(
            jnp.maximum(0.0, (theta / s) ** 2 - (dx / s) * (dp / s)),
        )
        gamma = jnp.where(stp > stx, -gamma, gamma)
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / (q + 1e-30)
        stpc = jnp.where(
            (r < 0) & (gamma != 0),
            stp + r * (stx - stp),
            jnp.where(stp > stx, stpmax, stpmin),
        )
        stpq = stp + (dp / (dp - dx + 1e-30)) * (stx - stp)
        stpf_b = jnp.where(jnp.abs(stpc - stp) < jnp.abs(stpq - stp), stpc, stpq)
        stpf_b = jnp.where(
            stp > stx,
            jnp.minimum(stp + 0.66 * (sty - stp), stpf_b),
            jnp.maximum(stp + 0.66 * (sty - stp), stpf_b),
        )
        stpf_nb = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)
        stpf_nb = jnp.clip(stpf_nb, stpmin, stpmax)
        stpf = jnp.where(brackt, stpf_b, stpf_nb)
        return stpf, brackt

    def _case4(ops):
        stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
        theta = 3.0 * (fp - fy) / (sty - stp + 1e-30) + dy + dp
        s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dy)), jnp.abs(dp))
        disc = (theta / s) ** 2 - (dy / s) * (dp / s)
        gamma = s * jnp.sqrt(jnp.maximum(disc, 0.0))
        gamma = jnp.where(stp > sty, -gamma, gamma)
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dy
        r = p / (q + 1e-30)
        stpc = stp + r * (sty - stp)
        stpc = jnp.where(disc < 0, stx + 0.5 * (sty - stx), stpc)
        stpf = jnp.where(brackt, stpc, jnp.where(stp > stx, stpmax, stpmin))
        return stpf, brackt

    stpf, new_brackt = lax.switch(
        index, [_case1, _case2, _case3, _case4], operands,
    )

    stpf = jnp.where(jnp.isfinite(stpf), stpf, (stx + sty) / 2.0)

    fp_gt_fx = fp > fx
    new_sty = jnp.where(fp_gt_fx, stp, jnp.where(sgnd < 0, stx, sty))
    new_fy = jnp.where(fp_gt_fx, fp, jnp.where(sgnd < 0, fx, fy))
    new_dy = jnp.where(fp_gt_fx, dp, jnp.where(sgnd < 0, dx, dy))
    new_stx = jnp.where(fp_gt_fx, stx, stp)
    new_fx = jnp.where(fp_gt_fx, fx, fp)
    new_dx = jnp.where(fp_gt_fx, dx, dp)

    return _DCStepResult(
        stx=new_stx, fx=new_fx, dx=new_dx,
        sty=new_sty, fy=new_fy, dy=new_dy,
        stp=stpf, brackt=new_brackt,
    )


class _DCSRCHState(NamedTuple):
    stp: Array
    f: Array
    g: Array
    g_full: Array
    stx: Array
    fx: Array
    gx: Array
    sty: Array
    fy: Array
    gy: Array
    brackt: Array
    stage: Array
    finit: Array
    ginit: Array
    gtest: Array
    width: Array
    width1: Array
    stmin: Array
    stmax: Array
    done: Array
    failed: Array
    nfev: Array


def _dcsrch_step_modified(state, stp, fval, gval, stpmin, stpmax):
    fm = fval - stp * state.gtest
    fxm = state.fx - state.stx * state.gtest
    fym = state.fy - state.sty * state.gtest
    gm = gval - state.gtest
    gxm = state.gx - state.gtest
    gym = state.gy - state.gtest
    r = _dcstep(
        state.stx, fxm, gxm, state.sty, fym, gym,
        stp, fm, gm, state.brackt, state.stmin, state.stmax,
    )
    return (
        r.stx, r.fx + r.stx * state.gtest, r.dx + state.gtest,
        r.sty, r.fy + r.sty * state.gtest, r.dy + state.gtest,
        r.stp, r.brackt,
    )


def _dcsrch_step_direct(state, stp, fval, gval, stpmin, stpmax):
    r = _dcstep(
        state.stx, state.fx, state.gx, state.sty, state.fy, state.gy,
        stp, fval, gval, state.brackt, state.stmin, state.stmax,
    )
    return (r.stx, r.fx, r.dx, r.sty, r.fy, r.dy, r.stp, r.brackt)


def line_search_dcsrch(f, xk, pk, old_fval=None, old_old_fval=None,
                       gfk=None, c1=1e-4, c2=0.9, maxiter=100,
                       alpha0=None):
    """Moré-Thuente line search satisfying strong Wolfe conditions."""
    xk, pk = jnp.asarray(xk), jnp.asarray(pk)
    xtol = 1e-14
    stpmin = 1e-15
    stpmax = 1e15

    def full_eval(alpha):
        val, grad = jax.value_and_grad(f)(xk + alpha * pk)
        dphi = jnp.real(_dot(grad, pk))
        return val, dphi, grad

    need_initial_eval = old_fval is None or gfk is None
    if need_initial_eval:
        phi0, dphi0, gfk = full_eval(jnp.array(0.0))
    else:
        phi0 = old_fval
        dphi0 = jnp.real(_dot(gfk, pk))

    if alpha0 is None:
        alpha0 = jnp.array(1.0)

    xtrapu = 4.0
    init_nfev = 1 if need_initial_eval else 0

    init_state = _DCSRCHState(
        stp=alpha0,
        f=phi0, g=dphi0, g_full=gfk,
        stx=jnp.array(0.0), fx=phi0, gx=dphi0,
        sty=jnp.array(0.0), fy=phi0, gy=dphi0,
        brackt=jnp.array(False),
        stage=jnp.array(1),
        finit=phi0, ginit=dphi0,
        gtest=c1 * dphi0,
        width=jnp.array(stpmax - stpmin),
        width1=jnp.array(2.0 * (stpmax - stpmin)),
        stmin=jnp.array(0.0),
        stmax=alpha0 + xtrapu * alpha0,
        done=jnp.array(False),
        failed=jnp.array(False),
        nfev=init_nfev,
    )

    def cond(state):
        return (~state.done) & (~state.failed) & (state.nfev < maxiter)

    def body(state):
        stp = state.stp
        fval, gval, gfull = full_eval(stp)
        nfev = state.nfev + 1

        ftest = state.finit + stp * state.gtest

        new_stage = jnp.where(
            (state.stage == 1) & (fval <= ftest) & (gval >= 0), 2, state.stage,
        )

        warn1 = state.brackt & ((stp <= state.stmin) | (stp >= state.stmax))
        warn2 = state.brackt & (state.stmax - state.stmin <= xtol * state.stmax)
        warn3 = (stp == stpmax) & (fval <= ftest) & (gval <= state.gtest)
        warn4 = (stp == stpmin) & ((fval > ftest) | (gval >= state.gtest))
        any_warn = warn1 | warn2 | warn3 | warn4

        converged = (fval <= ftest) & (jnp.abs(gval) <= c2 * jnp.abs(state.ginit))

        bad = ~jnp.isfinite(stp)
        done = any_warn | converged | bad
        failed = (any_warn & ~converged) | bad

        use_modified = (new_stage == 1) & (fval <= state.fx) & (fval > ftest)

        def _compute_next_step(args):
            state, stp, fval, gval, use_modified, new_stage = args
            new_stx, new_fx, new_gx, new_sty, new_fy, new_gy, new_stp, new_brackt = lax.cond(
                use_modified,
                lambda s: _dcsrch_step_modified(s, stp, fval, gval, stpmin, stpmax),
                lambda s: _dcsrch_step_direct(s, stp, fval, gval, stpmin, stpmax),
                state,
            )

            new_width = jnp.abs(new_sty - new_stx)
            need_bisect = new_brackt & (new_width >= 0.66 * state.width1)
            new_stp = jnp.where(
                need_bisect, new_stx + 0.5 * (new_sty - new_stx), new_stp,
            )
            new_width1 = jnp.where(new_brackt, state.width, state.width1)

            xtrapl = 1.1
            new_stmin = jnp.where(
                new_brackt,
                jnp.minimum(new_stx, new_sty),
                new_stp + xtrapl * (new_stp - new_stx),
            )
            new_stmax = jnp.where(
                new_brackt,
                jnp.maximum(new_stx, new_sty),
                new_stp + xtrapu * (new_stp - new_stx),
            )

            new_stp = jnp.clip(new_stp, stpmin, stpmax)

            stuck = new_brackt & (
                (new_stp <= new_stmin) | (new_stp >= new_stmax) |
                (new_stmax - new_stmin <= xtol * new_stmax)
            )
            new_stp = jnp.where(stuck, new_stx, new_stp)
            new_stp = jnp.where(jnp.isfinite(new_stp), new_stp, state.stx)

            return (new_stp, new_stx, new_fx, new_gx, new_sty, new_fy, new_gy,
                    new_brackt, new_stage, new_width, new_width1, new_stmin, new_stmax)

        def _no_step(args):
            state, stp, fval, gval, use_modified, new_stage = args
            return (state.stp, state.stx, state.fx, state.gx, state.sty, state.fy,
                    state.gy, state.brackt, state.stage, state.width, state.width1,
                    state.stmin, state.stmax)

        (new_stp, new_stx, new_fx, new_gx, new_sty, new_fy, new_gy,
         new_brackt, upd_stage, new_width, new_width1, new_stmin, new_stmax
         ) = lax.cond(
            ~done,
            _compute_next_step,
            _no_step,
            (state, stp, fval, gval, use_modified, new_stage),
        )

        return _DCSRCHState(
            stp=jnp.where(done, stp, new_stp),
            f=fval, g=gval, g_full=gfull,
            stx=new_stx, fx=new_fx, gx=new_gx,
            sty=new_sty, fy=new_fy, gy=new_gy,
            brackt=new_brackt,
            stage=upd_stage,
            finit=state.finit, ginit=state.ginit, gtest=state.gtest,
            width=new_width, width1=new_width1,
            stmin=new_stmin, stmax=new_stmax,
            done=done, failed=failed, nfev=nfev,
        )

    final = lax.while_loop(cond, body, init_state)

    return _LineSearchResults(
        failed=final.failed,
        nit=final.nfev,
        nfev=final.nfev,
        ngev=final.nfev,
        k=final.nfev,
        a_k=final.stp,
        f_k=final.f,
        g_k=final.g_full,
        status=jnp.where(final.failed, jnp.array(1), jnp.array(0)),
    )


# ============================================================
# L-BFGS with Moré-Thuente line search
# ============================================================
def _two_loop_recursion(state: LBFGSResults):
    dtype = state.rho_history.dtype
    his_size = len(state.rho_history)
    curr_size = jnp.where(state.k < his_size, state.k, his_size)
    q = -jnp.conj(state.g_k)
    a_his = jnp.zeros_like(state.rho_history)

    def body_fun1(j, carry):
        i = his_size - 1 - j
        _q, _a_his = carry
        a_i = state.rho_history[i] * _dot(jnp.conj(state.s_history[i]), _q).real.astype(dtype)
        _a_his = _a_his.at[i].set(a_i)
        _q = _q - a_i * jnp.conj(state.y_history[i])
        return _q, _a_his

    q, a_his = lax.fori_loop(0, curr_size, body_fun1, (q, a_his))
    q = state.gamma * q

    def body_fun2(j, _q):
        i = his_size - curr_size + j
        b_i = state.rho_history[i] * _dot(state.y_history[i], _q).real.astype(dtype)
        _q = _q + (a_his[i] - b_i) * state.s_history[i]
        return _q

    q = lax.fori_loop(0, curr_size, body_fun2, q)
    return q


def _update_history_vectors(history, new):
    return jnp.roll(history, -1, axis=0).at[-1, :].set(new)


def _update_history_scalars(history, new):
    return jnp.roll(history, -1, axis=0).at[-1].set(new)


def _minimize_lbfgs_mt(
    fun: Callable,
    x0: Array,
    maxiter: float | None = None,
    norm=np.inf,
    maxcor: int = 10,
    ftol: float = 2.220446049250313e-09,
    gtol: float = 1e-05,
    maxfun: float | None = None,
    maxgrad: float | None = None,
    maxls: int = 20,
    x_hook: Callable | None = None,
):
    """L-BFGS with Moré-Thuente line search.

    Args:
      fun: scalar function of a flat x.
      x0: initial guess.
      maxiter: max L-BFGS iterations.
      maxcor: history size.
      ftol: f-stopping (stops if f_k - f_{k+1} < ftol).
      gtol: gradient-norm stopping.
      maxfun: max function evaluations.
      maxgrad: max gradient evaluations.
      maxls: max line search function evaluations per iteration.
    """
    d = len(x0)
    dtype = x0.dtype

    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200
    if maxiter is None:
        maxiter = np.inf
    if maxfun is None:
        maxfun = np.inf
    if maxgrad is None:
        maxgrad = np.inf

    f_0, g_0 = jax.value_and_grad(fun)(x0)
    state_initial = LBFGSResults(
        converged=jnp.array(False, dtype=bool),
        failed=jnp.array(False, dtype=bool),
        k=0,
        nfev=1,
        ngev=1,
        x_k=x0,
        f_k=f_0,
        g_k=g_0,
        s_history=jnp.zeros((maxcor, d), dtype=dtype),
        y_history=jnp.zeros((maxcor, d), dtype=dtype),
        rho_history=jnp.zeros((maxcor,), dtype=dtype),
        gamma=1.,
        status=0,
        ls_status=0,
        old_old_fval=f_0 + jnp_linalg.norm(g_0) / 2,
    )

    def cond_fun(state: LBFGSResults):
        return (~state.converged) & (~state.failed)

    def body_fun(state: LBFGSResults):
        p_k = _two_loop_recursion(state)

        dphi0 = jnp.real(_dot(state.g_k, p_k))
        candidate = 1.01 * 2 * (state.f_k - state.old_old_fval) / (dphi0 + 1e-30)
        alpha0 = jnp.where(
            (dphi0 != 0) & (state.f_k < state.old_old_fval),
            jnp.clip(candidate, 1e-10, 1.0),
            1.0,
        )
        ls_results = line_search_dcsrch(
            f=fun,
            xk=state.x_k,
            pk=p_k,
            old_fval=state.f_k,
            gfk=state.g_k,
            maxiter=maxls,
            alpha0=alpha0,
        )

        gnorm = jnp_linalg.norm(state.g_k)
        fallback_alpha = jnp.where(gnorm > 0, 1e-4 / gnorm, 1e-4)
        # Treat NaN f_k from line search as failure (not just NaN a_k)
        ls_fk_finite = jnp.isfinite(ls_results.f_k)
        safe_alpha = jnp.where(
            ls_results.failed | ~jnp.isfinite(ls_results.a_k) | (ls_results.a_k <= 0)
            | ~ls_fk_finite,
            fallback_alpha,
            ls_results.a_k,
        )

        ls_ok = (
            ~ls_results.failed & jnp.isfinite(ls_results.a_k) & (ls_results.a_k > 0)
            & ls_fk_finite
        )
        s_k = jnp.asarray(safe_alpha).astype(p_k.dtype) * p_k
        x_kp1 = state.x_k + s_k
        f_kp1, g_kp1 = lax.cond(
            ls_ok,
            lambda _: (ls_results.f_k, ls_results.g_k),
            lambda _: jax.value_and_grad(fun)(x_kp1),
            None,
        )
        # Optional post-step hook: e.g., project x_kp1 onto a feasible set.
        # When x_hook is provided AND the hook actually modifies the iterate,
        # we re-evaluate (f, g) at the hooked point and update s_k so the
        # curvature pair (s_k, y_k) is consistent with the hooked trajectory.
        # If the hook is a no-op (iterate already in feasible region), we
        # skip the re-evaluation entirely.
        #
        # If the hook DID modify x_kp1, we also flag `proj_fired` so the
        # caller can drop the L-BFGS curvature-pair history at this step.
        # Adding a "post-projection" (s, y) pair would corrupt the Hessian
        # approximation: the displacement s_k now reflects an arbitrary
        # projection move on top of the L-BFGS-chosen step, breaking the
        # secant equation that justifies the quasi-Newton update.
        proj_fired = jnp.array(False)
        if x_hook is not None:
            x_hooked = x_hook(x_kp1)
            changed = jnp.any(x_hooked != x_kp1)
            f_kp1, g_kp1 = lax.cond(
                changed,
                lambda _: jax.value_and_grad(fun)(x_hooked),
                lambda _: (f_kp1, g_kp1),
                None,
            )
            x_kp1 = x_hooked
            s_k = x_kp1 - state.x_k
            proj_fired = changed
        # If even the fallback step yields NaN, revert to previous state and fail.
        is_nan_step = ~jnp.isfinite(f_kp1) | jnp.any(~jnp.isfinite(g_kp1))
        x_kp1 = jnp.where(is_nan_step, state.x_k, x_kp1)
        f_kp1 = jnp.where(is_nan_step, state.f_k, f_kp1)
        g_kp1 = jnp.where(is_nan_step, state.g_k, g_kp1)
        s_k = jnp.where(is_nan_step, jnp.zeros_like(s_k), s_k)
        y_k = g_kp1 - state.g_k
        rho_k_inv = jnp.real(_dot(y_k, s_k))
        rho_k_valid = jnp.isfinite(rho_k_inv) & (rho_k_inv != 0.)
        rho_k = jnp.where(rho_k_valid, jnp.reciprocal(rho_k_inv), 0.).astype(y_k.dtype)
        y_dot_y = jnp.real(_dot(jnp.conj(y_k), y_k))
        gamma = jnp.where(y_dot_y > 0, rho_k_inv / y_dot_y, state.gamma)

        status = jnp.array(0)
        status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
        status = jnp.where(state.ngev >= maxgrad, 3, status)
        status = jnp.where(state.nfev >= maxfun, 2, status)
        status = jnp.where(state.k >= maxiter, 1, status)
        status = jnp.where(is_nan_step, 5, status)  # NaN loss/grad

        converged = jnp_linalg.norm(g_kp1, ord=norm) < gtol

        state = state._replace(
            converged=converged,
            failed=(status > 0) & (~converged),
            k=state.k + 1,
            nfev=state.nfev + ls_results.nfev + jnp.where(ls_ok, 0, 1),
            ngev=state.ngev + ls_results.ngev + jnp.where(ls_ok, 0, 1),
            x_k=x_kp1.astype(state.x_k.dtype),
            f_k=f_kp1.astype(state.f_k.dtype),
            g_k=g_kp1.astype(state.g_k.dtype),
            # History update: if the projection fired, drop the curvature-pair
            # history (a "post-projection" (s, y) pair violates the secant
            # equation and corrupts the Hessian approximation).  Otherwise
            # update normally when rho_k is valid.
            s_history=lax.cond(
                proj_fired,
                lambda _: jnp.zeros_like(state.s_history),
                lambda _: lax.cond(
                    rho_k_valid,
                    lambda _: _update_history_vectors(
                        history=state.s_history, new=s_k),
                    lambda _: state.s_history,
                    None,
                ),
                None,
            ),
            y_history=lax.cond(
                proj_fired,
                lambda _: jnp.zeros_like(state.y_history),
                lambda _: lax.cond(
                    rho_k_valid,
                    lambda _: _update_history_vectors(
                        history=state.y_history, new=y_k),
                    lambda _: state.y_history,
                    None,
                ),
                None,
            ),
            rho_history=lax.cond(
                proj_fired,
                lambda _: jnp.zeros_like(state.rho_history),
                lambda _: lax.cond(
                    rho_k_valid,
                    lambda _: _update_history_scalars(
                        history=state.rho_history, new=rho_k),
                    lambda _: state.rho_history,
                    None,
                ),
                None,
            ),
            # On projection, also reset γ to the L-BFGS default scaling.
            # The cached gamma is from the pre-projection curvature pair
            # which is no longer in the history.
            gamma=jnp.where(proj_fired,
                            jnp.asarray(1.0, dtype=state.gamma.dtype),
                            gamma.astype(state.g_k.dtype)),
            status=jnp.where(converged, 0, status),
            ls_status=ls_results.status,
            old_old_fval=state.f_k,
        )

        return state

    return lax.while_loop(cond_fun, body_fun, state_initial)


# ============================================================
# Public wrapper: scipy-like interface
# ============================================================
class MinimizeResult(NamedTuple):
    x: Array
    fun: Array
    jac: Array
    nit: int | Array
    nfev: int | Array
    success: bool | Array
    status: int | Array


def minimize_lbfgs_mt(fun, x0, *, maxiter=500, maxcor=10, maxls=20,
                     ftol=2.22e-9, gtol=1e-5, x_hook=None):
    """Minimize fun(x) using L-BFGS with Moré-Thuente line search.

    Args:
      fun: scalar function of a flat x.
      x0: initial guess (1D array).
      maxiter: max L-BFGS iterations.
      maxcor: L-BFGS history size (like scipy maxcor).
      maxls: max line search function evaluations per iteration.
      ftol: function-value stopping threshold.
      gtol: gradient-norm (inf-norm) stopping threshold.
      x_hook: optional callable applied after each accepted step to the
        iterate x_{k+1}.  After hooking, (f, g) are re-evaluated at the
        hooked point and the curvature pair (s_k, y_k) is updated so the
        L-BFGS state is consistent with the hooked trajectory.  Useful
        for projection-based constraint handling.

    Returns:
      MinimizeResult with fields x, fun, jac, nit, nfev, success, status.
    """
    x0 = jnp.asarray(x0)
    final = _minimize_lbfgs_mt(
        fun=fun, x0=x0, maxiter=maxiter, maxcor=maxcor, maxls=maxls,
        ftol=ftol, gtol=gtol, x_hook=x_hook,
    )
    return MinimizeResult(
        x=final.x_k, fun=final.f_k, jac=final.g_k,
        nit=final.k, nfev=final.nfev,
        success=final.converged,
        status=final.status,
    )


# ============================================================
# Box-constrained L-BFGS-MT via smooth reparametrization
#
# Interval constraints are handled by an invertible change of variables (not
# projection, which is unreliable near active bounds): each coordinate is mapped
# from an unconstrained z so plain unconstrained L-BFGS applies and the iterate
# is feasible by construction (so e.g. log(1-p) never sees p>=1).
#   finite [lo, hi] : x = lo + (hi-lo)*sigmoid(z)
#   [lo, +inf)      : x = lo + softplus(z)
#   (-inf, hi]      : x = hi - softplus(z)
#   (-inf, +inf)    : x = z
# None in a bound means that side is unbounded (+/-inf).
# ============================================================
def _box_transforms(lower, upper):
    lo = jnp.asarray([-jnp.inf if l is None else float(l) for l in lower])
    hi = jnp.asarray([jnp.inf if u is None else float(u) for u in upper])
    lo_f, hi_f = jnp.isfinite(lo), jnp.isfinite(hi)
    both, lonly, uonly = lo_f & hi_f, lo_f & ~hi_f, ~lo_f & hi_f
    span = jnp.where(both, hi - lo, 1.0)              # guard span when not two-sided
    lo_s = jnp.where(lo_f, lo, 0.0)                   # finite sentinels for the arithmetic
    hi_s = jnp.where(hi_f, hi, 0.0)

    def to_constrained(z):
        x_both = lo_s + span * jax.nn.sigmoid(z)
        x_lonly = lo_s + jax.nn.softplus(z)
        x_uonly = hi_s - jax.nn.softplus(z)
        return jnp.where(both, x_both, jnp.where(lonly, x_lonly, jnp.where(uonly, x_uonly, z)))

    def to_raw(x):
        eps = 1e-9
        frac = jnp.clip((x - lo_s) / span, eps, 1 - eps)
        z_both = jnp.log(frac / (1 - frac))
        z_lonly = jnp.log(jnp.expm1(jnp.clip(x - lo_s, eps, None)))
        z_uonly = jnp.log(jnp.expm1(jnp.clip(hi_s - x, eps, None)))
        return jnp.where(both, z_both, jnp.where(lonly, z_lonly, jnp.where(uonly, z_uonly, x)))

    return to_constrained, to_raw


def minimize_lbfgs_mt_box(fun, x0, bounds, *, maxiter=500, maxcor=10, maxls=20,
                          ftol=2.22e-9, gtol=1e-5):
    """Box-constrained L-BFGS-MT via smooth reparametrization -- a drop-in for
    scipy.optimize.minimize(..., method='L-BFGS-B', bounds=...). `bounds` is a
    list/tuple of (lo, hi) pairs (None = unbounded on that side). `fun` is a
    scalar JAX function of a flat x (gradients via autodiff). Returns a
    MinimizeResult whose `x` is in the ORIGINAL (constrained) space."""
    to_c, to_raw = _box_transforms([b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize_lbfgs_mt(lambda z: fun(to_c(z)), to_raw(jnp.asarray(x0)),
                            maxiter=maxiter, maxcor=maxcor, maxls=maxls, ftol=ftol, gtol=gtol)
    return res._replace(x=to_c(res.x))


# ============================================================
# Full BFGS (dense inverse-Hessian) with the SAME Moré-Thuente line search.
#
# Mirrors scipy.optimize._minimize_bfgs (same algorithm + dcsrch line search,
# same status codes) but runs entirely on-device via lax.while_loop, and uses
# the O(d²) rank-2 inverse-Hessian update instead of scipy's two O(d³) dense
# matmuls (A1 @ Hk @ A2) — mathematically identical iterates, far cheaper/iter.
# Carries the dense (d, d) inverse-Hessian H_k as loop state.
# ============================================================
class BFGSResults(NamedTuple):
    converged: Array
    failed: Array
    k: int | Array
    nfev: int | Array
    ngev: int | Array
    x_k: Array
    f_k: Array
    g_k: Array
    H_k: Array            # dense inverse-Hessian approximation (d, d)
    status: int | Array
    ls_status: int | Array
    old_old_fval: float | Array


def _minimize_bfgs_mt(fun, x0, maxiter=None, norm=np.inf, ftol=2.220446049250313e-09,
                      gtol=1e-05, maxfun=None, maxgrad=None, maxls=20):
    d = len(x0); dtype = x0.dtype
    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200
    if maxiter is None: maxiter = np.inf
    if maxfun is None: maxfun = np.inf
    if maxgrad is None: maxgrad = np.inf

    f_0, g_0 = jax.value_and_grad(fun)(x0)
    state_initial = BFGSResults(
        converged=jnp.array(False, dtype=bool), failed=jnp.array(False, dtype=bool),
        k=0, nfev=1, ngev=1, x_k=x0, f_k=f_0, g_k=g_0,
        H_k=jnp.eye(d, dtype=dtype),            # H_0 = I, as in scipy
        status=0, ls_status=0,
        old_old_fval=f_0 + jnp_linalg.norm(g_0) / 2,
    )

    def cond_fun(state): return (~state.converged) & (~state.failed)

    def body_fun(state):
        p_k = -(state.H_k @ state.g_k)          # search direction (vs L-BFGS two-loop)
        dphi0 = jnp.real(_dot(state.g_k, p_k))
        candidate = 1.01 * 2 * (state.f_k - state.old_old_fval) / (dphi0 + 1e-30)
        alpha0 = jnp.where((dphi0 != 0) & (state.f_k < state.old_old_fval),
                           jnp.clip(candidate, 1e-10, 1.0), 1.0)
        ls = line_search_dcsrch(f=fun, xk=state.x_k, pk=p_k, old_fval=state.f_k,
                                gfk=state.g_k, maxiter=maxls, alpha0=alpha0)
        gnorm = jnp_linalg.norm(state.g_k)
        fallback_alpha = jnp.where(gnorm > 0, 1e-4 / gnorm, 1e-4)
        ls_fk_finite = jnp.isfinite(ls.f_k)
        safe_alpha = jnp.where(
            ls.failed | ~jnp.isfinite(ls.a_k) | (ls.a_k <= 0) | ~ls_fk_finite,
            fallback_alpha, ls.a_k)
        ls_ok = ~ls.failed & jnp.isfinite(ls.a_k) & (ls.a_k > 0) & ls_fk_finite
        s_k = jnp.asarray(safe_alpha).astype(p_k.dtype) * p_k
        x_kp1 = state.x_k + s_k
        f_kp1, g_kp1 = lax.cond(ls_ok, lambda _: (ls.f_k, ls.g_k),
                                lambda _: jax.value_and_grad(fun)(x_kp1), None)
        is_nan_step = ~jnp.isfinite(f_kp1) | jnp.any(~jnp.isfinite(g_kp1))
        x_kp1 = jnp.where(is_nan_step, state.x_k, x_kp1)
        f_kp1 = jnp.where(is_nan_step, state.f_k, f_kp1)
        g_kp1 = jnp.where(is_nan_step, state.g_k, g_kp1)
        s_k = jnp.where(is_nan_step, jnp.zeros_like(s_k), s_k)
        y_k = g_kp1 - state.g_k
        sy = jnp.real(_dot(y_k, s_k))
        # BFGS inverse-Hessian update, O(d²) rank-2 form (== scipy's A1·Hk·A2):
        #   H+ = H - ρ(s (Hy)ᵀ + (Hy) sᵀ) + ρ(ρ·yᵀHy + 1) s sᵀ,   ρ = 1/(yᵀs)
        # Only when the curvature condition holds (line search ⇒ yᵀs > 0).
        do_update = ls_ok & jnp.isfinite(sy) & (sy > 1e-18)
        rho = jnp.where(do_update, jnp.reciprocal(sy), 0.).astype(dtype)
        # Initial Hessian scaling (Nocedal & Wright eq. 6.20): on the first
        # accepted step rescale H_0 = I → (sᵀy)/(yᵀy)·I before the BFGS update,
        # so the first quasi-Newton step is well-scaled.  Critical on stiff /
        # ill-conditioned problems — this is the analogue of L-BFGS's γ.
        yy = jnp.real(_dot(y_k, y_k))
        scale = jnp.where((state.k == 0) & do_update & (yy > 0),
                          sy / yy, 1.0).astype(dtype)
        H_base = state.H_k * scale
        Hy = H_base @ y_k
        yHy = jnp.real(_dot(y_k, Hy))
        H_new = (H_base
                 - rho * (jnp.outer(s_k, Hy) + jnp.outer(Hy, s_k))
                 + (rho * (rho * yHy + 1.0)) * jnp.outer(s_k, s_k))
        H_kp1 = jnp.where(do_update, H_new, state.H_k)

        status = jnp.array(0)
        status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
        status = jnp.where(state.ngev >= maxgrad, 3, status)
        status = jnp.where(state.nfev >= maxfun, 2, status)
        status = jnp.where(state.k >= maxiter, 1, status)
        status = jnp.where(is_nan_step, 5, status)
        converged = jnp_linalg.norm(g_kp1, ord=norm) < gtol

        return state._replace(
            converged=converged, failed=(status > 0) & (~converged),
            k=state.k + 1,
            nfev=state.nfev + ls.nfev + jnp.where(ls_ok, 0, 1),
            ngev=state.ngev + ls.ngev + jnp.where(ls_ok, 0, 1),
            x_k=x_kp1.astype(state.x_k.dtype), f_k=f_kp1.astype(state.f_k.dtype),
            g_k=g_kp1.astype(state.g_k.dtype), H_k=H_kp1,
            status=jnp.where(converged, 0, status), ls_status=ls.status,
            old_old_fval=state.f_k)

    return lax.while_loop(cond_fun, body_fun, state_initial)


def minimize_bfgs_mt(fun, x0, *, maxiter=500, maxls=20, ftol=2.22e-9, gtol=1e-5):
    """Full BFGS (dense inverse-Hessian) with Moré-Thuente line search — same
    algorithm as scipy's BFGS, run entirely on-device.  Returns a MinimizeResult.
    Memory: O(d²) for the inverse Hessian; per-iter cost O(d²)."""
    x0 = jnp.asarray(x0)
    final = _minimize_bfgs_mt(fun=fun, x0=x0, maxiter=maxiter, maxls=maxls,
                              ftol=ftol, gtol=gtol)
    return MinimizeResult(x=final.x_k, fun=final.f_k, jac=final.g_k, nit=final.k,
                          nfev=final.nfev, success=final.converged, status=final.status)
