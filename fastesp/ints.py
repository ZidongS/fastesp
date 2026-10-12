"""Exact Gaussian Coulomb integrals for the ESP.

The electron part of the ESP at a point C is

    V_el(C) = - sum_{mu nu} P_{mu nu} < mu | 1/|r - C| | nu > .

A primitive Cartesian pair is handled with the McMurchie-Davidson Hermite
expansion.  With ``p = a+b``, ``P`` the Gaussian product centre,

    g_A^a(r) g_B^b(r) = K sum_t E_t  L_t(r - P) ,
    L_t(r-P) = (d/dP_x)^{tx} (d/dP_y)^{ty} (d/dP_z)^{tz} exp(-p|r-P|^2),

and the potential of a Hermite Gaussian is closed form:

    int L_t(r')/|R - r'| dr' = (-1)^|t|  (d/dR)^t F(R),
    F(R) = (2 pi / p) F_0(p R^2),      R = C - P.

The derivative is expanded exactly by the one-dimensional recurrence
``a[t+1][j] = -a[t][j] + (t-2j+2) a[t][j-1]`` :

    (d/dR)^t F = c sum_j a[t,j] (2p)^(|t|-|j|) R^(t-2j) F_{|t|-|j|}(pR^2).

Because the E_t and the ``a`` table are both analytic, the whole thing can be
contracted once per primitive pair into a table that is *independent of the ESP
point*, which is what the GPU kernel consumes.
"""
from __future__ import annotations

import numpy as np
from scipy.special import hyp1f1

__all__ = ["boys", "a_table", "E_coefficients", "dF_dR", "primitive_pair_integral"]


def boys(m, T):
    """Boys function F_m(T) = int_0^1 t^{2m} exp(-T t^2) dt."""
    m = np.asarray(m)
    T = np.asarray(T, dtype=np.float64)
    out = hyp1f1(m + 0.5, m + 1.5, -T) / (2 * m + 1)
    return out if out.ndim else float(out)


_A_CACHE: dict[int, np.ndarray] = {}


def a_table(lmax: int) -> np.ndarray:
    """a[t, j]: (d/dR)^t F_0 = sum_j a[t,j] (2p)^(t-j) R^(t-2j) F_{t-j} (1-D)."""
    if lmax in _A_CACHE:
        return _A_CACHE[lmax]
    a = np.zeros((lmax + 1, lmax // 2 + 2))
    a[0, 0] = 1.0
    for t in range(lmax):
        for j in range(a.shape[1]):
            v = -a[t, j]
            if j >= 1:
                v += (t - 2 * j + 2) * a[t, j - 1]
            a[t + 1, j] = v
    _A_CACHE[lmax] = a
    return a


def _e1d(i: int, j: int, p: float, xa: float, xb: float, tmax: int) -> np.ndarray:
    """1-D Hermite expansion coefficients E[t] for (u+xa)^i (u+xb)^j e^{-p u^2}."""
    e = np.zeros((i + 1, j + 1, tmax + 1))
    e[0, 0, 0] = 1.0
    for ii in range(i + 1):
        for jj in range(j + 1):
            if ii == 0 and jj == 0:
                continue
            src = (ii - 1, jj) if ii > 0 else (ii, jj - 1)
            x = xa if ii > 0 else xb
            prev = e[src[0], src[1]]
            for t in range(tmax + 1):
                v = 0.0
                if t >= 1:
                    v += prev[t - 1] / (2.0 * p)
                v += x * prev[t]
                if t + 1 <= tmax:
                    v += (t + 1) * prev[t + 1]
                e[ii, jj, t] = v
    return e[i, j, :tmax + 1]


def E_coefficients(la, lb, p, A, B, P):
    """Full 3-D E_t table (flattened over (tx,ty,tz)) for one primitive pair."""
    xa = P[0] - A[0]
    xb = P[0] - B[0]
    ya = P[1] - A[1]
    yb = P[1] - B[1]
    za = P[2] - A[2]
    zb = P[2] - B[2]
    ex = _e1d(la[0], lb[0], p, xa, xb, la[0] + lb[0])
    ey = _e1d(la[1], lb[1], p, ya, yb, la[1] + lb[1])
    ez = _e1d(la[2], lb[2], p, za, zb, la[2] + lb[2])
    out = {}
    for tx, cx in enumerate(ex):
        if cx == 0.0:
            continue
        for ty, cy in enumerate(ey):
            if cy == 0.0:
                continue
            for tz, cz in enumerate(ez):
                if cz == 0.0:
                    continue
                out[(tx, ty, tz)] = cx * cy * cz
    return out


def dF_dR(t, R, p, a_tab=None):
    """(d/dR)^t F(R) with F = (2 pi/p) F_0(p R^2)."""
    if a_tab is None:
        a_tab = a_table(sum(t))
    R = np.asarray(R, float)
    c = 2.0 * np.pi / p
    lt = sum(t)
    tot = 0.0
    for jx in range(t[0] // 2 + 1):
        ax = a_tab[t[0], jx]
        if ax == 0.0:
            continue
        for jy in range(t[1] // 2 + 1):
            ay = a_tab[t[1], jy]
            if ay == 0.0:
                continue
            for jz in range(t[2] // 2 + 1):
                az = a_tab[t[2], jz]
                if az == 0.0:
                    continue
                jj = jx + jy + jz
                tot += (ax * ay * az * (2.0 * p) ** (lt - jj)
                        * R[0] ** (t[0] - 2 * jx)
                        * R[1] ** (t[1] - 2 * jy)
                        * R[2] ** (t[2] - 2 * jz)
                        * boys(lt - jj, p * float(R @ R)))
    return c * tot


def primitive_pair_integral(alpha, A, la, beta, B, lb, C, a_tab=None, E=None):
    """< g_A^a | 1/|r-C| | g_B^b > for *bare* (unnormalised) primitives."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    p = alpha + beta
    P = (alpha * A + beta * B) / p
    ab = A - B
    K = float(np.exp(-alpha * beta / p * (ab @ ab)))
    R = C - P
    if E is None:
        E = E_coefficients(la, lb, p, A, B, P)
    if a_tab is None:
        a_tab = a_table(sum(la) + sum(lb))
    tot = 0.0
    for t, et in E.items():
        if et == 0.0:
            continue
        lt = sum(t)
        tot += et * (-1.0) ** lt * dF_dR(t, R, p, a_tab)
    return K * tot
