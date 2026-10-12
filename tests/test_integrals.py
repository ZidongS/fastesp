"""Validate the analytic Coulomb integrals against independent references."""
import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from fastesp.ints import a_table, boys, primitive_pair_integral

A_TAB = a_table(8)


def test_boys_known_values():
    # F_m(0) = 1/(2m+1)
    for m in range(6):
        assert boys(m, 1e-14) == pytest.approx(1.0 / (2 * m + 1), rel=1e-12)
    # F_0(T) = sqrt(pi)/(2 sqrt(T)) * erf(sqrt(T)) exactly
    for T in (0.1, 1.0, 5.0, 30.0):
        assert boys(0, T) == pytest.approx(
            0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T)), rel=1e-13)
    # large-T limit
    assert boys(0, 1e4) == pytest.approx(0.5 * math.sqrt(math.pi / 1e4), rel=1e-6)
    # dF_m/dT = -F_{m+1}
    T = 2.7
    h = 1e-6
    num = (boys(1, T + h) - boys(1, T - h)) / (2 * h)
    assert num == pytest.approx(-boys(2, T), rel=1e-6)


def test_ss_closed_form():
    rng = np.random.default_rng(0)
    for _ in range(8):
        A = rng.normal(0, 1, 3)
        B = A + rng.normal(0, 1, 3)
        C = rng.normal(0, 3, 3)
        al = float(np.exp(rng.uniform(-2, 2.5)))
        be = float(np.exp(rng.uniform(-2, 2.5)))
        p = al + be
        P = (al * A + be * B) / p
        K = math.exp(-al * be / p * float(((A - B) ** 2).sum()))
        R = float(np.linalg.norm(C - P))
        ref = K * (np.pi / p) ** 1.5 * math.erf(np.sqrt(p) * R) / R
        got = primitive_pair_integral(al, A, (0, 0, 0), be, B, (0, 0, 0), C, A_TAB)
        assert got == pytest.approx(ref, rel=1e-13)


def _quad(C, A, la, al, B, lb, be, R=14.0, ngl=240, ncos=220, nphi=440):
    """Independent reference: spherical quadrature about C.

    With u = C + s*n the 1/|u-C| cancels the s of the volume element, so the
    integrand is smooth and the quadrature converges spectrally.
    """
    xs, ws = leggauss(ngl)
    s = 0.5 * R * (xs + 1.0)
    wr = 0.5 * R * ws
    cz, wz = leggauss(ncos)
    phi = (np.arange(nphi) + 0.5) * (2 * np.pi / nphi)
    wphi = 2 * np.pi / nphi
    st = np.sqrt(np.clip(1 - cz ** 2, 0, None))
    nx = st[None, :, None] * np.cos(phi)[:, None, None] * np.ones((1, 1, 1))
    ny = st[None, :, None] * np.sin(phi)[:, None, None] * np.ones((1, 1, 1))
    nz = np.broadcast_to(cz[None, :, None], (nphi, ncos, 1)).copy()
    ss = s[None, None, :]
    X, Y, Z = C[0] + ss * nx, C[1] + ss * ny, C[2] + ss * nz
    rho = ((X - A[0]) ** la[0] * (Y - A[1]) ** la[1] * (Z - A[2]) ** la[2]
           * np.exp(-al * ((X - A[0]) ** 2 + (Y - A[1]) ** 2 + (Z - A[2]) ** 2))
           * (X - B[0]) ** lb[0] * (Y - B[1]) ** lb[1] * (Z - B[2]) ** lb[2]
           * np.exp(-be * ((X - B[0]) ** 2 + (Y - B[1]) ** 2 + (Z - B[2]) ** 2)))
    return float((rho * ss * wr[None, None, :] * (wz[None, :, None] * wphi)).sum())


@pytest.mark.parametrize("la,lb", [
    ((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (0, 0, 0)), ((0, 1, 0), (1, 0, 0)),
    ((1, 1, 0), (0, 0, 1)), ((2, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 2)),
    ((2, 1, 0), (1, 0, 0)), ((2, 0, 0), (2, 0, 0)), ((3, 0, 0), (1, 1, 0)),
    ((2, 1, 0), (2, 1, 0)), ((3, 2, 0), (2, 1, 0)), ((3, 3, 3), (3, 3, 3)),
])
def test_general_pair_vs_quadrature(la, lb):
    A = np.array([0.3, -0.4, 0.2])
    B = A + np.array([1.1, 0.5, -0.6])
    C = np.array([0.1, 1.6, 1.1])
    al, be = 0.45, 0.72
    ana = primitive_pair_integral(al, A, la, be, B, lb, C, A_TAB)
    ref = _quad(C, A, la, al, B, lb, be)
    assert ana == pytest.approx(ref, rel=1e-11, abs=1e-14)


def test_a_table_matches_derivative_definition():
    """a[t][j] reproduces d^t/dx^t F_0 (1-D) against a 6th-order stencil."""
    from scipy.special import hyp1f1

    W = [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]

    def F0(T):
        return hyp1f1(0.5, 1.5, -T)

    p, x = 0.83, 1.37
    T = p * x * x
    h = 3e-3

    def d(order, xx):
        if order == 0:
            return F0(p * xx * xx)
        return sum(W[k] * d(order - 1, xx + (k - 3) * h) for k in range(7)) / h

    for t in range(6):
        # d^t/dx^t F_0 = sum_j a[t,j] (2p)^(t-j) x^(t-2j) F_{t-j}(T)
        ana = sum(A_TAB[t, j] * (2 * p) ** (t - j) * x ** (t - 2 * j)
                  * boys(t - j, T) for j in range(t // 2 + 1))
        # finite differences of order 5 with h=3e-3 carry ~1e-3 relative noise
        assert ana == pytest.approx(d(t, x), rel=2e-3, abs=1e-7)
