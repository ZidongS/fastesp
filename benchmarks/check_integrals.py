"""Independent validation of the analytic Gaussian Coulomb integrals.

1. s-s pair against the closed form  K (pi/p)^{3/2} erf(sqrt(p) R)/R
2. E-coefficient Hermite expansion against the literal polynomial product
3. general (a,b) pairs against spherical quadrature about C (no singularity:
   u = C + s*n makes 1/|u-C| cancel the s of the volume element)
"""
import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastesp.ints import (boys, a_table, E_coefficients, dF_dR,  # noqa: E402
                          primitive_pair_integral)


def check_ss(n=8, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        A = rng.normal(0, 1, 3); B = A + rng.normal(0, 1, 3); C = rng.normal(0, 3, 3)
        al = float(np.exp(rng.uniform(-2, 2.5))); be = float(np.exp(rng.uniform(-2, 2.5)))
        p = al + be; P = (al * A + be * B) / p
        K = math.exp(-al * be / p * float(((A - B) ** 2).sum()))
        R = float(np.linalg.norm(C - P))
        ref = K * (np.pi / p) ** 1.5 * math.erf(np.sqrt(p) * R) / R
        got = primitive_pair_integral(al, A, (0, 0, 0), be, B, (0, 0, 0), C)
        worst = max(worst, abs(got - ref) / abs(ref))
    return worst


def check_hermite(seed=3, n=20):
    """sum_t E_t L_t(u) must equal the literal product polynomial."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        A = rng.normal(0, 1, 3); B = A + rng.normal(0, 1, 3)
        al = float(np.exp(rng.uniform(-1, 1.5))); be = float(np.exp(rng.uniform(-1, 1.5)))
        la = tuple(int(x) for x in rng.integers(0, 4, 3))
        lb = tuple(int(x) for x in rng.integers(0, 4, 3))
        p = al + be; P = (al * A + be * B) / p
        E = E_coefficients(la, lb, p, A, B, P)
        u = rng.normal(0, 0.4, 3)
        # L_t(u) = (-1)^|t| d^t/du^t exp(-p|u|^2); build via Hermite recursion
        # d/du exp(-p u^2) = -2 p u exp(-p u^2)
        import itertools
        val = 0.0
        for t, et in E.items():
            val += et * _hermite_val(t, u, p)
        lhs = 1.0
        for k in range(3):
            lhs *= (u[k] + P[k] - A[k]) ** la[k] * (u[k] + P[k] - B[k]) ** lb[k]
        lhs *= math.exp(-p * float(u @ u))
        denom = max(abs(lhs), abs(val), 1e-300)
        worst = max(worst, abs(lhs - val) / denom)
    return worst


def _hermite_val(t, u, p):
    """(-1)^|t| (d/du)^t exp(-p|u|^2) evaluated at u, by 1-D Hermite polynomials."""
    # d^n/du^n exp(-p u^2) = (-2p)^n He_n(sqrt(2p) u) exp(-p u^2) / 2^(n/2) ... use
    # the explicit Hermite polynomial via sympy-free recurrence on the derivative.
    from numpy.polynomial.hermite_e import hermeval
    val = math.exp(-p * float(u @ u))
    for k in range(3):
        n = t[k]
        if n == 0:
            continue
        c = [0.0] * (n + 1); c[n] = 1.0
        # d^n/du^n e^{-p u^2} = (-1)^n (2p)^{n/2} He_n(sqrt(2p) u) e^{-p u^2}
        val *= (-1) ** n * (2 * p) ** (n / 2.0) * hermeval(math.sqrt(2 * p) * u[k], c)
    return (-1.0) ** sum(t) * val


def _int_sphere(C, A, la, al, B, lb, be, R=14.0, ngl=240, ncos=220, nphi=440):
    xs, ws = leggauss(ngl)
    s = 0.5 * R * (xs + 1.0); wr = 0.5 * R * ws
    cz, wz = leggauss(ncos)
    phi = (np.arange(nphi) + 0.5) * (2 * np.pi / nphi); wphi = 2 * np.pi / nphi
    st = np.sqrt(np.clip(1 - cz ** 2, 0, None))
    nx = st[None, :, None] * np.cos(phi)[:, None, None] * np.ones((1, 1, 1))
    ny = st[None, :, None] * np.sin(phi)[:, None, None] * np.ones((1, 1, 1))
    nz = np.broadcast_to(cz[None, :, None], (nphi, ncos, 1)).copy()
    ss = s[None, None, :]
    X = C[0] + ss * nx; Y = C[1] + ss * ny; Z = C[2] + ss * nz
    rho = ((X - A[0]) ** la[0] * (Y - A[1]) ** la[1] * (Z - A[2]) ** la[2]
           * np.exp(-al * ((X - A[0]) ** 2 + (Y - A[1]) ** 2 + (Z - A[2]) ** 2))
           * (X - B[0]) ** lb[0] * (Y - B[1]) ** lb[1] * (Z - B[2]) ** lb[2]
           * np.exp(-be * ((X - B[0]) ** 2 + (Y - B[1]) ** 2 + (Z - B[2]) ** 2)))
    return float((rho * ss * wr[None, None, :] * (wz[None, :, None] * wphi)).sum())


CASES = [((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (0, 0, 0)), ((0, 1, 0), (1, 0, 0)),
         ((1, 1, 0), (0, 0, 1)), ((2, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 2)),
         ((2, 1, 0), (1, 0, 0)), ((2, 0, 0), (2, 0, 0)), ((3, 0, 0), (1, 1, 0)),
         ((2, 1, 0), (2, 1, 0)), ((3, 2, 0), (2, 1, 0)), ((3, 3, 3), (3, 3, 3))]


def check_grid(cases=CASES, R=14.0):
    A = np.array([0.3, -0.4, 0.2]); B = A + np.array([1.1, 0.5, -0.6]); C = np.array([0.1, 1.6, 1.1])
    al, be = 0.45, 0.72
    worst = 0.0
    for la, lb in cases:
        val = _int_sphere(C, A, la, al, B, lb, be, R)
        ana = primitive_pair_integral(al, A, la, be, B, lb, C)
        rel = abs(ana - val) / max(abs(val), 1e-30)
        worst = max(worst, rel)
        print(f"  {la} {lb}  analytic {ana:+.10e}   quadrature {val:+.10e}   rel {rel:.2e}")
    return worst


if __name__ == "__main__":
    print("check 1: s-s vs closed form                 worst rel = %.2e" % check_ss())
    print("check 2: Hermite E expansion vs polynomial  worst rel = %.2e" % check_hermite())
    print("check 3: general pairs vs quadrature")
    print("                                            worst rel = %.2e" % check_grid())
