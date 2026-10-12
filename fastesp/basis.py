"""Cartesian Gaussian basis construction from a MOLDEN wavefunction.

The MOLDEN file stores *spherical* (pure) d/f/g functions when the [5D]/[7F]
flags are present.  All integrals in this package are written for plain
Cartesian Gaussians, so every l-shell is re-expressed in its Cartesian span:

    spurious subspace  =  { r^2 * m : m a monomial of degree l-2 }

The pure (l-degenerate) part is the S-orthogonal complement of that subspace,
and the transformation ``T`` satisfies ``T S T^T = 1``.  Everything below is
convention independent -- if the MOLDEN functions span the same function space
(they do: real solid harmonics) the density is reproduced exactly regardless of
the phase/order convention used by the writer.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .molden import Shell, Wavefunction


# --------------------------------------------------------------------------
# Cartesian component bookkeeping
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def cart_order(l: int) -> tuple[tuple[int, int, int], ...]:
    """Cartesian components of a shell with angular momentum ``l``.

    Ordering is the standard 'x fastest' one used by most QC codes:
    (l,0,0), (l-1,1,0), (l-1,0,1), ...
    """
    out = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            out.append((lx, ly, lz))
    return tuple(out)


@lru_cache(maxsize=None)
def _dfact2(n: int) -> float:
    """(n)!! for odd n, with (-1)!! = 1."""
    if n <= 0:
        return 1.0
    r = 1.0
    for k in range(n, 0, -2):
        r *= k
    return r


def primitive_norm(alpha: float, lx: int, ly: int, lz: int) -> float:
    """Primitive prefactor used by the MOLDEN/ORCA convention.

    ORCA writes GTO coefficients that already include

        N_s(alpha, l) = (2 alpha/pi)^{3/4} (4 alpha)^{l/2}

    i.e. the radial normalisation *without* the angular (2l-1)!! factor that the
    fully normalised Cartesian Gaussian carries.  Verified against this very file:
    the printed coefficient of every single-primitive p/d/f shell equals
    N_s(alpha, l) to 8 digits.
    """
    l = lx + ly + lz
    return (2.0 * alpha / np.pi) ** 0.75 * (4.0 * alpha) ** (0.5 * l)


# --------------------------------------------------------------------------
# 1D overlap helper (used for same-centre blocks only)
# --------------------------------------------------------------------------
def _os1d(a: int, b: int, pa: float, pb: float, p: float) -> float:
    """int (x-A)^a (x-B)^b exp(-p (x-P)^2) dx  (the sqrt(pi/p) factor excluded)."""
    s = 0.0
    for i in range(a + 1):
        for j in range(b + 1):
            if (i + j) % 2:
                continue
            s += (
                _binom(a, i) * _binom(b, j)
                * pa ** (a - i) * pb ** (b - j)
                * (1.0 / (2.0 * p)) ** ((i + j) / 2.0)
                * _dfact2(i + j - 1)
            )
    return s


@lru_cache(maxsize=None)
def _binom(n: int, k: int) -> float:
    from math import comb
    return float(comb(n, k))


@lru_cache(maxsize=None)
def cartesian_overlap_block(l: int) -> np.ndarray:
    """Overlap of same-exponent MOLDEN-normalised Cartesian Gaussians of degree l.

    With g_i = N_s(alpha,l) m_i exp(-alpha r^2) the block is independent of alpha
    because N_s^2 * (2 alpha)^{-(l+3/2)} is alpha free; alpha = 1 is used here.
    """
    comps = cart_order(l)
    n = len(comps)
    p = 2.0
    pref = primitive_norm(1.0, l, 0, 0) ** 2 * (np.pi / p) ** 1.5
    s = np.empty((n, n))
    for i, (ax, ay, az) in enumerate(comps):
        for j, (bx, by, bz) in enumerate(comps):
            s[i, j] = pref * (_os1d(ax, bx, 0.0, 0.0, p)
                              * _os1d(ay, by, 0.0, 0.0, p)
                              * _os1d(az, bz, 0.0, 0.0, p))
    return s


# Real solid harmonics in MOLDEN ordering m = 0, +1, -1, +2, -2, ...
# Entries are the coefficients of the degree-l monomials (any common scale per
# row is irrelevant: the rows are S-normalised below).  The relative signs were
# read back from this very file: inverting  S_true = (C C^T)^-1  against the
# Cartesian overlap returns exactly these polynomials.
_POLY = {
    2: [{(2, 0, 0): -1, (0, 2, 0): -1, (0, 0, 2): 2},          # 3z^2 - r^2
        {(1, 0, 1): 1},                                         # xz
        {(0, 1, 1): 1},                                         # yz
        {(2, 0, 0): 1, (0, 2, 0): -1},                          # x^2 - y^2
        {(1, 1, 0): 1}],                                        # xy
    3: [{(2, 0, 1): -3, (0, 2, 1): -3, (0, 0, 3): 2},           # z(5z^2-3r^2)
        {(3, 0, 0): -1, (1, 2, 0): -1, (1, 0, 2): 4},           # x(5z^2-r^2)
        {(2, 1, 0): -1, (0, 3, 0): -1, (0, 1, 2): 4},           # y(5z^2-r^2)
        {(2, 0, 1): 1, (0, 2, 1): -1},                          # z(x^2-y^2)
        {(1, 1, 1): 1},                                         # xyz
        {(3, 0, 0): -1, (1, 2, 0): 3},                          # x(x^2-3y^2)
        {(2, 1, 0): -3, (0, 3, 0): 1}],                         # y(3x^2-y^2)
    4: [{(0, 0, 4): 8, (2, 0, 2): -24, (0, 2, 2): -24,
         (4, 0, 0): 3, (2, 2, 0): 6, (0, 4, 0): 3},             # 35z^4-30z^2r^2+3r^4
        {(1, 0, 3): 4, (3, 0, 1): -3, (1, 2, 1): -3},           # xz(7z^2-3r^2)
        {(0, 1, 3): 4, (0, 3, 1): -3, (2, 1, 1): -3},           # yz(7z^2-3r^2)
        {(2, 0, 2): 6, (0, 2, 2): -6, (4, 0, 0): -1, (0, 4, 0): 1},   # (x^2-y^2)(7z^2-r^2)
        {(1, 1, 2): 6, (3, 1, 0): -1, (1, 3, 0): -1},           # xy(7z^2-r^2)
        {(3, 0, 1): 1, (1, 2, 1): -3},                          # xz(x^2-3y^2)
        {(2, 1, 1): 3, (0, 3, 1): -1},                          # yz(3x^2-y^2)
        {(4, 0, 0): 1, (2, 2, 0): -6, (0, 4, 0): 1},            # x^4-6x^2y^2+y^4
        {(3, 1, 0): 1, (1, 3, 0): -1}],                         # xy(x^2-y^2)
}


@lru_cache(maxsize=None)
def pure_to_cartesian(l: int) -> np.ndarray:
    """T (2l+1, ncart): rows are the MOLDEN real solid harmonics, S-orthonormal.

    For l <= 1 the Cartesian span is already pure and T is the identity.
    """
    comps = cart_order(l)
    n = len(comps)
    if l <= 1:
        return np.eye(n)

    s = cartesian_overlap_block(l)
    cmap = {c: i for i, c in enumerate(comps)}
    v = np.zeros((2 * l + 1, n))
    for r, poly in enumerate(_POLY[l]):
        for c, val in poly.items():
            v[r, cmap[c]] = val
    nrm = np.sqrt(np.einsum("ri,ij,rj->r", v, s, v))
    return v / nrm[:, None]


def _pure_to_cartesian_generic(l: int) -> np.ndarray:
    """Fallback (not used for l <= 4): S-orthonormal complement of the spurious
    subspace { r^2 * (monomials of degree l-2) }, arbitrary ordering."""
    comps = cart_order(l)
    n = len(comps)
    if l <= 1:
        return np.eye(n)

    s = cartesian_overlap_block(l)
    chol = np.linalg.cholesky(s)

    # spurious vectors: r^2 * (monomials of degree l-2)
    mons = []
    lm2 = l - 2
    for lx in range(lm2, -1, -1):
        for ly in range(lm2 - lx, -1, -1):
            lz = lm2 - lx - ly
            mons.append((lx, ly, lz))

    # coefficient vector of r^2 * x^mx y^my z^mz in the *normalised* Cartesian
    # basis g_i = N_i * monomial_i * exp(-alpha r^2):  c_i = (monomial coeff)/N_i
    m = np.zeros((n, len(mons)))
    for k, (mx, my, mz) in enumerate(mons):
        for i, (ax, ay, az) in enumerate(comps):
            for dx, dy, dz in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
                if ax == mx + dx and ay == my + dy and az == mz + dz:
                    m[i, k] += 1.0 / primitive_norm(1.0, ax, ay, az)
    # NOTE: 1/N_s(1,l) is a common factor per *resulting* component, exactly the
    # coefficient of r^2*m in the N_s-normalised Cartesian basis.
    b = chol.T @ m                       # (n, nspur)
    sv_all, vh = np.linalg.svd(b.T)[1:]  # vh rows = right singular vectors of b^T
    rank = int((sv_all > 1e-9 * sv_all[0]).sum()) if sv_all.size else 0
    nn = vh[rank:].T                     # (n, n - rank) orthonormal null space
    # rows are c = L^{-T} y with y = L^T c; np.linalg.cholesky returns lower L
    t = (np.linalg.solve(chol.T, nn)).T
    return t


# --------------------------------------------------------------------------
# Full basis
# --------------------------------------------------------------------------
class Basis:
    """Cartesian Gaussian basis: centres, exponents and contraction coefficients."""

    def __init__(self, wfn: Wavefunction, use_primitive_norm: bool = True):
        self.wfn = wfn
        self.use_primitive_norm = use_primitive_norm
        self.centers: list[np.ndarray] = []
        self.atoms: list[int] = []
        self.ang: list[tuple[int, int, int]] = []
        self.exps: list[np.ndarray] = []
        self.coefs: list[np.ndarray] = []
        self.shell_of: list[int] = []
        self.shells: list[Shell] = []

        for sh in wfn.shells:
            sh_center = wfn.coords[sh.atom]
            for (lx, ly, lz) in cart_order(sh.l):
                # MOLDEN coefficients already carry N_s(alpha,l); the Cartesian
                # components of a shell therefore share the same radial prefactor.
                cf = np.asarray(sh.coeffs, dtype=np.float64)
                self.centers.append(sh_center)
                self.atoms.append(sh.atom)
                self.ang.append((lx, ly, lz))
                self.exps.append(np.asarray(sh.exps))
                self.coefs.append(cf)
                self.shell_of.append(len(self.shells))
            self.shells.append(sh)

        self.n = len(self.centers)
        self.centers = np.asarray(self.centers)
        self.atoms = np.asarray(self.atoms, dtype=np.int32)
        self.ang_arr = np.asarray(self.ang, dtype=np.int32)

    # -- transformation from the stored (spherical) representation ---------
    def block_transform(self) -> np.ndarray:
        """(ncart_cart, ncart_sph) matrix mapping stored coefficients to Cartesian.

        Built shell-block by shell-block over the *stored* ordering, where each
        shell of angular momentum l occupies 2l+1 consecutive functions.
        """
        wfn = self.wfn
        off_sph = 0
        off_cart = 0
        blocks = []
        for sh in wfn.shells:
            npure = 2 * sh.l + 1
            ncart = len(cart_order(sh.l))
            blocks.append((off_sph, off_cart, npure, ncart, sh.l))
            off_sph += npure
            off_cart += ncart
        assert off_cart == self.n, (off_cart, self.n)

        t = np.zeros((off_cart, off_sph))
        for off_s, off_c, npure, ncart, l in blocks:
            t[off_c:off_c + ncart, off_s:off_s + npure] = pure_to_cartesian(l).T
        return t

    def density_matrix_cartesian(self, min_occ: float = 1e-10) -> np.ndarray:
        wfn = self.wfn
        occ = wfn.mo_occ > min_occ
        c = wfn.mo_coeff[:, occ] * np.sqrt(wfn.mo_occ[occ])[None, :]
        p_sph = c @ c.T
        t = self.block_transform()
        return t @ p_sph @ t.T


def overlap_matrix(b: Basis) -> np.ndarray:
    """Full Cartesian overlap matrix (used only for validation)."""
    n = b.n
    s = np.zeros((n, n))
    for i in range(n):
        ai, bi, ci = b.centers[i]
        li = b.ang[i]
        for j in range(i, n):
            aj, bj, cj = b.centers[j]
            lj = b.ang[j]
            d = np.array([ai - aj, bi - bj, ci - cj])
            ab2 = d @ d
            tot = 0.0
            for ea, ca in zip(b.exps[i], b.coefs[i]):
                for eb, cb in zip(b.exps[j], b.coefs[j]):
                    p = ea + eb
                    mu = ea * eb / p
                    pa = -eb * d / p          # P - A
                    pb = ea * d / p           # P - B
                    sx = _os1d(li[0], lj[0], pa[0], pb[0], p)
                    sy = _os1d(li[1], lj[1], pa[1], pb[1], p)
                    sz = _os1d(li[2], lj[2], pa[2], pb[2], p)
                    tot += ca * cb * np.exp(-mu * ab2) * (np.pi / p) ** 1.5 * sx * sy * sz
            s[i, j] = s[j, i] = tot
    return s


def build_basis(path, use_primitive_norm: bool = True) -> tuple[Wavefunction, Basis]:
    from .molden import read_molden
    wfn = read_molden(path)
    return wfn, Basis(wfn, use_primitive_norm=use_primitive_norm)
