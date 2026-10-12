"""Exact ESP from a Gaussian-basis density.

    V(C) = sum_A Z_A/|C-R_A|  -  sum_{mu nu} P_{mu nu} <mu|1/|r-C||nu>

The electron part uses the contracted Hermite / McMurchie-Davidson scheme.  For
every *primitive pair* of a shell pair the entire angular structure is contracted
once, at setup time, into a table of (alpha, m) coefficients such that

    I_pair(C) = K * sum_{alpha,m} G[alpha,m] * R^alpha * F_m(p R^2),   R = C - P

which is independent of C.  Per ESP point the work is then one Boys vector, one
monomial block and one sparse dot product per primitive pair -- a shape that maps
directly onto GPU threads.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .basis import Basis, cart_order
from .ints import boys

# M_n = int |He_n(x)| exp(-x^2/2) dx, used for the screening bound
_M_ABS = np.array([2.5066282746310002, 2.0, 1.5957691216057308, 4.0,
                   10.212920779565312, 33.63284766793385, 136.5963625280984,
                   665.1416330443616, 3768.863619586618, 25506.330595911374,
                   202331.81842358326, 1872154.2572862155])


@lru_cache(maxsize=None)
def _a_table(lmax: int) -> np.ndarray:
    a = np.zeros((lmax + 1, lmax // 2 + 2))
    a[0, 0] = 1.0
    for t in range(lmax):
        for j in range(a.shape[1]):
            v = -a[t, j]
            if j >= 1:
                v += (t - 2 * j + 2) * a[t, j - 1]
            a[t + 1, j] = v
    return a


@lru_cache(maxsize=None)
def _mono_index(lmax: int) -> tuple:
    idx = []
    for a in range(lmax + 1):
        for b in range(lmax + 1 - a):
            for c in range(lmax + 1 - a - b):
                idx.append((a, b, c))
    return tuple(idx)


@lru_cache(maxsize=None)
def _pair_type(L: int):
    """Everything that depends only on L = l1 + l2 for a shell pair.

    Returns (M0, w, mono, mvals, npow) where
      M0    : (nterm, T^3) integer/rational map etab -> (alpha,m) coefficients
      w     : (T^3,) product of |He| moments used for the bound
      mono  : (nalpha,3) monomial index of the alpha part of each term
      mvals : (nterm,) Boys order of each term
      npow  : (nterm,) power of (2p) multiplying each term
    """
    T = L + 1
    tab = _a_table(L)
    terms: dict[tuple[int, int, int, int], np.ndarray] = {}
    for ax in range(T):
        for ay in range(T):
            for az in range(T):
                lt = ax + ay + az
                if lt > L:
                    continue
                col = ax * T * T + ay * T + az
                sgn = (-1.0) ** lt
                for jx in range(ax // 2 + 1):
                    a1 = tab[ax, jx]
                    if a1 == 0:
                        continue
                    for jy in range(ay // 2 + 1):
                        a2 = tab[ay, jy]
                        if a2 == 0:
                            continue
                        for jz in range(az // 2 + 1):
                            a3 = tab[az, jz]
                            if a3 == 0:
                                continue
                            key = (ax - 2 * jx, ay - 2 * jy, az - 2 * jz,
                                   lt - (jx + jy + jz))
                            arr = terms.get(key)
                            if arr is None:
                                arr = terms[key] = np.zeros(T ** 3)
                            arr[col] += sgn * a1 * a2 * a3
    keys = sorted(terms)
    M0 = np.array([terms[k] for k in keys])            # (nterm, T^3)
    mono = np.array([k[:3] for k in keys], dtype=np.int64)
    mvals = np.array([k[3] for k in keys], dtype=np.int64)
    # the term is  c_F (2p)^m  a[t,j] R^(t-2j) F_m  with m the Boys order of the key
    npow = np.array([k[3] for k in keys], dtype=np.int64)
    # |t| and the |He| moment product, both flat-indexed over (tx,ty,tz)
    tt = np.arange(T)
    abs_t = (tt[:, None, None] + tt[None, :, None] + tt[None, None, :]).ravel()
    w = (_M_ABS[:T, None, None] * _M_ABS[None, :T, None]
         * _M_ABS[None, None, :T]).ravel()
    return M0, w, abs_t, mono, mvals, npow


def _e1d_all(i: int, j: int, p, xa, xb) -> np.ndarray:
    """E[t] for (u+xa)^i (u+xb)^j e^{-p u^2}.

    ``p``, ``xa``, ``xb`` may be scalars or arrays of shape (B,), in which case
    the result has a leading batch axis.  Batching over the primitive pairs of a
    shell pair is what makes this cheap.
    """
    tmax = i + j
    batched = np.ndim(p) > 0
    if batched:
        b = np.shape(p)[0]
        e = np.zeros((b, i + 1, j + 1, tmax + 1))
        xa = np.asarray(xa)[:, None]
        xb = np.asarray(xb)[:, None]
        ip = (1.0 / (2.0 * np.asarray(p)))[:, None]
        e[:, 0, 0, 0] = 1.0
    else:
        e = np.zeros((i + 1, j + 1, tmax + 1))
        ip = 1.0 / (2.0 * p)
        e[0, 0, 0] = 1.0
    tap = np.arange(1, tmax + 1) if tmax else np.zeros(0)
    for ii in range(i + 1):
        for jj in range(j + 1):
            if ii == 0 and jj == 0:
                continue
            src = (ii - 1, jj) if ii > 0 else (ii, jj - 1)
            x = xa if ii > 0 else xb
            if batched:
                prev = e[:, src[0], src[1]]
                cur = e[:, ii, jj]
                cur[:, 1:] += prev[:, :-1] * ip
                cur += x * prev
                cur[:, :-1] += tap * prev[:, 1:]
            else:
                prev = e[src[0], src[1]]
                cur = e[ii, jj]
                cur[1:] += prev[:-1] * ip
                cur += x * prev
                cur[:-1] += tap * prev[1:]
    return e


class PrimPair:
    __slots__ = ("p", "center", "K", "alpha", "m", "coef", "bound", "npow", "reff")

    def __init__(self, p, center, K, alpha, m, coef, bound, npow, reff):
        self.p, self.center, self.K = p, center, K
        self.alpha, self.m, self.coef = alpha, m, coef
        self.bound, self.npow, self.reff = bound, npow, reff


class ESPEngine:
    def __init__(self, basis: Basis, density: np.ndarray):
        self.basis = basis
        self.density = density
        self.prims: list[PrimPair] = []
        self.pair_atoms: list[tuple[int, int]] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        b = self.basis
        coords = b.wfn.coords
        layout = []
        off = 0
        for sh in b.wfn.shells:
            nc = len(cart_order(sh.l))
            layout.append((off, nc, sh))
            off += nc

        for si, (oi, ni, shi) in enumerate(layout):
            Ai = coords[shi.atom]
            ci = cart_order(shi.l)
            coef_i = np.array([b.coefs[oi + k] for k in range(ni)])
            for sj in range(si, len(layout)):
                oj, nj, shj = layout[sj]
                # <mu|1/r|nu> is symmetric, so both density blocks contribute
                blk = self.density[oi:oi + ni, oj:oj + nj].copy()
                if si != sj:
                    blk += self.density[oj:oj + nj, oi:oi + ni].T
                if np.abs(blk).max() < 1e-15:
                    continue
                Bj = coords[shj.atom]
                cj = cart_order(shj.l)
                coef_j = np.array([b.coefs[oj + k] for k in range(nj)])
                self._make_pair(shi, shj, blk, ci, cj, coef_i, coef_j, Ai, Bj)

    def _make_pair(self, shi, shj, blk, ci, cj, coef_i, coef_j, Ai, Bj):
        L = shi.l + shj.l
        T = L + 1
        M0, wvec, abs_t, mono, mvals, npow = _pair_type(L)
        nterm = M0.shape[0]
        ab = Ai - Bj
        ab2 = float(ab @ ab)
        # selection index for the einsum over component pairs
        kxi, kxj, kyi, kyj, kzi, kzj, wgt0 = [], [], [], [], [], [], []
        for mi, li in enumerate(ci):
            for mj, lj in enumerate(cj):
                kxi.append(li[0]); kxj.append(lj[0])
                kyi.append(li[1]); kyj.append(lj[1])
                kzi.append(li[2]); kzj.append(lj[2])
                wgt0.append(blk[mi, mj])
        kxi = np.array(kxi); kxj = np.array(kxj)
        kyi = np.array(kyi); kyj = np.array(kyj)
        kzi = np.array(kzi); kzj = np.array(kzj)
        wgt0 = np.array(wgt0)
        ncomp = len(wgt0)

        # ---- all primitive pairs of this shell pair at once ------------------
        ni_p, nj_p = len(shi.exps), len(shj.exps)
        al = np.asarray(shi.exps)[:, None] * np.ones((1, nj_p))
        be = np.asarray(shj.exps)[None, :] * np.ones((ni_p, 1))
        p = (al + be).ravel()
        ii_idx = np.repeat(np.arange(ni_p), nj_p)
        jj_idx = np.tile(np.arange(nj_p), ni_p)
        alr, ber = al.ravel(), be.ravel()
        P = (alr[:, None] * Ai[None, :]
             + ber[:, None] * Bj[None, :]) / p[:, None]           # (B,3)
        XPA = P - Ai
        XPB = P - Bj
        K = np.exp(-(alr * ber) / p * ab2)
        sel = K >= 1e-30
        if not np.any(sel):
            return
        p_s, ii_s, jj_s = p[sel], ii_idx[sel], jj_idx[sel]
        P_s, XPA_s, XPB_s, K_s = P[sel], XPA[sel], XPB[sel], K[sel]
        B = p_s.size

        ex = _e1d_all(shi.l, shj.l, p_s, XPA_s[:, 0], XPB_s[:, 0])   # (B,Tx,Ty,TzT)
        ey = _e1d_all(shi.l, shj.l, p_s, XPA_s[:, 1], XPB_s[:, 1])
        ez = _e1d_all(shi.l, shj.l, p_s, XPA_s[:, 2], XPB_s[:, 2])
        # coefficients enter as (ncomp, B); all components share the same shell
        # contraction, so index with the raw angular triples
        cw = (coef_i[kxi[:, None], ii_s[None, :]]
              * coef_j[kxj[:, None], jj_s[None, :]])                # (ncomp,B)
        wgt = (wgt0[:, None] * cw).T                                  # (B,ncomp)
        Ax = ex[:, kxi, kxj]                                          # (B,ncomp,T)
        Ay = ey[:, kyi, kyj]
        Az = ez[:, kzi, kzj]
        etab = np.einsum("qk,qka,qkb,qkc->qabc", wgt, Ax, Ay, Az)     # (B,T,T,T)
        contrib = etab.reshape(B, -1) @ M0.T                          # (B,nterm)
        cf = 2.0 * np.pi / p_s
        coefv = cf[:, None] * (2.0 * p_s)[:, None] ** npow[None, :] * contrib
        powt = (2.0 * p_s)[:, None] ** (abs_t / 2.0)            # (B,T^3)
        boundv = ((2.0 * p_s) ** -1.5
                  * (np.abs(etab).reshape(B, -1) * wvec[None, :] * powt).sum(axis=1))
        reff = 1.6 * np.sqrt((L + 2.0) / p_s)
        for b in range(B):
            self.prims.append(PrimPair(p_s[b], P_s[b], K_s[b], mono, mvals,
                                       coefv[b], boundv[b], npow, reff[b]))

    # ------------------------------------------------------------------
    def esp_electron(self, points, tol: float = 1e-15, verbose: bool = False):
        pts = np.asarray(points, float)
        n = len(pts)
        out = np.zeros(n)
        used = skipped = 0
        for pp in self.prims:
            d = np.sqrt(((pts - pp.center) ** 2).sum(axis=1))
            if pp.K * pp.bound / max(float(d.max()) - pp.reff, 0.05) < tol:
                skipped += 1
                continue
            used += 1
            R = pts - pp.center
            T = pp.p * (R * R).sum(axis=1)
            mmax = int(pp.m.max())
            F = np.stack([boys(k, T) for k in range(mmax + 1)], axis=1)
            mono = np.ones((n, len(pp.alpha)))
            for k, (a, b, c) in enumerate(map(tuple, pp.alpha)):
                if a:
                    mono[:, k] *= R[:, 0] ** a
                if b:
                    mono[:, k] *= R[:, 1] ** b
                if c:
                    mono[:, k] *= R[:, 2] ** c
            out += pp.K * (mono * F[:, pp.m] * pp.coef).sum(axis=1)
        if verbose:
            print(f"    primitive pairs: used {used}, skipped {skipped}")
        return out

    def esp(self, points, tol: float = 1e-15, verbose: bool = False):
        elec = self.esp_electron(points, tol=tol, verbose=verbose)
        w = self.basis.wfn
        pts = np.asarray(points, float)
        nuc = np.zeros(len(pts))
        zs = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "S": 16.0, "P": 15.0}
        for sym, xyz in zip(w.symbols, w.coords):
            nuc += zs[sym] / np.sqrt(((pts - xyz) ** 2).sum(axis=1))
        return nuc - elec
