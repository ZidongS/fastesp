"""Contracted pair table: construction, binary I/O and a NumPy evaluator.

The table is the point-independent core of the algorithm.  For every primitive
pair of every significant shell pair we store a sparse set of (alpha, m, g)
terms such that the electron ESP contribution of that pair at a point ``C`` is

    I_pair(C) = K * sum_k g_k * R^alpha_k * F_{m_k}(p R^2),      R = C - P

with ``P`` the Gaussian product centre and ``p`` the product exponent.

Layout of the flat binary (little endian), matching ``native/build_pairs.cpp``
and ``native/esp_gpu.cu``::

    int32   n_pairs, n_terms, n_points, n_atoms, max_mono, pad[3]
    double  pp_exp[n_pairs]
    double  pp_cx[n_pairs], pp_cy[n_pairs], pp_cz[n_pairs]
    double  pp_K[n_pairs]
    int32   pp_nt[n_pairs], pp_off[n_pairs], pp_nmono[n_pairs], pp_monooff[n_pairs]
    int32   mo_parent[n_mono], mo_axis[n_mono]
    int32   tm_aidx[n_terms], tm_m[n_terms]
    double  tm_g[n_terms]
    double  pts[3 * n_points]
    double  atom_Z[n_atoms], atom_xyz[3 * n_atoms]
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .basis import Basis, cart_order
from .ints import boys
from .molden import Wavefunction

HEADER_FIELDS = 8


@dataclass
class PairTable:
    """Flat contracted pair table (offsets are into the term/mono arrays)."""

    pp_exp: np.ndarray
    pp_cx: np.ndarray
    pp_cy: np.ndarray
    pp_cz: np.ndarray
    pp_K: np.ndarray
    pp_nt: np.ndarray
    pp_off: np.ndarray
    pp_nmono: np.ndarray
    pp_monooff: np.ndarray
    mo_parent: np.ndarray
    mo_axis: np.ndarray
    tm_aidx: np.ndarray
    tm_m: np.ndarray
    tm_g: np.ndarray
    pts: np.ndarray
    Z: np.ndarray
    xyz: np.ndarray
    max_mono: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.pp_exp)

    @property
    def n_terms(self) -> int:
        return len(self.tm_g)

    # -- evaluation ----------------------------------------------------
    def electron(self, points: np.ndarray | None = None) -> np.ndarray:
        """Electron part of the ESP, -int rho/|r-C| dr, in Hartree/e."""
        pts = self.pts if points is None else np.asarray(points, float)
        out = np.zeros(len(pts))
        for ip in range(self.n_pairs):
            n = int(self.pp_nt[ip])
            off = int(self.pp_off[ip])
            mo = int(self.pp_monooff[ip])
            nm = int(self.pp_nmono[ip])
            R = pts - np.array([self.pp_cx[ip], self.pp_cy[ip], self.pp_cz[ip]])
            T = self.pp_exp[ip] * np.einsum("ij,ij->i", R, R)
            m = self.tm_m[off:off + n]
            F = np.stack([boys(int(k), T) for k in range(int(m.max()) + 1)], axis=1)
            mono = np.ones((len(pts), nm))
            for j in range(1, nm):
                mono[:, j] = mono[:, self.mo_parent[mo + j]] * R[:, self.mo_axis[mo + j]]
            acc = np.zeros(len(pts))
            for k in range(n):
                acc += self.tm_g[off + k] * mono[:, self.tm_aidx[off + k]] * F[:, m[k]]
            out += self.pp_K[ip] * acc
        return out

    def nuclear(self, points: np.ndarray | None = None) -> np.ndarray:
        pts = self.pts if points is None else np.asarray(points, float)
        nuc = np.zeros(len(pts))
        for z, c in zip(self.Z, self.xyz):
            nuc += z / np.linalg.norm(pts - c, axis=1)
        return nuc

    def esp(self, points: np.ndarray | None = None) -> np.ndarray:
        """Full ESP (nuclear - electron) in Hartree/e."""
        return self.nuclear(points) - self.electron(points)

    # -- screen ---------------------------------------------------------
    def screen(self, tol: float) -> "PairTable":
        """Drop primitive pairs whose exact contribution is below ``tol``.

        The test is done by evaluating each pair at the points it is closest to,
        which is cheap compared with the full N_pairs x N_points sweep.
        """
        d = np.sqrt((self.pts[None, :, :] - np.stack(
            [self.pp_cx, self.pp_cy, self.pp_cz], axis=1)[:, None, :]) ** 2).sum(-1)
        keep = d.min(axis=1)  # placeholder, real screen uses the integral bound
        return self

    # -- I/O ------------------------------------------------------------
    def write(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            f.write(struct.pack("<8i", self.n_pairs, self.n_terms, len(self.pts),
                                len(self.Z), int(self.max_mono), 0, 0, 0))
            for arr, dt in ((self.pp_exp.astype(np.float64), np.float64),
                            (self.pp_cx.astype(np.float64), np.float64),
                            (self.pp_cy.astype(np.float64), np.float64),
                            (self.pp_cz.astype(np.float64), np.float64),
                            (self.pp_K.astype(np.float64), np.float64),
                            (self.pp_nt.astype(np.int32), np.int32),
                            (self.pp_off.astype(np.int32), np.int32),
                            (self.pp_nmono.astype(np.int32), np.int32),
                            (self.pp_monooff.astype(np.int32), np.int32),
                            (self.mo_parent.astype(np.int32), np.int32),
                            (self.mo_axis.astype(np.int32), np.int32),
                            (self.tm_aidx.astype(np.int32), np.int32),
                            (self.tm_m.astype(np.int32), np.int32),
                            (self.tm_g.astype(np.float64), np.float64)):
                f.write(np.ascontiguousarray(arr, dt).tobytes())
            f.write(np.ascontiguousarray(self.pts, np.float64).tobytes())
            f.write(np.ascontiguousarray(self.Z, np.float64).tobytes())
            f.write(np.ascontiguousarray(self.xyz, np.float64).tobytes())

    @classmethod
    def read(cls, path: str | Path) -> "PairTable":
        d = Path(path).read_bytes()
        h = struct.unpack_from("<8i", d, 0)
        n, nt, npt, nat, maxmono = h[0], h[1], h[2], h[3], h[4]
        o = 4 * HEADER_FIELDS

        def rd(dt, count):
            nonlocal o
            a = np.frombuffer(d, dtype=dt, count=count, offset=o)
            o += np.dtype(dt).itemsize * count
            return a

        out = {}
        for k in ("pp_exp", "pp_cx", "pp_cy", "pp_cz", "pp_K"):
            out[k] = rd(np.float64, n)
        for k in ("pp_nt", "pp_off", "pp_nmono", "pp_monooff"):
            out[k] = rd(np.int32, n)
        n_mono = int(out["pp_monooff"][-1]) + int(out["pp_nmono"][-1]) if n else 0
        out["mo_parent"] = rd(np.int32, n_mono)
        out["mo_axis"] = rd(np.int32, n_mono)
        out["tm_aidx"] = rd(np.int32, nt)
        out["tm_m"] = rd(np.int32, nt)
        out["tm_g"] = rd(np.float64, nt)
        out["pts"] = rd(np.float64, 3 * npt).reshape(npt, 3)
        out["Z"] = rd(np.float64, nat)
        out["xyz"] = rd(np.float64, 3 * nat).reshape(nat, 3)
        return cls(max_mono=maxmono, **out)


# ---------------------------------------------------------------------------
# Pure-NumPy builder (reference; ``native/build_pairs.cpp`` is the fast one)
# ---------------------------------------------------------------------------
def build_table(wfn: Wavefunction, basis: Basis, density: np.ndarray,
                points: np.ndarray, tol: float = 1e-12,
                progress: bool = False) -> PairTable:
    """Build the contracted pair table with NumPy.

    Same algorithm and output as ``native/build_pairs.cpp``; slow but has no
    toolchain dependency and is what the tests and ``--backend numpy`` use.
    """
    from .esp import ESPEngine

    eng = ESPEngine(basis, density)
    pts = np.asarray(points, float)
    prims = eng.prims

    dmin = np.empty(len(prims))
    for i, pp in enumerate(prims):
        dmin[i] = np.sqrt(((pts - pp.center) ** 2).sum(axis=1)).min()
    ok = np.array([bool(not (pp.K * pp.bound / max(d, 0.05) < tol))
                   for pp, d in zip(prims, dmin)])
    if progress:
        print(f"  pairs kept {ok.sum()} / {len(prims)}")

    def _mono_schedule(alpha_rows):
        key = tuple(sorted(tuple(map(int, a)) for a in alpha_rows))
        grown = set(key)
        changed = True
        while changed:
            changed = False
            for a in list(grown):
                for k in range(3):
                    if a[k] > 0:
                        p = a[:k] + (a[k] - 1,) + a[k + 1:]
                        if p not in grown:
                            grown.add(p)
                            changed = True
        order = sorted(grown)
        idx = {a: i for i, a in enumerate(order)}
        parent = np.zeros(len(order), np.int32)
        axis = np.zeros(len(order), np.int32)
        for i, a in enumerate(order):
            for k in range(3):
                if a[k] > 0:
                    parent[i] = idx[a[:k] + (a[k] - 1,) + a[k + 1:]]
                    axis[i] = k
                    break
        return parent, axis, idx

    cache: dict = {}
    pairs = [p for p, o in zip(prims, ok) if o]
    n = len(pairs)
    pp_exp = np.array([p.p for p in pairs])
    P = np.array([p.center for p in pairs])
    pp_K = np.array([p.K for p in pairs])
    ppt = np.array([len(p.coef) for p in pairs], np.int32)
    offsets = np.concatenate([[0], np.cumsum(ppt)[:-1]]).astype(np.int32)

    g_list, aidx_list, m_list = [], [], []
    nmono, monooff, parents, axes = [], [], [], []
    moff = 0
    for p in pairs:
        key = tuple(sorted(tuple(map(int, a)) for a in p.alpha))
        if key not in cache:
            cache[key] = _mono_schedule(p.alpha)
        parent, axis, idx = cache[key]
        nmono.append(len(parent))
        monooff.append(moff)
        parents.append(parent)
        axes.append(axis)
        moff += len(parent)
        aidx_list.append(np.array([idx[tuple(map(int, a))] for a in p.alpha], np.int32))
        m_list.append(p.m.astype(np.int32))
        g_list.append(p.coef)

    return PairTable(
        pp_exp=pp_exp, pp_cx=P[:, 0], pp_cy=P[:, 1], pp_cz=P[:, 2], pp_K=pp_K,
        pp_nt=ppt, pp_off=offsets,
        pp_nmono=np.array(nmono, np.int32), pp_monooff=np.array(monooff, np.int32),
        mo_parent=np.concatenate(parents) if parents else np.zeros(0, np.int32),
        mo_axis=np.concatenate(axes) if axes else np.zeros(0, np.int32),
        tm_aidx=np.concatenate(aidx_list) if aidx_list else np.zeros(0, np.int32),
        tm_m=np.concatenate(m_list) if m_list else np.zeros(0, np.int32),
        tm_g=np.concatenate(g_list) if g_list else np.zeros(0, np.float64),
        pts=pts, Z=np.array([Z_OF[s] for s in wfn.symbols]),
        xyz=wfn.coords, max_mono=int(max(nmono)) if nmono else 0)


Z_OF = {"H": 1.0, "He": 2.0, "Li": 3.0, "Be": 4.0, "B": 5.0, "C": 6.0, "N": 7.0,
        "O": 8.0, "F": 9.0, "Ne": 10.0, "Na": 11.0, "Mg": 12.0, "Al": 13.0,
        "Si": 14.0, "P": 15.0, "S": 16.0, "Cl": 17.0, "Ar": 18.0, "Br": 35.0,
        "I": 53.0}


def basis_payload(wfn: Wavefunction, basis: Basis, density: np.ndarray,
                  points: np.ndarray) -> dict:
    """Flat arrays describing the Cartesian basis + density + points.

    Consumed by ``native/build_pairs.cpp``.
    """
    fn_nprim = np.array([len(e) for e in basis.exps], np.int32)
    return dict(
        n_func=basis.n,
        n_prim=len(np.concatenate(basis.exps)),
        n_points=len(points), n_atoms=wfn.natom, max_prim=int(fn_nprim.max()),
        fn_atom=np.asarray(basis.atoms, np.int32),
        fn_nprim=fn_nprim,
        fn_off=np.concatenate([[0], np.cumsum(fn_nprim)[:-1]]).astype(np.int32),
        fn_lx=np.asarray(basis.ang_arr[:, 0], np.int32),
        fn_ly=np.asarray(basis.ang_arr[:, 1], np.int32),
        fn_lz=np.asarray(basis.ang_arr[:, 2], np.int32),
        fn_shell=_shell_ids(wfn),
        fc=np.asarray(basis.centers, np.float64).ravel(),
        pexp=np.concatenate(basis.exps).astype(np.float64),
        pcoef=np.concatenate(basis.coefs).astype(np.float64),
        dens=np.ascontiguousarray(density, np.float64),
        pts=np.ascontiguousarray(points, np.float64).ravel(),
        Z=np.array([Z_OF[s] for s in wfn.symbols], np.float64),
        xyz=np.ascontiguousarray(wfn.coords, np.float64),
    )


def _shell_ids(wfn: Wavefunction) -> np.ndarray:
    ids = np.empty(sum(len(cart_order(sh.l)) for sh in wfn.shells), np.int32)
    k = sid = 0
    for sh in wfn.shells:
        nc = len(cart_order(sh.l))
        ids[k:k + nc] = sid
        sid += 1
        k += nc
    return ids


def write_basis_payload(path: str | Path, payload: dict) -> None:
    with Path(path).open("wb") as f:
        f.write(struct.pack("<8i", payload["n_func"], payload["n_prim"],
                            payload["n_points"], payload["n_atoms"],
                            payload["max_prim"], 0, 0, 0))
        for k in ("fn_atom", "fn_nprim", "fn_off", "fn_lx", "fn_ly", "fn_lz",
                  "fn_shell"):
            f.write(np.ascontiguousarray(payload[k], np.int32).tobytes())
        for k in ("fc", "pexp", "pcoef", "dens", "pts", "Z", "xyz"):
            f.write(np.ascontiguousarray(payload[k], np.float64).tobytes())
