"""Verify basis / density conventions through the total electron count.

For the Cartesian basis built by fastesp.basis the total number of electrons is
Tr(P_cart S_cart) and must come out as the MOLDEN occupancy sum (196 for frag54).
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastesp.molden import read_molden
from fastesp.basis import Basis, cart_order, _os1d, primitive_norm


def overlap_cart(b, block=64):
    """Cartesian overlap matrix, shell-block vectorised."""
    n = b.n
    S = np.zeros((n, n))
    # group functions by shell
    shells = []
    off = 0
    for sh in b.wfn.shells:
        nc = len(cart_order(sh.l))
        shells.append((off, nc))
        off += nc
    for si, (oi, ni) in enumerate(shells):
        for sj in range(si, len(shells)):
            oj, nj = shells[sj]
            blk = np.zeros((ni, nj))
            cI = b.centers[oi:oi + ni]; aI = b.ang_arr[oi:oi + ni]
            cJ = b.centers[oj:oj + nj]; aJ = b.ang_arr[oj:oj + nj]
            for mi in range(ni):
                for mj in range(nj):
                    A = cI[mi]; B = cJ[mj]
                    li = aI[mi]; lj = aJ[mj]
                    d = A - B
                    val = 0.0
                    for al, ca in zip(b.exps[oi + mi], b.coefs[oi + mi]):
                        for be, cb in zip(b.exps[oj + mj], b.coefs[oj + mj]):
                            p = al + be
                            pa = -be * d / p; pb = al * d / p
                            v = np.exp(-al * be / p * float(d @ d)) * (np.pi / p) ** 1.5
                            for k in range(3):
                                v *= _os1d(int(li[k]), int(lj[k]), pa[k], pb[k], p)
                            val += ca * cb * v
                    blk[mi, mj] = val
            S[oi:oi + ni, oj:oj + nj] = blk
            S[oj:oj + nj, oi:oi + ni] = blk.T
    return S


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/frag54.molden.input"
    w = read_molden(path)
    b = Basis(w)
    t0 = time.time()
    S = overlap_cart(b)
    print("overlap %.1fs" % (time.time() - t0))
    print("S diag: min %.8f max %.8f" % (np.diag(S).min(), np.diag(S).max()))
    c = w.mo_coeff[:, w.mo_occ > 1e-10] * np.sqrt(w.mo_occ[w.mo_occ > 1e-10])
    t = b.block_transform()
    P_sph = c @ c.T
    P_cart = t.T @ P_sph @ t
    print("nelec  (T^T P T) : %.8f" % float((P_cart * S).sum()))
    P_alt = t @ P_sph @ t.T
    print("nelec  (T P T^T) : %.8f" % float((P_alt * S).sum()))
    print("expected         : %.8f" % w.nelec)
