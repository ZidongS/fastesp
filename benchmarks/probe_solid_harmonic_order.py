"""Work out ORCA's MOLDEN real-solid-harmonic convention from the data itself.

S_true = (C C^T)^-1 is the exact basis overlap implied by the orthonormal MOLDEN
MO block.  For a candidate Cartesian real-solid-harmonic basis we build the
stored-basis overlap  S_model = T^T S_cart T  and compare with S_true.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastesp.molden import read_molden                                   # noqa: E402
from fastesp.basis import Basis, cartesian_overlap_block, cart_order     # noqa: E402
from check_density import overlap_cart                                   # noqa: E402

# directions of the standard real solid harmonics (any common scale per row is fine)
SHAPES = {
    2: {"xy": {(1, 1, 0): 1.0}, "yz": {(0, 1, 1): 1.0},
        "3z2-r2": {(2, 0, 0): -1.0, (0, 2, 0): -1.0, (0, 0, 2): 2.0},
        "xz": {(1, 0, 1): 1.0}, "x2-y2": {(2, 0, 0): 1.0, (0, 2, 0): -1.0}},
    3: {"y(3x2-y2)": {(2, 1, 0): 3.0, (0, 3, 0): -1.0},
        "xyz": {(1, 1, 1): 1.0},
        "y(5z2-r2)": {(2, 1, 0): -1.0, (0, 3, 0): -1.0, (0, 1, 2): 4.0},
        "z(5z2-3r2)": {(2, 0, 1): -3.0, (0, 2, 1): -3.0, (0, 0, 3): 2.0},
        "x(5z2-r2)": {(3, 0, 0): -1.0, (1, 2, 0): -1.0, (1, 0, 2): 4.0},
        "z(x2-y2)": {(2, 0, 1): 1.0, (0, 2, 1): -1.0},
        "x(x2-3y2)": {(3, 0, 0): 1.0, (1, 2, 0): -3.0}},
}


def candidate_block(l, order=None, signs=None):
    """S-orthonormalised (2l+1) x ncart candidate solid harmonics."""
    S = cartesian_overlap_block(l)
    comps = {c: i for i, c in enumerate(cart_order(l))}
    names = list(SHAPES[l]) if order is None else list(order)
    V = np.zeros((len(names), len(S)))
    for r, nm in enumerate(names):
        for k, c in SHAPES[l][nm].items():
            V[r, comps[k]] = c
    if signs is not None:
        V = V * np.asarray(signs)[:, None]
    nrm = np.sqrt(np.einsum("ri,ij,rj->r", V, S, V))
    return V / nrm[:, None], names


def build_t(b, order_l, sign_l):
    """(ncart, nsph) transform using the candidate blocks for l>=2."""
    t = b.block_transform().copy()
    off = coff = 0
    for sh in b.wfn.shells:
        n = 2 * sh.l + 1
        nc = len(cart_order(sh.l))
        if sh.l >= 2:
            V, _ = candidate_block(sh.l, order_l[sh.l], sign_l[sh.l])
            t[coff:coff + nc, off:off + n] = V.T
        off += n
        coff += nc
    return t


def main():
    w = read_molden("data/frag54.molden.input")
    b = Basis(w)
    Sc = overlap_cart(b)
    St = np.linalg.inv(w.mo_coeff @ w.mo_coeff.T)
    base_order = {2: None, 3: None}
    base_sign = {2: [1] * 5, 3: [1] * 7}
    t = build_t(b, base_order, base_sign)
    S0 = t.T @ Sc @ t
    print("plain candidate (SHAPES order): max|S0-St| = %.3e" % np.abs(S0 - St).max())

    # brute force permutation + signs per l, using rows of l-shells against all columns
    offs = []
    off = 0
    for sh in w.shells:
        offs.append((off, 2 * sh.l + 1, sh))
        off += 2 * sh.l + 1
    for L in (2, 3):
        n = 2 * L + 1
        names = list(SHAPES[L])
        o, nn, sh = next(s for s in offs if s[2].l == L)
        rows = slice(o, o + nn)
        best = None
        for perm in itertools.permutations(range(n)):
            order = [names[p] for p in perm]
            V, _ = candidate_block(L, order)
            # S-orthonormalised candidate rows -> stored-basis values
            t = build_t(b, {2: order if L == 2 else None, 3: order if L == 3 else None},
                        {2: [1] * 5, 3: [1] * 7})
            S0 = t.T @ Sc @ t
            res = np.abs((S0 - St)[rows, :]).max()
            if best is None or res < best[0]:
                best = (res, perm)
        print("l=%d best perm res %.3e : %s" % (L, best[0], [names[p] for p in best[1]]))


if __name__ == "__main__":
    main()
