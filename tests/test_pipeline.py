"""End-to-end checks that need no external wavefunction file.

A tiny synthetic MOLDEN is built on the fly, so the whole chain -- MOLDEN
reading, spherical -> Cartesian basis, density matrix, contracted pair table,
evaluation -- is exercised and compared against a direct numerical quadrature
of the density.
"""
from __future__ import annotations

import numpy as np
import pytest

from fastesp import Basis, build_table, read_molden
from fastesp.basis import (cart_order, cartesian_overlap_block,
                           overlap_matrix, pure_to_cartesian)

TEMPLATE = """[Molden Format]
[Atoms] AU
{symbols}
[GTO]
{shells}
[5D]
[7F]
[MO]
{mos}
"""


def _spherical_size(shells):
    """Number of spherical functions -- what MOLDEN stores."""
    return sum(2 * l + 1 for per_atom in shells for (l, _, _) in per_atom)


def _write_molden(path, atoms, shells, mo_coeffs, occ=2.0):
    sym = "\n".join(f"{s} {i+1} {z} {x:.10f} {y:.10f} {w:.10f}"
                    for i, (s, z, (x, y, w)) in enumerate(atoms))
    sh = []
    for iatom in range(len(atoms)):
        sh.append(f" {iatom+1} 0")
        for (l, exps, coefs) in shells[iatom]:
            sh.append(f"{'spdfg'[l]}   {len(exps)} 1.0 ")
            for a, c in zip(exps, coefs):
                sh.append(f"   {a:20.10f} {c:20.10f}")
    mos = []
    for k in range(mo_coeffs.shape[1]):
        mos.append(f" Sym=     {k+1}a")
        mos.append(" Ene=    -0.50000000000000E+00")
        mos.append(" Spin= Alpha")
        mos.append(f" Occup=  {occ:.6f}")
        for i, v in enumerate(mo_coeffs[:, k]):
            mos.append(f"{i+1:4d} {v:20.12f}")
    path.write_text(TEMPLATE.format(symbols=sym, shells="\n".join(sh),
                                    mos="\n".join(mos)))
    return path


def make_wfn(tmp_path, atoms, shells, nocc, seed=0, name="mol.molden.input"):
    """Build a MOLDEN whose MOs are orthonormal in the MOLDEN basis metric.

    A random Euclidean-orthonormal coefficient matrix is *not* enough: the
    stored basis is generally non-orthogonal, so the coefficients are Loewdin
    orthogonalised against the overlap implied by the basis itself.  This is the
    same self-consistency (C^T S C = I) that a real SCF wavefunction satisfies,
    and that fastesp relies on to pin down the MOLDEN conventions.
    """
    nsp = _spherical_size(shells)
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(nsp, nsp)))[0]
    path = _write_molden(tmp_path / name, atoms, shells, q[:, :nocc])

    wfn = read_molden(path)
    basis = Basis(wfn)
    t = basis.block_transform()
    s_sph = t.T @ overlap_matrix(basis) @ t
    w, v = np.linalg.eigh(s_sph)
    x = v @ np.diag(w ** -0.5) @ v.T
    _write_molden(path, atoms, shells, x @ q[:, :nocc])
    return read_molden(path), path


# ---------------------------------------------------------------------------
def test_spherical_transform_is_orthonormal():
    for l in (0, 1, 2, 3, 4):
        T = pure_to_cartesian(l)
        S = cartesian_overlap_block(l)
        assert np.abs(T @ S @ T.T - np.eye(2 * l + 1)).max() < 1e-14


def test_mos_are_orthonormal_in_the_basis(tmp_path):
    """The synthetic wavefunction itself must satisfy C^T S C = I."""
    atoms = [("O", 8, (0.0, 0.0, 0.0)), ("H", 1, (1.8, 0.0, 0.0))]
    shells = [[(0, [3.5, 0.7], [0.5, 0.6]), (1, [1.2], [0.7])],
              [(0, [1.0], [0.6])]]
    wfn, _ = make_wfn(tmp_path, atoms, shells, nocc=4)
    basis = Basis(wfn)
    t = basis.block_transform()
    s_sph = t.T @ overlap_matrix(basis) @ t
    C = wfn.mo_coeff[:, wfn.mo_occ > 1e-10]
    assert np.abs(C.T @ s_sph @ C - np.eye(C.shape[1])).max() < 1e-10


def test_electron_count_sum_rule(tmp_path):
    """Tr(P S) must equal the MOLDEN occupancy sum."""
    atoms = [("C", 6, (0.0, 0.0, 0.0))]
    shells = [[(0, [5.0, 1.0], [0.4, 0.5]), (1, [0.8], [0.6]),
               (2, [0.6], [0.4])]]
    wfn, _ = make_wfn(tmp_path, atoms, shells, nocc=3, name="c.molden.input")
    basis = Basis(wfn)
    P = basis.density_matrix_cartesian()
    S = overlap_matrix(basis)
    assert float((P * S).sum()) == pytest.approx(wfn.nelec, abs=1e-9)


def test_pair_table_matches_direct_quadrature(tmp_path):
    """Contracted pair table vs a spherical quadrature of rho(r)/|C-r|."""
    atoms = [("O", 8, (0.0, 0.0, 0.0)), ("H", 1, (1.8, 0.0, 0.0))]
    shells = [[(0, [3.5, 0.7], [0.5, 0.6]), (1, [1.2], [0.7])],
              [(0, [1.0], [0.6])]]
    wfn, _ = make_wfn(tmp_path, atoms, shells, nocc=4)
    basis = Basis(wfn)
    density = basis.density_matrix_cartesian()

    pts = np.array([[3.0, 0.4, 0.3], [-1.5, 1.2, -0.7]])
    got = build_table(wfn, basis, density, pts, tol=0.0).electron()
    want = _quad_electron(wfn, basis, density, pts)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-10)


def _phi(wfn, basis, X, Y, Z):
    out = []
    for j in range(basis.n):
        x, y, z = X - basis.centers[j][0], Y - basis.centers[j][1], Z - basis.centers[j][2]
        lx, ly, lz = map(int, basis.ang[j])
        acc = np.zeros_like(X)
        for a, c in zip(basis.exps[j], basis.coefs[j]):
            acc += c * np.exp(-a * (x * x + y * y + z * z))
        out.append((x ** lx) * (y ** ly) * (z ** lz) * acc)
    return np.stack(out)


def _quad_electron(wfn, basis, density, pts, R=14.0, ngl=160, ncos=140, nphi=280):
    """int rho(r)/|C-r| dr by quadrature about C (no singularity: u = C + s*n)."""
    from numpy.polynomial.legendre import leggauss
    xs, ws = leggauss(ngl)
    s = 0.5 * R * (xs + 1.0)
    wr = 0.5 * R * ws
    cz, wz = leggauss(ncos)
    phi_a = (np.arange(nphi) + 0.5) * (2 * np.pi / nphi)
    wphi = 2 * np.pi / nphi
    st = np.sqrt(np.clip(1 - cz ** 2, 0, None))
    nx = st[None, :, None] * np.cos(phi_a)[:, None, None] * np.ones((1, 1, 1))
    ny = st[None, :, None] * np.sin(phi_a)[:, None, None] * np.ones((1, 1, 1))
    nz = np.broadcast_to(cz[None, :, None], (nphi, ncos, 1)).copy()
    ss = s[None, None, :]
    wang = wz[None, :, None] * wphi
    out = np.zeros(len(pts))
    for ip, C in enumerate(pts):
        X, Y, Z = C[0] + ss * nx, C[1] + ss * ny, C[2] + ss * nz
        phi = _phi(wfn, basis, X, Y, Z)            # (nbasis, nphi, ncos, ngl)
        sh = phi.shape
        pf = phi.reshape(sh[0], -1)
        rho = np.einsum("an,ab,bn->n", pf, density, pf).reshape(sh[1:])
        out[ip] = float((rho * ss * wr[None, None, :] * wang).sum())
    return out


def test_screening_does_not_change_the_answer(tmp_path):
    atoms = [("C", 6, (0.0, 0.0, 0.0)), ("H", 1, (2.0, 0.0, 0.0))]
    shells = [[(0, [4.0, 0.9], [0.5, 0.5]), (1, [0.7], [0.6])],
              [(0, [1.0], [0.6])]]
    wfn, _ = make_wfn(tmp_path, atoms, shells, nocc=4)
    basis = Basis(wfn)
    density = basis.density_matrix_cartesian()
    pts = np.array([[3.2, 0.5, -1.1], [-2.0, -1.0, 0.9], [0.1, 2.4, 0.4]])
    full = build_table(wfn, basis, density, pts, tol=0.0).electron()
    for tol in (1e-8, 1e-10, 1e-12):
        got = build_table(wfn, basis, density, pts, tol=tol).electron()
        assert np.abs(got - full).max() < 1e-9
