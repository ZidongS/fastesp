#!/usr/bin/env python3
"""Cross-check our basis/density against Multiwfn's, point by point.

The electron density is a pure basis-function evaluation (Multiwfn sums the
basis products directly, no quadrature), so if our MOLDEN reading or density
matrix were wrong this would show up immediately.  Agreement here, combined
with a 1e-7 a.u. ESP difference, localises Multiwfn's error in its Becke
quadrature of the ESP rather than in the wavefunction.

    V_el(C) = -int rho(r)/|C-r| dr          <- Multiwfn: numerical quadrature
                                            <- ours:     exact analytic
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastesp.molden import read_molden   # noqa: E402
from fastesp.basis import Basis          # noqa: E402

# override with MULTIWFN_BIN / ORCA_DIR for your installation
MWFN = Path(os.environ.get("MULTIWFN_BIN", "Multiwfn_noGUI"))
ORCA_DIR = Path(os.environ.get("ORCA_DIR", ""))


def _unlimited():
    import resource
    s, h = resource.getrlimit(resource.RLIMIT_STACK)
    t = resource.RLIM_INFINITY if h == resource.RLIM_INFINITY else h
    if s != t:
        resource.setrlimit(resource.RLIMIT_STACK, (t, h))


def multiwfn_density(molden: Path, pts_bohr: Path, work: Path, nthreads=8) -> np.ndarray:
    work.mkdir(parents=True, exist_ok=True)
    (work / "settings.ini").write_text(
        f'nthreads= {nthreads}\norca_2mklpath= "{ORCA_DIR}/orca_2mkl"\n')
    out = work / "rho.txt"
    stdin = "\n".join(["5", "1", "100", str(pts_bohr.resolve()),
                       str(out.resolve()), "q"]) + "\n"
    env = os.environ.copy()
    env["Multiwfnpath"] = str(MWFN.parent) + "/"
    env.setdefault("KMP_STACKSIZE", "4G")
    env.setdefault("OMP_STACKSIZE", "4G")
    p = subprocess.run([str(MWFN), str(molden.resolve()), "-ispecial", "1"],
                       input=stdin, text=True, cwd=work, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       preexec_fn=_unlimited)
    if not out.exists():
        raise RuntimeError(p.stdout[-2000:])
    return np.loadtxt(out, skiprows=1)[:, 3]


def our_density(molden: Path, pts_bohr: Path) -> np.ndarray:
    w = read_molden(molden)
    b = Basis(w)
    P = b.density_matrix_cartesian()
    pts = np.loadtxt(pts_bohr, skiprows=1)
    phi = np.empty((len(pts), b.n))
    for j in range(b.n):
        x = pts[:, 0] - b.centers[j][0]
        y = pts[:, 1] - b.centers[j][1]
        z = pts[:, 2] - b.centers[j][2]
        lx, ly, lz = b.ang[j]
        acc = np.zeros(len(pts))
        for a, c in zip(b.exps[j], b.coefs[j]):
            acc += c * np.exp(-a * (x * x + y * y + z * z))
        phi[:, j] = (x ** lx) * (y ** ly) * (z ** lz) * acc
    return np.einsum("ni,ij,nj->n", phi, P, phi)


def main():
    molden = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/frag54.molden.input"
    pts = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/pts_bohr.txt"
    ours = our_density(molden, pts)
    theirs = multiwfn_density(molden, pts, ROOT / "results/rho")
    d = ours - theirs
    print(f"rho range {ours.min():.6e} .. {ours.max():.6e}")
    print(f"max abs difference {np.abs(d).max():.3e}   "
          f"max rel {(np.abs(d)/np.abs(theirs)).max():.3e}   "
          f"mean {d.mean():+.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
