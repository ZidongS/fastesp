#!/usr/bin/env python3
"""Compute the ESP on a Bondi vdW surface -- the production use case.

    python examples/vdw_surface_esp.py mol.molden.input --n-points 3000

Points are placed on the union of the atomic Bondi vdW spheres; the potential is
then evaluated with the fastest compiled backend available.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastesp import Basis, read_molden                       # noqa: E402
from fastesp.cli import native_paths, run_esp                # noqa: E402

BONDI = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
BOHR = 0.529177210903
EH_TO_V = 27.211386245988


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def vdw_surface_points(wfn, n_points: int, seed: int = 0) -> np.ndarray:
    """Approximately uniform points on the exposed Bondi vdW surface (Bohr)."""
    rad = np.array([BONDI[s] / BOHR for s in wfn.symbols])
    share = (4 * np.pi * rad ** 2) / (4 * np.pi * rad ** 2).sum()
    rng = np.random.default_rng(seed)
    out = []
    for a in range(wfn.natom):
        n = int(np.ceil(n_points * share[a] * 3.0)) + 50
        p = wfn.coords[a] + rad[a] * fibonacci_sphere(n)
        d = np.linalg.norm(p[:, None, :] - wfn.coords[None, :, :], axis=2)
        out.append(p[(d >= rad[None, :] - 1e-9).all(axis=1)])
    cand = np.concatenate(out)
    if len(cand) > n_points:
        cand = cand[rng.choice(len(cand), n_points, replace=False)]
    return cand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavefunction", type=Path)
    ap.add_argument("--n-points", type=int, default=3000)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args()

    pts = vdw_surface_points(read_molden(a.wavefunction), a.n_points)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "fastesp_example_pts.txt"
    with tmp.open("w") as f:
        f.write(f"{len(pts)}\n")
        for x, y, z in pts:
            f.write(f"{x:16.9f} {y:16.9f} {z:16.9f}\n")

    cpu, gpu = native_paths()
    backend = os.environ.get("FASTESP_BACKEND") or ("cuda" if gpu else "numpy")
    print(f"surface points : {len(pts)}")
    esp = run_esp(a.wavefunction, tmp, a.out, backend=backend, tol=a.tol)
    print(f"ESP            : {esp.min() * EH_TO_V:+.3f} .. {esp.max() * EH_TO_V:+.3f} V"
          f"   mean {esp.mean() * EH_TO_V:+.4f} V")
    if a.out:
        print(f"written to     : {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
