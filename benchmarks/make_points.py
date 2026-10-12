"""Generate larger ESP sampling sets on the same Bondi vdW surface.

The reference set in frag_data_test lies exactly on the union of the atomic
Bondi vdW spheres.  This reproduces that construction with a denser per-atom
spherical grid so the two methods can be compared on bigger workloads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastesp.molden import read_molden   # noqa: E402

BONDI = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
BOHR = 0.529177210903


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molden", default="data/frag54.molden.input")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--exposed-only", action="store_true", default=True)
    args = ap.parse_args()

    w = read_molden(args.molden)
    rad = np.array([BONDI[s] / BOHR for s in w.symbols])
    area = 4 * np.pi * rad ** 2
    share = area / area.sum()
    rng = np.random.default_rng(args.seed)

    cand = []
    for a in range(w.natom):
        # oversample, then keep only points that are on the exposed vdW surface
        n = int(np.ceil(args.n * share[a] * 3.0)) + 50
        pts = w.coords[a] + rad[a] * fibonacci_sphere(n)
        d = np.linalg.norm(pts[:, None, :] - w.coords[None, :, :], axis=2)
        ok = (d >= rad[None, :] - 1e-9).all(axis=1)
        cand.append(pts[ok])
    cand = np.concatenate(cand, axis=0)
    print(f"candidates {len(cand)}")
    if len(cand) > args.n:
        sel = rng.choice(len(cand), size=args.n, replace=False)
        cand = cand[sel]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"{len(cand)}\n")
        for x, y, z in cand:
            f.write(f"{x:16.9f} {y:16.9f} {z:16.9f}\n")
    print(f"wrote {out} with {len(cand)} points")


if __name__ == "__main__":
    main()
