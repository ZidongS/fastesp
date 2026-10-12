#!/usr/bin/env python3
"""Show that Multiwfn's 1.8e-7 a.u. difference from us is *its* grid error.

Multiwfn evaluates the wavefunction ESP by Becke numerical quadrature; its grid
is radpot x sphpot points per atom (settings.ini).  Refining the grid must move
Multiwfn towards the exact analytic value, which is what this script measures.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# override with MULTIWFN_BIN / ORCA_DIR for your installation
MWFN = Path(os.environ.get("MULTIWFN_BIN", "Multiwfn_noGUI"))
ORCA_DIR = Path(os.environ.get("ORCA_DIR", ""))


def unlimited_stack():
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    target = resource.RLIM_INFINITY if hard == resource.RLIM_INFINITY else hard
    if soft != target:
        resource.setrlimit(resource.RLIMIT_STACK, (target, hard))


def run(molden: Path, pts: Path, work: Path, radpot: int, sphpot: int,
        nthreads: int = 8):
    work.mkdir(parents=True, exist_ok=True)
    (work / "settings.ini").write_text(
        f'nthreads= {nthreads}\n'
        f'orca_2mklpath= "{ORCA_DIR}/orca_2mkl"\n'
        f'iautointgrid= 0\n'
        f'radpot= {radpot}\n'
        f'sphpot= {sphpot}\n')
    out = work / "esp.txt"
    if out.exists():
        out.unlink()
    stdin = "\n".join(["5", "12", "100", str(pts.resolve()), str(out.resolve()), "q"]) + "\n"
    env = os.environ.copy()
    env["Multiwfnpath"] = str(MWFN.parent) + "/"
    env.setdefault("KMP_STACKSIZE", "4G")
    env.setdefault("OMP_STACKSIZE", "4G")
    t0 = time.perf_counter()
    p = subprocess.run([str(MWFN), str(molden.resolve()), "-ispecial", "1"],
                       input=stdin, text=True, cwd=work, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       preexec_fn=unlimited_stack)
    dt = time.perf_counter() - t0
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(p.stdout[-3000:])
    return dt, np.loadtxt(out, skiprows=1)[:, 3]


def main():
    molden = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/frag54.molden.input")
    pts = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/pts_bohr.txt")
    ours = Path(sys.argv[3] if len(sys.argv) > 3 else ROOT / "results/bench/gpu_300.npy")
    work = ROOT / "results/gridconv"
    ref = np.load(ours)

    combos = [(75, 434), (100, 590), (120, 974), (150, 1454), (200, 2354)]
    print(f"{'radpot':>7s} {'sphpot':>7s} {'grid pts':>12s} {'wall s':>9s} "
          f"{'max|Mwfn-exact| a.u.':>22s} {'mV':>10s}")
    for radpot, sphpot in combos:
        dt, v = run(molden, pts, work / f"{radpot}_{sphpot}", radpot, sphpot)
        d = np.abs(v - ref).max()
        print(f"{radpot:7d} {sphpot:7d} {radpot*sphpot*54:12d} {dt:9.1f} "
              f"{d:22.4e} {d*27.211386245988*1000:10.4f}", flush=True)


if __name__ == "__main__":
    main()
