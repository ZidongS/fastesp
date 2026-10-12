#!/usr/bin/env python3
"""Head-to-head benchmark: Multiwfn wavefunction ESP vs the exact GPU solver.

Run under sbatch on a 5090 node; see sbatch/benchmark.sbatch.

Sections
  1. Multiwfn load-only baseline (both thread settings)
  2. Multiwfn ESP at the reference 300 point set (1 and 8 threads)
  3. Multiwfn ESP at the 3000 point set (8 threads)
  4. Our pipeline: dump basis -> build contracted pair table -> GPU solve
  5. Accuracy: our ESP against Multiwfn at the identical points
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = Path("/home/shuzidong/workfile/multipole2potential/experiment/frag_surface_esp")

sys.path.insert(0, str(ROOT))
from fastesp import Basis, read_molden                      # noqa: E402
from fastesp.table import basis_payload, write_basis_payload  # noqa: E402
sys.path.insert(0, str(EXP / "scripts"))

MWFN = Path("/home/shuzidong/workfile/dielec/QMSEP/multiwfn_install/"
            "Multiwfn_2026.4.10_bin_Linux_noGUI/Multiwfn_noGUI")
ORCA_DIR = Path("/home/shuzidong/workfile/dielec/QMSEP/orca_install/"
                "orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg")
EH_TO_V = 27.211386245988


def sh(cmd, **kw):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{cmd} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return dt, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molden", default=str(ROOT / "data/frag54.molden.input"))
    ap.add_argument("--work", default=str(ROOT / "results/bench"))
    ap.add_argument("--mw-cpu-points", type=int, default=3000,
                    help="largest point set handed to Multiwfn")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--skip-multiwfn", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    molden = Path(args.molden).resolve()
    rows = []

    point_sets = {
        300: ROOT / "data/pts_bohr.txt",
        3000: ROOT / "data/pts3000_bohr.txt",
        30000: ROOT / "data/pts30000_bohr.txt",
    }
    pts = {n: np.loadtxt(p, skiprows=1) for n, p in point_sets.items()}

    # ---------------- Multiwfn -------------------------------------------
    mw = {}
    if not args.skip_multiwfn:
        import run_esp as RE

        base = {}
        for nt in (1, 8):
            base[nt] = RE.baseline_seconds(molden, work / f"base{nt}", nt,
                                           mwfn=MWFN, orca_dir=ORCA_DIR, repeats=3)
            print(f"  Multiwfn load-only baseline nthreads={nt}: {base[nt]:.2f} s",
                  flush=True)

        for npts, nt in ((300, 1), (300, 8), (args.mw_cpu_points, 8)):
            if npts > args.mw_cpu_points:
                continue
            runs = []
            for _ in range(args.repeats):
                r = RE.run_esp(molden, pts[npts], work / f"mw_{npts}_{nt}",
                               nthreads=nt, mwfn=MWFN, orca_dir=ORCA_DIR,
                               baseline=base[nt])
                runs.append(r)
            r = runs[-1]
            esp = r.esp_hartree
            mw[(npts, nt)] = esp
            rows.append(dict(method=f"Multiwfn ({nt} CPU threads)",
                             n_points=npts, nthreads=nt,
                             wall_s=float(np.mean([x.seconds for x in runs])),
                             load_s=base[nt],
                             esp_s=float(np.mean([x.esp_seconds for x in runs])),
                             ms_per_point=float(np.mean([x.esp_seconds for x in runs]))
                             / npts * 1e3))
            print(f"  Multiwfn {npts} pts, {nt} threads: "
                  f"{rows[-1]['esp_s']:.2f} s ESP-only "
                  f"({rows[-1]['ms_per_point']:.2f} ms/point)", flush=True)
            if npts == 300:
                np.savetxt(work / f"mw_{npts}_{nt}_esp.txt", np.column_stack([r.points_bohr, r.esp_hartree]))

    # ---------------- our pipeline ---------------------------------------
    for npts in (300, 3000, 30000):
        pfile = point_sets[npts]
        basis_bin = work / f"basis_{npts}.bin"
        table = work / f"pairs_{npts}.bin"
        out = work / f"gpu_{npts}.txt"

        # wavefunction load + basis build + payload dump: the "load" half of the
        # pipeline, comparable to Multiwfn's own wavefunction load
        t0 = time.perf_counter()
        wfn = read_molden(str(molden))
        bs = Basis(wfn)
        dens = bs.density_matrix_cartesian()
        payload = basis_payload(wfn, bs, dens, np.loadtxt(pfile, skiprows=1))
        write_basis_payload(basis_bin, payload)
        t_dump = time.perf_counter() - t0

        t0 = time.perf_counter()
        sh([str(ROOT / "bin/build_pairs"), str(basis_bin), str(table), "1e-12"])
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        dt, p = sh([str(ROOT / "bin/esp_gpu"), str(table), str(out),
                    "--repeats", "5"])
        t_all = time.perf_counter() - t0
        gpu_ms = min(float(l.split()[-2]) for l in p.stdout.splitlines()
                     if l.startswith("run "))
        t_solve = gpu_ms / 1e3

        g = np.loadtxt(out, skiprows=1)[:, 3]
        rows.append(dict(method="this work (GPU, exact analytic)",
                         n_points=npts, nthreads=1,
                         wall_s=t_dump + t_build + t_solve,
                         load_s=t_dump, esp_s=t_build + t_solve,
                         ms_per_point=(t_build + t_solve) / npts * 1e3))
        print(f"  ours {npts} pts: dump {t_dump:.2f} s + build {t_build:.3f} s "
              f"+ solve {t_solve*1e3:.2f} ms  ({1000*(t_build+t_solve)/npts:.4f} ms/point)",
              flush=True)
        if npts == 300:
            np.save(work / "gpu_300.npy", g)

    # ---------------- accuracy -------------------------------------------
    acc = {}
    if (300, 8) in mw:
        g = np.load(work / "gpu_300.npy")
        m = mw[(300, 8)]
        d = g - m
        acc = dict(
            n_points=300,
            max_abs_au=float(np.abs(d).max()),
            rms_au=float(np.sqrt((d ** 2).mean())),
            max_abs_mV=float(np.abs(d).max() * EH_TO_V * 1000),
            rms_mV=float(np.sqrt((d ** 2).mean()) * EH_TO_V * 1000),
            esp_min_V=float(g.min() * EH_TO_V), esp_max_V=float(g.max() * EH_TO_V),
        )
        print(f"  accuracy vs Multiwfn: max {acc['max_abs_au']:.3e} a.u. "
              f"({acc['max_abs_mV']:.4f} mV), rms {acc['rms_mV']:.4f} mV", flush=True)

    with (work / "results.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    (work / "accuracy.json").write_text(json.dumps(acc, indent=2))
    (work / "results.json").write_text(json.dumps(rows, indent=2))

    # ---------------- summary --------------------------------------------
    print("\n================ SUMMARY ================")
    print(f"{'method':34s} {'points':>7s} {'ESP s':>10s} {'ms/point':>10s}")
    for r in rows:
        print(f"{r['method']:34s} {r['n_points']:7d} {r['esp_s']:10.4f} "
              f"{r['ms_per_point']:10.4f}")
    if rows:
        base = [r for r in rows if r["method"].startswith("Multiwfn (8") ]
        mine = [r for r in rows if r["method"].startswith("this work")]
        for b in base:
            for m in mine:
                if m["n_points"] == b["n_points"]:
                    print(f"  speedup at {b['n_points']} points vs Multiwfn 8 threads: "
                          f"{b['esp_s']/max(m['esp_s'],1e-9):.1f}x")


if __name__ == "__main__":
    main()
