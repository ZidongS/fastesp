"""Minimal Multiwfn driver for the wavefunction ESP at an arbitrary point set.

Multiwfn takes its thread count from ``nthreads`` in ``settings.ini``, looked up
in the *current working directory* first, so every run gets its own scratch
directory; runs with different thread settings then cannot contaminate each
other and can proceed concurrently.

Menu stream (see the Multiwfn manual, "calculate ESP at a batch of points"):

    5      output property within a spatial region
    12     total electrostatic potential (ESP)
    100    load the point set from an external file
    <points file>      first line = count, then "x y z" in Bohr
    <output file>      one line per point: "x y z ESP"  (Hartree/e)
    q      quit

Timing model::

    t_total    = wall clock of a run with N points
    t_baseline = wall clock of a run that only loads the wavefunction and quits
    t_esp      = t_total - t_baseline      (the cost of the N points)
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

SETTINGS = 'nthreads= {nthreads}\norca_2mklpath= "{orca_dir}/orca_2mkl"\n'


def _env(mwfn: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["Multiwfnpath"] = str(mwfn.resolve().parent) + "/"
    env.setdefault("KMP_STACKSIZE", "4G")
    env.setdefault("OMP_STACKSIZE", "4G")
    return env


def _unlimited_stack():
    """Multiwfn needs an unlimited stack or it segfaults building the density."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    target = resource.RLIM_INFINITY if hard == resource.RLIM_INFINITY else hard
    if soft != target:
        resource.setrlimit(resource.RLIMIT_STACK, (target, hard))


def write_points(path: Path, points_bohr: np.ndarray) -> Path:
    with open(path, "w", encoding="ascii") as fh:
        fh.write(f"{len(points_bohr)}\n")
        for x, y, z in points_bohr:
            fh.write(f"{x:16.9f} {y:16.9f} {z:16.9f}\n")
    return path


def read_esp(path: Path) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 4:
                try:
                    rows.append([float(x) for x in f[:4]])
                except ValueError:
                    continue
    return np.asarray(rows)


@dataclass
class EspRun:
    n_points: int
    nthreads: int
    seconds: float
    baseline_seconds: float
    points_bohr: np.ndarray
    esp_hartree: np.ndarray
    log: Path

    @property
    def esp_seconds(self) -> float:
        return max(self.seconds - self.baseline_seconds, 0.0)

    @property
    def seconds_per_point(self) -> float:
        return self.esp_seconds / self.n_points if self.n_points else 0.0


def _call(cmd: Sequence[str], stdin: str, cwd: Path, mwfn: Path):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=stdin, text=True, cwd=cwd,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       check=False, env=_env(mwfn), preexec_fn=_unlimited_stack)
    return time.perf_counter() - t0, p.stdout, p.returncode


def baseline_seconds(molden: Path, workdir: Path, nthreads: int, mwfn: Path,
                     orca_dir: Path, repeats: int = 3) -> float:
    """Wall time of load-wavefunction-and-quit (fixed startup overhead)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "settings.ini").write_text(
        SETTINGS.format(nthreads=nthreads, orca_dir=orca_dir))
    cmd = [str(mwfn), str(molden.resolve()), "-ispecial", "1"]
    return float(np.mean([_call(cmd, "q\n", workdir, mwfn)[0]
                          for _ in range(repeats)]))


def run_esp(molden: Path, points_bohr: np.ndarray, workdir: Path,
            nthreads: int = 8, mwfn: Path = Path("Multiwfn_noGUI"),
            orca_dir: Path = Path("."), baseline: Optional[float] = None) -> EspRun:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "settings.ini").write_text(
        SETTINGS.format(nthreads=nthreads, orca_dir=orca_dir))

    pts_file = write_points(workdir / "points_bohr.txt", points_bohr)
    esp_file = workdir / "esp_out.txt"
    if esp_file.exists():
        esp_file.unlink()

    stdin = "\n".join(["5", "12", "100", str(pts_file), str(esp_file), "q"]) + "\n"
    log = workdir / "multiwfn.log"
    dt, out, rc = _call([str(mwfn), str(molden.resolve()), "-ispecial", "1"],
                        stdin, workdir, mwfn)
    log.write_text(out)
    if rc != 0:
        raise RuntimeError(f"Multiwfn exited {rc}; see {log}")
    if not esp_file.exists():
        raise RuntimeError(f"Multiwfn produced no ESP output; see {log}")

    raw = read_esp(esp_file)
    if len(raw) != len(points_bohr):
        raise RuntimeError(f"expected {len(points_bohr)} ESP values, got {len(raw)}")

    base = baseline if baseline is not None else baseline_seconds(
        molden, workdir / "_baseline", nthreads, mwfn, orca_dir)
    return EspRun(len(points_bohr), nthreads, dt, base, raw[:, :3], raw[:, 3], log)
