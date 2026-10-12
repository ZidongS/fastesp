"""Command line interface for fastesp.

    fastesp esp WF.molden --points pts_bohr.txt -o esp.txt
    fastesp build-native            # compile the C++/CUDA backends on demand
    fastesp selftest                # validate the analytic integrals

``fastesp esp`` picks the fastest backend it can: the CUDA solver if a GPU and
``nvcc`` are available, else the OpenMP C++ table builder with the NumPy
evaluator, else the pure NumPy reference.  ``build-native`` compiles both
backends into a per-user cache directory.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import __version__
from .basis import Basis
from .molden import read_molden
from .table import PairTable, basis_payload, build_table, write_basis_payload

NATIVE_DIR = Path(__file__).resolve().parent / "native"


# ---------------------------------------------------------------------------
def cache_dir() -> Path:
    d = Path(os.environ.get("FASTESP_CACHE",
                            Path.home() / ".cache" / "fastesp"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _which(name: str) -> str | None:
    return shutil.which(name)


def _nvcc() -> str | None:
    cand = [os.environ.get("NVCC"), "nvcc", "/usr/local/cuda/bin/nvcc"]
    for c in cand:
        if not c:
            continue
        p = c if os.sep in c else shutil.which(c)
        if p and Path(p).exists():
            return p
    return None


def _src_hash() -> str:
    h = hashlib.sha256()
    for f in ("build_pairs.cpp", "esp_gpu.cu"):
        h.update((NATIVE_DIR / f).read_bytes())
    return h.hexdigest()[:12]


def _default_arch() -> str:
    return os.environ.get("FASTESP_CUDA_ARCH", "sm_120")


def native_paths() -> tuple[Path | None, Path | None]:
    """(build_pairs, esp_gpu) if already compiled, else (None, None)."""
    if os.environ.get("FASTESP_NATIVE_DIR"):
        d = Path(os.environ["FASTESP_NATIVE_DIR"])
        b, g = d / "build_pairs", d / "esp_gpu"
        return (b if b.exists() else None), (g if g.exists() else None)
    d = cache_dir() / f"{_src_hash()}-{_default_arch()}"
    b, g = d / "build_pairs", d / "esp_gpu"
    return (b if b.exists() else None), (g if g.exists() else None)


def build_native(force: bool = False, arch: str | None = None,
                 quiet: bool = False) -> tuple[Path | None, Path | None]:
    arch = arch or _default_arch()
    out = cache_dir() / f"{_src_hash()}-{arch}"
    out.mkdir(parents=True, exist_ok=True)
    cpu, gpu = out / "build_pairs", out / "esp_gpu"

    cxx = _which(os.environ.get("CXX", "g++"))
    if cxx is None:
        print("fastesp: no C++ compiler found (set CXX)", file=sys.stderr)
    elif force or not cpu.exists():
        for extra in (["-march=native"], []):
            cmd = [cxx, "-O3", "-fopenmp", *extra, "-o", str(cpu),
                   str(NATIVE_DIR / "build_pairs.cpp")]
            if not quiet:
                print("  " + " ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                break
            if extra:
                continue
            print(r.stderr[-2000:], file=sys.stderr)
            cpu = None
    nv = _nvcc()
    if nv is None:
        if not quiet:
            print("fastesp: nvcc not found; GPU backend unavailable", file=sys.stderr)
    elif force or not gpu.exists():
        cmd = [nv, "-O3", f"-arch={arch}", "-o", str(gpu), str(NATIVE_DIR / "esp_gpu.cu")]
        if not quiet:
            print("  " + " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            gpu = None
    return (cpu if cpu.exists() else None), (gpu if gpu.exists() else None)


# ---------------------------------------------------------------------------
def run_esp(molden: Path, points_file: Path, out_file: Path | None,
            backend: str = "auto", tol: float = 1e-12, geom: str = "bohr",
            verbose: bool = True) -> np.ndarray:
    t0 = time.perf_counter()
    wfn = read_molden(molden)
    basis = Basis(wfn)
    density = basis.density_matrix_cartesian()
    t_load = time.perf_counter() - t0

    pts = _read_points(points_file, geom)
    cpu, gpu = native_paths()
    if backend == "auto":
        backend = "cuda" if (gpu and _has_gpu()) else ("cpp" if cpu else "numpy")
    if backend == "cuda" and not gpu:
        raise SystemExit("fastesp: GPU backend not built; run `fastesp build-native`")

    t_solve_kernel = None
    t0 = time.perf_counter()
    if backend == "numpy":
        table = build_table(wfn, basis, density, pts, tol=tol)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        esp = table.esp()
        t_solve = time.perf_counter() - t0
    else:
        with tempfile.TemporaryDirectory(prefix="fastesp-") as td:
            td = Path(td)
            basis_bin, pairs_bin = td / "basis.bin", td / "pairs.bin"
            write_basis_payload(basis_bin, basis_payload(wfn, basis, density, pts))
            r = subprocess.run([str(cpu), str(basis_bin), str(pairs_bin), str(tol)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"build_pairs failed:\n{r.stderr[-2000:]}")
            t_build = time.perf_counter() - t0
            t0 = time.perf_counter()
            if backend == "cuda":
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as fh:
                    tmp_out = fh.name
                reps = int(os.environ.get("FASTESP_REPEATS", "5"))
                r = subprocess.run([str(gpu), str(pairs_bin), tmp_out,
                                    "--repeats", str(reps)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f"esp_gpu failed:\n{r.stderr[-2000:]}")
                esp = np.loadtxt(tmp_out, skiprows=1)[:, 3]
                os.unlink(tmp_out)
                # the kernel time excludes process start-up (CUDA context setup),
                # which would otherwise dominate for small point sets
                t_solve_kernel = min(
                    float(l.split()[-2]) for l in r.stdout.splitlines()
                    if l.startswith("run ")) / 1e3
            else:
                esp = PairTable.read(pairs_bin).esp()
                t_solve_kernel = None
            t_solve = time.perf_counter() - t0

    if verbose:
        print(f"  backend      : {backend}")
        print(f"  basis        : {basis.n} Cartesian functions, "
              f"{len(np.concatenate(basis.exps))} primitives")
        print(f"  load         : {t_load:8.3f} s")
        print(f"  pair table   : {t_build:8.4f} s  (point-independent)")
        if t_solve_kernel is not None:
            print(f"  ESP kernel   : {t_solve_kernel:8.4f} s")
            print(f"  ESP incl. process start-up: {t_solve:.4f} s "
                  f"(CUDA context set-up; amortised when many point sets are run)")
            tot = t_build + t_solve_kernel
        else:
            print(f"  ESP solve    : {t_solve:8.4f} s")
            tot = t_build + t_solve
        print(f"  total (excl. load): {tot:.4f} s for {len(pts)} points "
              f"({1e3 * tot / len(pts):.4f} ms/point)")
        print(f"  ESP range    : {esp.min() * 27.211386245988:+.4f} .. "
              f"{esp.max() * 27.211386245988:+.4f} V")

    if out_file:
        with Path(out_file).open("w") as f:
            f.write(f"{len(pts)}\n")
            for (x, y, z), v in zip(pts, esp):
                f.write(f" {x:20.12f} {y:20.12f} {z:20.12f} {v:22.14e}\n")
        if verbose:
            print(f"  wrote        : {out_file}")
    return esp


def _has_gpu() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


def _read_points(path: Path, geom: str) -> np.ndarray:
    txt = Path(path).read_text().splitlines()
    rows = []
    for line in txt:
        f = line.split()
        if len(f) >= 3:
            try:
                rows.append([float(x) for x in f[:3]])
            except ValueError:
                continue
    pts = np.asarray(rows, float)
    if pts.shape[1] != 3:
        raise SystemExit(f"fastesp: cannot parse points from {path}")
    if geom == "angstrom":
        pts = pts / 0.529177210903
    return pts


# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> int:
    """Validate the analytic integrals against independent references."""
    import math
    from .ints import primitive_pair_integral
    from numpy.polynomial.legendre import leggauss

    rng = np.random.default_rng(0)
    worst_ss = 0.0
    for _ in range(8):
        A = rng.normal(0, 1, 3); B = A + rng.normal(0, 1, 3); C = rng.normal(0, 3, 3)
        al = float(np.exp(rng.uniform(-2, 2.5))); be = float(np.exp(rng.uniform(-2, 2.5)))
        p = al + be; P = (al * A + be * B) / p
        K = math.exp(-al * be / p * float(((A - B) ** 2).sum()))
        R = float(np.linalg.norm(C - P))
        ref = K * (np.pi / p) ** 1.5 * math.erf(np.sqrt(p) * R) / R
        got = primitive_pair_integral(al, A, (0, 0, 0), be, B, (0, 0, 0), C)
        worst_ss = max(worst_ss, abs(got - ref) / abs(ref))

    def quad(C, A, la, al, B, lb, be, R=14.0, ngl=240, ncos=220, nphi=440):
        xs, ws = leggauss(ngl)
        s = 0.5 * R * (xs + 1.0); wr = 0.5 * R * ws
        cz, wz = leggauss(ncos)
        phi = (np.arange(nphi) + 0.5) * (2 * np.pi / nphi); wphi = 2 * np.pi / nphi
        st = np.sqrt(np.clip(1 - cz ** 2, 0, None))
        nx = st[None, :, None] * np.cos(phi)[:, None, None] * np.ones((1, 1, 1))
        ny = st[None, :, None] * np.sin(phi)[:, None, None] * np.ones((1, 1, 1))
        nz = np.broadcast_to(cz[None, :, None], (nphi, ncos, 1)).copy()
        ss = s[None, None, :]
        X, Y, Z = C[0] + ss * nx, C[1] + ss * ny, C[2] + ss * nz
        rho = ((X - A[0]) ** la[0] * (Y - A[1]) ** la[1] * (Z - A[2]) ** la[2]
               * np.exp(-al * ((X - A[0]) ** 2 + (Y - A[1]) ** 2 + (Z - A[2]) ** 2))
               * (X - B[0]) ** lb[0] * (Y - B[1]) ** lb[1] * (Z - B[2]) ** lb[2]
               * np.exp(-be * ((X - B[0]) ** 2 + (Y - B[1]) ** 2 + (Z - B[2]) ** 2)))
        return float((rho * ss * wr[None, None, :]
                      * (wz[None, :, None] * wphi)).sum())

    A = np.array([0.3, -0.4, 0.2]); B = A + np.array([1.1, 0.5, -0.6])
    C = np.array([0.1, 1.6, 1.1]); al, be = 0.45, 0.72
    worst = 0.0
    for la, lb in [((1, 0, 0), (0, 0, 0)), ((2, 0, 0), (0, 1, 0)),
                   ((1, 1, 0), (0, 0, 1)), ((3, 2, 0), (2, 1, 0)),
                   ((3, 3, 3), (3, 3, 3))]:
        ana = primitive_pair_integral(al, A, la, be, B, lb, C)
        ref = quad(C, A, la, al, B, lb, be)
        worst = max(worst, abs(ana - ref) / max(abs(ref), 1e-300))
        if verbose:
            print(f"  <{la}|1/r|{lb}>  analytic {ana:+.10e}  quadrature {ref:+.10e}"
                  f"  rel {abs(ana - ref) / max(abs(ref), 1e-300):.1e}")
    ok = worst_ss < 1e-13 and worst < 1e-11
    print(f"  s-s vs closed form   worst rel {worst_ss:.1e}")
    print(f"  general vs quadrature worst rel {worst:.1e}")
    print("  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("fastesp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"fastesp {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("esp", help="electrostatic potential at a set of points")
    e.add_argument("wavefunction", type=Path, help="MOLDEN file")
    e.add_argument("--points", type=Path, required=True,
                   help="text file, one 'x y z' per line (Bohr unless --angstrom)")
    e.add_argument("-o", "--out", type=Path, default=None)
    e.add_argument("--backend", choices=["auto", "cuda", "cpp", "numpy"], default="auto")
    e.add_argument("--tol", type=float, default=1e-12,
                   help="primitive pair screening threshold in Hartree/e")
    e.add_argument("--angstrom", action="store_true", help="input points in Angstrom")
    e.add_argument("-q", "--quiet", action="store_true")

    b = sub.add_parser("build-native", help="compile the C++/CUDA backends")
    b.add_argument("--arch", default=None, help="CUDA arch, e.g. sm_120")
    b.add_argument("--force", action="store_true")

    s = sub.add_parser("selftest", help="validate the analytic integrals")
    s.add_argument("-q", "--quiet", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "esp":
        run_esp(a.wavefunction, a.points, a.out, backend=a.backend, tol=a.tol,
                geom="angstrom" if a.angstrom else "bohr", verbose=not a.quiet)
        return 0
    if a.cmd == "build-native":
        cpu, gpu = build_native(force=a.force, arch=a.arch)
        print(f"  build_pairs: {cpu}")
        print(f"  esp_gpu    : {gpu}")
        return 0 if cpu or gpu else 1
    if a.cmd == "selftest":
        return selftest(verbose=not a.quiet)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
