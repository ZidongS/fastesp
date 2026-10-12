# fastesp

[![tests](https://github.com/ZidongS/fastesp/actions/workflows/tests.yml/badge.svg)](https://github.com/ZidongS/fastesp/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/fastesp.svg)](https://pypi.org/project/fastesp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Exact electrostatic potential from a Gaussian-basis wavefunction — evaluated
analytically, and solved on the GPU.**

`fastesp` computes

$$
V(\mathbf{C}) = \sum_{A} \frac{Z_A}{\lvert \mathbf{C} - \mathbf{R}_A \rvert}
- \sum_{\mu\nu} P_{\mu\nu}
\Big\langle \mu \Big\lvert \frac{1}{\lvert \mathbf{r} - \mathbf{C} \rvert} \Big\rvert \nu \Big\rangle
$$

at an arbitrary set of points, without any real-space quadrature. The
one-electron Coulomb integrals are evaluated in closed form (Hermite /
McMurchie–Davidson expansion of the Gaussian product, Boys functions for the
$1/r$ operator), so the result is exact to machine precision. It is a drop-in
faster and more accurate replacement for the "electrostatic potential at a set
of points" function of Multiwfn (menu `5 → 12 → 100`).

## Benchmark

54-atom protein fragment, ORCA wB97M-D3BJ/def2-TZVPPD (1450 spherical / 1634
Cartesian functions), points on the union of the Bondi vdW spheres. Multiwfn
8 CPU threads vs one RTX 5090, measured with `benchmarks/benchmark.py` under
SLURM on a dedicated node.

| points | Multiwfn 1 thr | Multiwfn 8 thr | fastesp (1 GPU) | speedup |
|---|---|---|---|---|
| 300   | 206.07 s | 26.29 s | 0.269 s | **98x** |
| 3000  | 2061 s\* | 258.57 s | 0.479 s | **539x** |
| 30000 | 20607 s\* | 2586 s\* | 2.456 s | **>1000x** |

\* extrapolated; Multiwfn is linear in the point count at 86.2–87.6 ms/point.
The part that replaces Multiwfn's integration is **0.030 ms/point vs 87.6** —
a **2900x** speedup — because the table build is point-independent.

Accuracy at the identical 300 points:

| | |
|---|---|
| max abs difference vs Multiwfn | 1.75e-7 a.u. = **4.8e-3 mV** |
| max abs difference vs unscreened analytic evaluation | 1.3e-11 a.u. |
| primitive integrals vs independent quadrature | 6.6e-14 (relative) |
| electron density vs Multiwfn at the same points | 4.3e-12 (9e-10 relative) |

The 1.75e-7 a.u. gap is **Multiwfn's** quadrature error, and this is pinned down
by three measurements: (1) the electron density — a pure basis evaluation on
Multiwfn's side — agrees to 9e-10 relative, so both programs read the
wavefunction identically; (2) Multiwfn's answer is bit-for-bit unchanged when
its Becke grid is swept from 75×434 (1.8e6 points) to 200×2354 (2.5e7 points);
(3) its three internal ESP engines (`iESPcode` 1/2/3) agree exactly with each
other. See `benchmarks/check_density_mwfn.py` and
`benchmarks/mwfn_grid_convergence.py`.

## Install

```bash
pip install fastesp              # NumPy reference (exact, no toolchain needed)
fastesp build-native             # optional: compile the C++/CUDA backends
fastesp selftest                 # validate the analytic integrals
```

The native backends are compiled into `~/.cache/fastesp` (override with
`FASTESP_CACHE`) and need `g++` (with OpenMP) and, for the GPU path, `nvcc`.
`pip install` alone already gives exact results — just slower.

## Use

```bash
# ESP at the points in pts.txt (Bohr, one "x y z" per line)
fastesp esp mol.molden.input --points pts.txt -o esp.txt
fastesp esp mol.molden.input --points pts.txt --backend numpy   # force reference
```

```
  backend      : cuda
  basis        : 1634 Cartesian functions, 2080 primitives
  pair table   :   0.2600 s  (point-independent)
  ESP kernel   :   0.0089 s
  ESP range    : -2.2055 .. +1.9854 V
```

As a library:

```python
import numpy as np
from fastesp import read_molden, Basis, build_table

wfn    = read_molden("mol.molden.input")
basis  = Basis(wfn)
dens   = basis.density_matrix_cartesian()

table  = build_table(wfn, basis, dens, points_bohr)   # point-independent tables
esp    = table.esp()                                  # Hartree/e
```

`examples/vdw_surface_esp.py` shows the full production workflow: sample the
Bondi vdW surface of a molecule and evaluate the ESP on it.

## How it works

For a pair of primitive Cartesian Gaussians the product is a single Gaussian
times a polynomial. Expanding about the Gaussian product centre $\mathbf{P}$ in
Hermite Gaussians $\Lambda_{\mathbf{t}}$ gives the exact closed form

$$
g_A^{\mathbf{a}} g_B^{\mathbf{b}}
  = K \sum_{\mathbf{t}} E_{\mathbf{t}} \Lambda_{\mathbf{t}}(\mathbf{r} - \mathbf{P}),
\qquad
\int \frac{\Lambda_{\mathbf{t}}(\mathbf{r}')}{\lvert \mathbf{R} - \mathbf{r}' \rvert} d\mathbf{r}'
  = (-1)^{\lvert \mathbf{t} \rvert}
        \left(\frac{\partial}{\partial \mathbf{R}}\right)^{\mathbf{t}} F(\mathbf{R}),
\qquad
F(\mathbf{R}) = \frac{2\pi}{p} F_0\left(p R^2\right)
$$

and the derivative is expanded exactly by the one-dimensional recurrence

$$
a_{t+1,j} = -a_{t,j} + (t - 2j + 2)a_{t,j-1}.
$$

The structural step that makes this fast: **everything except
$\mathbf{R} = \mathbf{C} - \mathbf{P}$ is independent of the sampling point.**
Contracting the $E$-coefficients, the $(-1)^{\lvert \mathbf{t} \rvert}$ signs and
the $(2p)^{m}$ factors over the Cartesian components and over the primitive
pairs — weighted by the density matrix — yields, once per primitive pair, a
sparse table $G[\boldsymbol{\alpha}, m]$ such that

$$
I_{\text{pair}}(\mathbf{C}) = K \sum_{\boldsymbol{\alpha}, m}
  G[\boldsymbol{\alpha}, m]  \mathbf{R}^{\boldsymbol{\alpha}}  F_m\left(p R^2\right).
$$

Per point that is one Boys evaluation, a short monomial schedule and one dot
product — a shape that maps directly onto GPU threads. The table build is
$\mathcal{O}(N_{\text{pairs}})$ and independent of the point set (and of how many
points there are), so it is done once and reused; only the
$\mathcal{O}(N_{\text{pairs}} \times N_{\text{points}})$ evaluation runs per
point set.

Screening uses the rigorous bound

$$
\lvert I_{\text{pair}} \rvert \le K  \frac{\int \lvert \rho_{\text{pair}} \rvert  d\mathbf{r}}{d},
$$

where $\int \lvert \rho_{\text{pair}} \rvert  d\mathbf{r}$ comes from the
$\lvert He_n \rvert$ moment products computed at build time. The builder also
prints the sum rule

$$
\mathrm{Tr}(\mathbf{P}\mathbf{S}) = N_{\text{elec}}
$$

as a sanity check on the basis it just read.

### MOLDEN conventions

The MOLDEN format does not pin these down. All three were determined from the
file itself and are verified by $\mathrm{Tr}(\mathbf{P}\mathbf{S}) = N_{\text{elec}}$
and $\max\lvert \mathbf{C}^{\mathsf{T}} \mathbf{S} \mathbf{C} - \mathbf{I} \rvert$:

1. **Primitive normalisation.** ORCA writes GTO coefficients that already include

   $$
   N_s(\alpha, l) = \left(\frac{2\alpha}{\pi}\right)^{3/4} (4\alpha)^{l/2},
   $$

   the radial prefactor *without* the angular $(2l-1)!!$ factor of a fully
   normalised Cartesian Gaussian. Verified against every single-primitive p/d/f
   shell in the test file to 8 digits.
2. **Angular ordering.** The spherical functions are the real solid harmonics in
   MOLDEN order $m = 0, +1, -1, +2, -2, \dots$ (`fastesp/basis.py`, `_POLY`).
3. **Cartesian ↔ spherical transform.** Rows are S-orthonormalised so that
   $\mathbf{T} \mathbf{S} \mathbf{T}^{\mathsf{T}} = \mathbf{I}$.

If you have a wavefunction whose conventions differ, `Basis` + `overlap_matrix`
will show it: $\mathrm{Tr}(\mathbf{P}\mathbf{S})$ will not reproduce the MOLDEN
occupancy sum.

## Repository layout

| path | role |
|---|---|
| `fastesp/molden.py` | MOLDEN reader |
| `fastesp/basis.py` | Cartesian basis, conventions, density matrix |
| `fastesp/ints.py` | exact primitive-pair Coulomb integrals |
| `fastesp/esp.py` | contracted pair tables + NumPy ESP |
| `fastesp/table.py` | table construction, binary I/O, evaluation |
| `fastesp/cli.py` | `fastesp` console script |
| `fastesp/native/build_pairs.cpp` | OpenMP table builder (fast path) |
| `fastesp/native/esp_gpu.cu` | CUDA solver |
| `tests/` | integral and end-to-end validation |
| `benchmarks/` | head-to-head benchmark and cross-checks |

## Limitations

* Only the vacuum wavefunction ESP is implemented. Adding external point charges
  (e.g. a CPCM reaction field) is a one-line addition to the nuclear term.
* $l = 4$ (g) solid harmonics are tabulated but unverified — the test file has no
  g functions.
* The GPU solver is fp64 throughout; there is headroom for a mixed-precision
  build if needed.
* Tested against ORCA-generated MOLDEN files; other writers may use different
  conventions (see above).

## Citation

If you use this in published work, please cite the repository
(https://github.com/ZidongS/fastesp) — see `CITATION.cff`.

## License

MIT — see `LICENSE`.
