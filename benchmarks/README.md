# benchmarks

Head-to-head measurement of `fastesp` against Multiwfn's wavefunction ESP, plus
the cross-checks that localise the residual difference.

Submit on a GPU node — never on a login node:

```bash
sbatch sbatch/benchmark.sbatch
```

| script | what it does |
|---|---|
| `benchmark.py` | Multiwfn (1 and 8 threads) vs `fastesp` at 300 / 3000 / 30000 points; writes `results/bench/results.csv` and `accuracy.json` |
| `check_integrals.py` | analytic primitive integrals vs a closed form and vs spherical quadrature |
| `check_density_mwfn.py` | electron density at the same points, ours vs Multiwfn's — isolates wavefunction handling from ESP integration |
| `mwfn_grid_convergence.py` | sweeps Multiwfn's `radpot`/`sphpot` Becke grid to show its ESP is grid-insensitive |
| `check_basis.py` | `Tr(P S)` and `max|C^T S C - I|` for a given MOLDEN, i.e. are the conventions right |
| `probe_solid_harmonic_order.py` | reverse-engineers the spherical-harmonic ordering from the MOLDEN MO orthonormality |
| `make_points.py` | generates denser Bondi vdW surface point sets matching the reference file |

## Tool paths

`benchmark.py`, `check_density_mwfn.py` and `mwfn_grid_convergence.py` shell out
to Multiwfn and ORCA's `orca_2mkl`. Edit the `MWFN` / `ORCA_DIR` constants at the
top of those files for your installation.

## Test case

`data/frag54.molden.input` — 54-atom protein fragment, ORCA
wB97M-D3BJ/def2-TZVPPD, 1450 spherical / 1634 Cartesian functions, 196 electrons.
`data/pts_bohr.txt` is the reference 300-point Bondi vdW surface sampling set.

## Reference numbers (RTX 5090, dedicated node)

```
method                              points      ESP s   ms/point
Multiwfn (1 CPU threads)               300   206.0700   686.8999
Multiwfn (8 CPU threads)               300    26.2877    87.6255
Multiwfn (8 CPU threads)              3000   258.5724    86.1908
fastesp (GPU)                          300     0.2688     0.8959
fastesp (GPU)                         3000     0.4793     0.1598
fastesp (GPU)                        30000     2.4564     0.0819

accuracy vs Multiwfn at 300 points: max 1.751e-07 a.u. (0.0048 mV)
```
