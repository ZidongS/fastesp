"""fastesp -- exact electrostatic potential from a Gaussian-basis wavefunction.

The electron part of the ESP at a point C is

    V_el(C) = -sum_{mu nu} P_{mu nu} < mu | 1/|r - C| | nu >

evaluated *analytically*: the Gaussian product is expanded in Hermite Gaussians
(McMurchie-Davidson) and the potential of each Hermite Gaussian is a closed-form
derivative of the Boys function.  No real-space quadrature is involved, so the
result is exact to machine precision.

Quick start::

    from fastesp import read_molden, Basis, build_table

    wfn = read_molden("mol.molden.input")
    basis = Basis(wfn)
    density = basis.density_matrix_cartesian()
    table = build_table(wfn, basis, density, points_bohr)
    esp = table.esp()                      # (n_points,) in Hartree/e

See :mod:`fastesp.cli` (console script ``fastesp``) for the compiled C++/CUDA
backends, which are orders of magnitude faster on large point sets.
"""
from __future__ import annotations

from .basis import Basis, cart_order, overlap_matrix
from .esp import ESPEngine
from .ints import boys, primitive_pair_integral
from .molden import Wavefunction, read_molden
from .table import (PairTable, basis_payload, build_table, write_basis_payload)

__version__ = "0.1.0"

__all__ = [
    "Wavefunction", "read_molden",
    "Basis", "cart_order", "overlap_matrix",
    "boys", "primitive_pair_integral",
    "PairTable", "build_table",
    "basis_payload", "write_basis_payload",
    "__version__",
]
