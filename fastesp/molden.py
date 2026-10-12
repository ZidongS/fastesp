"""Minimal MOLDEN reader.

Only what is needed to rebuild the electron density of a closed/open-shell
single-determinant wavefunction: atoms, contracted Gaussian shells and MO
coefficients.

Everything is kept in atomic units (Bohr).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

BOHR = 0.529177210903  # angstrom per bohr (CODATA 2018)


@dataclass
class Shell:
    """A contracted (or single-primitive) Gaussian shell on one atom."""
    atom: int          # 0-based atom index
    l: int             # angular momentum
    exps: np.ndarray   # (nprim,) exponents
    coeffs: np.ndarray  # (nprim,) contraction coefficients


@dataclass
class Wavefunction:
    symbols: list[str]
    coords: np.ndarray       # (natom, 3) Bohr
    shells: list[Shell]
    mo_energy: np.ndarray    # (nmo,)
    mo_occ: np.ndarray       # (nmo,)
    mo_coeff: np.ndarray     # (nao, nmo) spherical/Cartesian as stored
    spherical: dict[int, bool] = field(default_factory=dict)

    @property
    def natom(self) -> int:
        return len(self.symbols)

    @property
    def nao(self) -> int:
        return self.mo_coeff.shape[0]

    @property
    def nelec(self) -> float:
        return float(self.mo_occ.sum())


_ATOM_RE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")


def _read_section(lines: list[str], start: int) -> tuple[list[str], int]:
    """Return the lines of a section body (between this header and the next)."""
    body: list[str] = []
    i = start + 1
    while i < len(lines) and not lines[i].lstrip().startswith("["):
        body.append(lines[i])
        i += 1
    return body, i


def read_molden(path: str | Path) -> Wavefunction:
    lines = Path(path).read_text(errors="ignore").splitlines()

    headers = {}
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", s)
        if m:
            headers.setdefault(m.group(1).strip(), (i, m.group(2).strip()))

    def body(name: str) -> list[str]:
        return _read_section(lines, headers[name][0])[0]

    # ---- atoms -------------------------------------------------------
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in body("Atoms"):
        m = _ATOM_RE.match(line)
        if not m:
            continue
        symbols.append(m.group(1))
        # fields 4..6 are x, y, z, in AU (this reader only supports [Atoms] AU)
        coords.append([float(m.group(4)), float(m.group(5)), float(m.group(6))])
    coords_arr = np.asarray(coords, dtype=np.float64)

    # ---- basis -------------------------------------------------------
    shells: list[Shell] = []
    gto = body("GTO")
    i = 0
    while i < len(gto):
        s = gto[i].split()
        if len(s) == 0:
            i += 1
            continue
        if len(s) == 2 and s[0].lstrip("-").isdigit():
            atom = int(s[0]) - 1
            i += 1
            continue
        # shell header:  <l-letter> <nprim> <scale>
        l_char = s[0]
        l = "spdfghijkl".index(l_char.lower())
        nprim = int(s[1])
        exps = np.empty(nprim)
        cf = np.empty(nprim)
        for k in range(nprim):
            f = gto[i + 1 + k].split()
            exps[k] = float(f[0])
            cf[k] = float(f[1])
        shells.append(Shell(atom=atom, l=l, exps=exps, coeffs=cf))
        i += 1 + nprim

    spherical = {l: (f"{2 * l + 1}{'SPDFGH'[l]}" in headers) for l in range(5)}

    # ---- MOs ---------------------------------------------------------
    mo = body("MO")
    energies: list[float] = []
    occs: list[float] = []
    coeff_cols: list[tuple[np.ndarray, np.ndarray]] = []
    i = 0
    while i < len(mo):
        s = mo[i].strip()
        if s.startswith("Sym="):
            i += 1
            e = 0.0
            o = 0.0
            while i < len(mo):
                t = mo[i].strip()
                if t.startswith("Ene="):
                    e = float(t.split("=", 1)[1])
                elif t.startswith("Occup="):
                    o = float(t.split("=", 1)[1])
                elif t.startswith("Spin=") or not t:
                    pass
                else:
                    break
                i += 1
            idx: list[int] = []
            vals: list[float] = []
            while i < len(mo) and not mo[i].strip().startswith("Sym="):
                f = mo[i].split()
                if len(f) >= 2:
                    idx.append(int(f[0]))
                    vals.append(float(f[-1]))
                i += 1
            energies.append(e)
            occs.append(o)
            coeff_cols.append((np.asarray(idx, dtype=np.int64), np.asarray(vals)))
            continue
        i += 1

    nao = max(int(ix.max()) for ix, _ in coeff_cols)
    nmo = len(coeff_cols)
    coeff = np.zeros((nao, nmo))
    for j, (ix, v) in enumerate(coeff_cols):
        coeff[ix - 1, j] = v

    return Wavefunction(
        symbols=symbols,
        coords=coords_arr,
        shells=shells,
        mo_energy=np.asarray(energies),
        mo_occ=np.asarray(occs),
        mo_coeff=coeff,
        spherical=spherical,
    )
