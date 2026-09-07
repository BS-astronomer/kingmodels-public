#!/usr/bin/env python3
"""
build_envelope_slopes.py
========================
Produces `envelope_slopes.csv`, the input `rebuild_paper_figures.fig13()`
reads. This script is the generator for that file
(see REPRODUCTION_GUIDE.md gotcha 5); this makes the bundle self-contained.

Definition (from the manuscript, Sect. "primordial clump" / Fig.
\\ref{fig:slopes}): the *effective power-law slope* p of a density profile
is minus the least-squares slope of ln(rho) vs ln(r), fitted between the
radii enclosing 10 and 90 per cent of that profile's own mass. It is
computed for

  * the total primordial clump   rho_tot = rho_star + rho_gas   (p_tot), and
  * the residual gas alone        rho_gas                        (p_gas),

at global SFE = 0.10, 0.15, 0.20, for a grid of King concentrations W0.

Output columns (exactly what fig13 expects):
    W0, pgas_sfe0.10, ptot_sfe0.10, pgas_sfe0.15, ptot_sfe0.15,
        pgas_sfe0.20, ptot_sfe0.20

Pure numpy/scipy via king_sfe_analysis; no Agama. Runtime: a few seconds.
"""

import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
from scipy.integrate import cumulative_trapezoid
import sys
sys.path.append('../01_king_sfe_analysis')
from king_sfe_analysis import king_model, gas_density, tsf_for_sfe

OUTDIR = './'
W0_GRID = [0.5, 1.0] + [float(w) for w in range(2, 21)]     # matches the paper's grid
SFE_LIST = (0.10, 0.15, 0.20)


def _enclosed_mass(x, rho):
    """M(<r) for a spherical density on radial grid x (G=M=r0=1 units)."""
    return 4.0 * np.pi * cumulative_trapezoid(rho * x**2, x, initial=0.0)


def _slope_10_90(x, rho, Menc):
    """-d ln rho / d ln r, least-squares over [r10, r90] of this profile."""
    Mtot = Menc[-1]
    if not np.isfinite(Mtot) or Mtot <= 0:
        return np.nan
    frac = Menc / Mtot
    r10 = np.interp(0.10, frac, x)
    r90 = np.interp(0.90, frac, x)
    sel = (x >= r10) & (x <= r90) & (rho > 0)
    if sel.sum() < 5:
        return np.nan
    p = np.polyfit(np.log(x[sel]), np.log(rho[sel]), 1)[0]
    return -p


def envelope_slopes(W0):
    m = king_model(W0)
    x, rho_s = m['x'], m['rho']
    out = {}
    for sfe in SFE_LIST:
        try:
            tsf = tsf_for_sfe(m, sfe)
        except Exception as e:                       # unreachable SFE for this W0
            print(f"  W0={W0:5.1f} SFE={sfe}: tsf_for_sfe failed ({e}); NaN")
            out[sfe] = (np.nan, np.nan)
            continue
        rho_g = gas_density(rho_s, tsf)
        rho_t = rho_s + rho_g
        p_gas = _slope_10_90(x, rho_g, _enclosed_mass(x, rho_g))
        p_tot = _slope_10_90(x, rho_t, _enclosed_mass(x, rho_t))
        out[sfe] = (p_gas, p_tot)
    return out


def main():
    rows = []
    for W0 in W0_GRID:
        s = envelope_slopes(W0)
        row = [W0]
        for sfe in SFE_LIST:
            row += [s[sfe][0], s[sfe][1]]
        rows.append(row)
        print(f"W0={W0:5.1f}  " + "  ".join(
            f"SFE{sfe}: p_gas={s[sfe][0]:.3f} p_tot={s[sfe][1]:.3f}"
            for sfe in SFE_LIST))
    hdr = ('W0, pgas_sfe0.10, ptot_sfe0.10, pgas_sfe0.15, ptot_sfe0.15, '
           'pgas_sfe0.20, ptot_sfe0.20')
    np.savetxt(OUTDIR + 'envelope_slopes.csv', np.array(rows), fmt='%.4f',
               delimiter=', ', header=hdr, comments='')
    print("\nWrote envelope_slopes.csv -- rebuild_paper_figures.py fig13() "
          "picks it up automatically.")


if __name__ == '__main__':
    main()
