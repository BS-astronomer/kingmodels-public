#!/usr/bin/env python3
"""
build_fig12_continuum.py
========================
Precomputes the dense continuum frozen-DF curves F_b(SFE) for the four
W0 shown in fig12 (the MC-validation panels), so that rebuild_paper_figures.py
fig12() can draw a smooth curve WITHOUT the ~15-min on-the-fly evaluation.

Each fbound_at_sfe() call is the analytic Eddington inversion + the
reconstruction-check / N_EPS ladder (see adams_df_boundfraction.py), so
this is the expensive part. Run it once; fig12() then reads the CSV.
Re-run only if the DF machinery changes.

Output: fig12_continuum.csv   (W0, SFE, F_b)   -- ~40 SFE points per W0.
"""
import sys
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
sys.path.append('../01_king_sfe_analysis')
sys.path.append('../04_adams_df_boundfraction')
from king_sfe_analysis import king_model
from adams_df_boundfraction import fbound_at_sfe

# same panels and x-ranges as rebuild_paper_figures.fig12()
XMAX = {3.0: 0.55, 6.0: 0.5, 9.0: 0.4, 12.0: 0.23}
N_SFE = 40
OUT = 'fig12_continuum.csv'


def main():
    rows = []
    for w, xmax in XMAX.items():
        m = king_model(w)
        for s in np.linspace(0.01, xmax, N_SFE):
            fb = fbound_at_sfe(m, float(s), convention='total')
            rows.append([w, s, fb])
            print(f'  W0={w:4.1f} SFE={s:.3f}  F_b={fb:.4f}', flush=True)
    np.savetxt(OUT, np.array(rows), fmt='%.6f', delimiter=', ',
               header='W0, SFE, F_b', comments='')
    print(f'\nWrote {OUT} -- rebuild_paper_figures.py fig12() picks it up.')


if __name__ == '__main__':
    main()
