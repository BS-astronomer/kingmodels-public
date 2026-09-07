#!/usr/bin/env python3
"""
build_published_continuum.py
============================
Precomputes the dense continuum frozen-DF curve F_b(SFE_10) for the two
published families shown in the lower panel of the validation figure
(manuscript Fig. 7): Plummer and Dehnen gamma=0.

Each fbound_at_sfe() call is an analytic Eddington inversion plus the
reconstruction-check / N_EPS ladder (~1.6 s), so this is the expensive
part. Run it once; rebuild_paper_figures.fig12() then reads the CSV and
needs no DF evaluation at all. Re-run only if the DF machinery
changes.

Output: published_continuum.csv  (family, SFE_10, F_b)
        family: 0 = Plummer, 1 = Dehnen gamma=0
"""
import sys
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
sys.path += ['../01_king_sfe_analysis', '../03_multi_profile_analysis',
             '../04_adams_df_boundfraction']
from multi_profile_analysis import plummer_model, dehnen_model
from adams_df_boundfraction import fbound_at_sfe

N_SFE = 35
OUT = 'published_continuum.csv'


def main():
    fams = [(0, 'Plummer', plummer_model()),
            (1, 'Dehnen g=0', dehnen_model(0.0))]
    rows = []
    for code, name, m in fams:
        for s in np.linspace(0.02, 0.70, N_SFE):
            fb = fbound_at_sfe(m, float(s), convention='ap10')
            rows.append([code, s, fb])
            print(f'  {name:12s} SFE10={s:.3f}  F_b={fb:.4f}', flush=True)
    np.savetxt(OUT, np.array(rows), fmt='%.6f', delimiter=', ',
               header='family(0=Plummer;1=Dehnen g=0), SFE_10, F_b', comments='')
    print(f'\nWrote {OUT} -- rebuild_paper_figures.fig8_validation() reads it.')


if __name__ == '__main__':
    main()
