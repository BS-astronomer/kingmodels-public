#!/usr/bin/env python3
"""
df_positivity_check.py
======================
Is the Eddington-inverted distribution function f(eps) non-negative over
the whole energy range?

Eddington inversion can return a formally correct reconstruction of
rho_star while f itself develops negative regions, so the reconstruction
self-check in bound_fraction() does not answer this. Production code
clips (f = max(f_raw, 0)), which makes the returned f trivially
non-negative; the question is how much is being clipped, and where.

For each King model (W0 = 0.5-20) at its own frozen-DF threshold SFE, and
for the four published families at their published thresholds, we record
the PRE-CLIP diagnostic already returned by eddington_f_analytic:

  neg_frac              fraction of energy-grid nodes with f_raw < 0
  worst_rel_negativity  min(f_raw) / max(f)  -- the depth of the most
                        negative excursion relative to the peak of the DF

Output: df_positivity.csv  (kind, W0, SFE, neg_frac, worst_rel_neg, fmax)
"""
import sys
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
sys.path += ['../01_king_sfe_analysis', '../03_multi_profile_analysis']
from king_sfe_analysis import king_model, analyze, tsf_for_sfe
from multi_profile_analysis import plummer_model, dehnen_model, analyze_generic
from adams_df_boundfraction import (eddington_f_analytic, relative_potential,
                                    threshold_by_bisection)

W0_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
PUB = [('Plummer', None, 0.150), ('Dehnen g=0', 0.0, 0.090),
       ('Dehnen g=1', 1.0, 0.0241), ('Dehnen g=2', 2.0, 0.0148)]
OUT = 'df_positivity.csv'


def diag_for(model, rho_g):
    # psi_t exactly as _bound_fraction_at computes it
    psi_tot, _ = relative_potential(model['x'], model['rho'] + rho_g)
    psi_t = psi_tot[-1] if np.isfinite(model.get('rt', np.inf)) else 0.0
    _, f, d = eddington_f_analytic(model, rho_g, psi_t, return_diag=True)
    return d, float(f.max())


def main():
    rows = []
    print('KING FAMILY (at each model\'s frozen-DF threshold)')
    for w in W0_LIST:
        m = king_model(float(w))
        try:
            sfe = threshold_by_bisection(m)[0]
        except Exception:
            sfe = 0.10
        r = analyze(m, tsf_for_sfe(m, float(sfe)))
        d, fmax = diag_for(m, r['rho_g'])
        rows.append([0, w, sfe, d['neg_frac'], d['worst_rel_negativity'], fmax])
        print(f'  W0={w:4.1f} SFE={sfe:.4f}  neg_frac={d["neg_frac"]:.3e} '
              f'worst_rel_neg={d["worst_rel_negativity"]:+.3e}', flush=True)

    print('\nPUBLISHED FAMILIES (at published thresholds)')
    for name, g, thr in PUB:
        m = plummer_model() if g is None else dehnen_model(g)
        from multi_profile_analysis import tsf_for_threshold
        try:
            r = analyze_generic(m, tsf_for_threshold(m, thr, 'ap10'))
        except Exception as e:
            print(f'  {name}: skipped ({e})'); continue
        d, fmax = diag_for(m, r['rho_g'])
        code = 1 if g is None else 2 + int(g)
        rows.append([code, np.nan, thr, d['neg_frac'],
                     d['worst_rel_negativity'], fmax])
        print(f'  {name:12s} SFE={thr:.4f}  neg_frac={d["neg_frac"]:.3e} '
              f'worst_rel_neg={d["worst_rel_negativity"]:+.3e}', flush=True)

    a = np.array(rows, dtype=float)
    np.savetxt(OUT, a, fmt='%.8g', delimiter=', ',
               header='kind(0=King;1=Plummer;2..4=Dehnen g=0/1/2), W0, SFE, '
                      'neg_frac, worst_rel_negativity, f_max', comments='')
    nf, wr = a[:, 3], a[:, 4]
    print(f'\n{"="*62}\nSUMMARY over {len(a)} models')
    print(f'  max neg_frac             = {np.nanmax(nf):.4e}')
    print(f'  worst relative negativity = {np.nanmin(wr):+.4e}')
    print(f'  models with ANY f_raw<0   = {int(np.sum(nf > 0))}/{len(a)}')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
