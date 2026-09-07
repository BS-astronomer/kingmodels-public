#!/usr/bin/env python3
"""
mc_bound_fraction_n1e5.py
=========================
Frozen-DF Monte Carlo at N = 1e5: the particle dataset behind the paper's
quoted thresholds and F_b(SFE) curves.

Two choices differ from mc_bound_fraction_agama.py, both so that the threshold
carries a real SAMPLING uncertainty rather than an SFE-grid half-width:

  * N_SEEDS = 150 per (W0, SFE) instead of 50, so the survival fraction
    p = P(F_b > F_MIN) is measured to a binomial standard error of at most
    sqrt(0.25/150) = 0.041;
  * a REFINED SFE grid straddling each transition (plus the original broad
    points, which are what the F_b(SFE) curves of Fig. 7a are drawn from),
    so the 0.5-crossing is interpolated over a small interval instead of a
    coarse one.

Per-realization rows are retained so that mc_thresholds_n1e5.py can bootstrap
over seeds to get the threshold and its uncertainty.

Cost: ~0.4 s per realization, so roughly an hour for the whole grid.
Output: mc_bound_fractions_n1e5.csv
        (W0, SFE_target, SFE_actual, seed, F_bound, n_iter, N_part)
"""
import sys
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
sys.path.append('../01_king_sfe_analysis')
import agama
from mc_bound_fraction_agama import (build_model, bound_fraction_particles,
                                     tsf_for_sfe)

N_PART = 100_000
N_SEEDS = 150
OUTCSV = 'mc_bound_fractions_n1e5.csv'

# broad points (curve shape, kept from the original run) + fine grid across
# the transition located by the earlier N=1e4 pass (0.302/0.190/0.047/0.023)
SFE_LIST = {
    3.0: sorted(set([0.25, 0.35, 0.40] + list(np.round(np.arange(0.270, 0.3351, 0.005), 4)))),
    6.0: sorted(set([0.16, 0.30, 0.40] + list(np.round(np.arange(0.155, 0.2201, 0.005), 4)))),
    9.0: sorted(set([0.10, 0.17, 0.20, 0.30] + list(np.round(np.arange(0.030, 0.0701, 0.0025), 4)))),
    12.0: sorted(set([0.12, 0.20, 0.30, 0.40] + list(np.round(np.arange(0.010, 0.0401, 0.002), 4)))),
}
SEEDS = list(np.random.default_rng(20260906).integers(1_000_000, 99_999_999,
                                                     size=N_SEEDS))


def main():
    total = sum(len(v) for v in SFE_LIST.values()) * N_SEEDS
    print(f'{total} realizations at N={N_PART:,}  (~{total*0.4/60:.0f} min)',
          flush=True)
    results = []
    done = 0
    for W0 in sorted(SFE_LIST):
        for sfe_t in SFE_LIST[W0]:
            tsf = tsf_for_sfe(W0, sfe_t)
            pot_total, king_pot, df, sfe_act = build_model(W0, tsf)
            gm = agama.GalaxyModel(potential=pot_total, df=df)
            Fs = []
            for seed in SEEDS:
                agama.setRandomSeed(int(seed))
                np.random.seed(seed)
                xv, _ = gm.sample(N_PART)
                F, nit = bound_fraction_particles(xv[:, :3], xv[:, 3:])
                results.append([W0, sfe_t, sfe_act, seed, F, nit, N_PART])
                Fs.append(F)
                done += 1
            Fs = np.array(Fs)
            surv = float(np.mean(Fs > 0.02))
            print(f'  W0={W0:4.1f} SFE={sfe_act:.4f}  <F_b>={Fs.mean():.4f} '
                  f'survival={surv:.3f}  [{done}/{total}]', flush=True)
            np.savetxt(OUTCSV, np.array(results), fmt='%.6f', delimiter=', ',
                       header='W0, SFE_target, SFE_actual, seed, F_bound, '
                              'n_iter, N_part', comments='')
    print(f'\nWrote {OUTCSV} ({len(results)} rows)')


if __name__ == '__main__':
    main()
