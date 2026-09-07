#!/usr/bin/env python3
"""
mc_threshold_convergence.py
===========================
Is the frozen-DF Monte Carlo threshold converged in particle number?

The thresholds the paper quotes are measured at N = 1e5. Because the survival
probability at fixed SFE falls with N (see mc_finiteN_scaling.py), the
threshold itself drifts upward with N, and the quoted values are only useful
if that drift has flattened by 1e5. This re-measures the threshold at a higher
N on the same SFE grid, so the two can be compared directly.

Only the SFE points that bracket each transition are re-run: away from the
transition the outcome is deterministic at every N and contributes nothing.

Usage:  python mc_threshold_convergence.py [N_part] [reference_csv]
        defaults: 300000  mc_bound_fractions_n1e5.csv
Output: mc_bound_fractions_n<N>.csv, in the same schema as the reference, so
        mc_thresholds_n1e5.py can be pointed at it unchanged.
"""
import sys
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
sys.path.append('../01_king_sfe_analysis')
import agama
from mc_bound_fraction_agama import (build_model, bound_fraction_particles,
                                     tsf_for_sfe)

F_MIN = 0.02
N_SEEDS = 150
PAD = 3          # grid points kept either side of the transition


def main():
    N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    ref = sys.argv[2] if len(sys.argv) > 2 else 'mc_bound_fractions_n1e5.csv'
    out = f'mc_bound_fractions_n{N_PART}.csv'
    d = np.loadtxt(ref, delimiter=',', skiprows=1, ndmin=2)
    seeds = np.random.default_rng(90909).integers(1_000_000, 99_999_999,
                                                  size=N_SEEDS)
    rows = []
    for W0 in np.unique(d[:, 0]):
        sub = d[d[:, 0] == W0]
        sfes = np.unique(np.round(sub[:, 2], 6))
        p = np.array([np.mean(sub[np.round(sub[:, 2], 6) == s, 4] > F_MIN)
                      for s in sfes])
        idx = [i for i in range(1, len(sfes)) if p[i - 1] < 0.5 <= p[i]]
        if not idx:
            print(f'W0={W0}: no crossing in {ref}, skipping')
            continue
        j = idx[-1]
        lo, hi = max(j - PAD, 0), min(j + PAD, len(sfes) - 1)
        grid = sfes[lo:hi + 1]
        print(f'W0={W0:4.1f}: {len(grid)} SFE points, '
              f'{grid[0]:.4f}-{grid[-1]:.4f}', flush=True)
        for sfe_t in grid:
            tsf = tsf_for_sfe(W0, float(sfe_t))
            pot, _, df, sfe_act = build_model(W0, tsf)
            gm = agama.GalaxyModel(potential=pot, df=df)
            Fs = []
            for s in seeds:
                agama.setRandomSeed(int(s))
                np.random.seed(int(s))
                xv, _ = gm.sample(N_PART)
                F, nit = bound_fraction_particles(xv[:, :3], xv[:, 3:])
                rows.append([W0, sfe_t, sfe_act, s, F, nit, N_PART])
                Fs.append(F)
            print(f'    SFE={sfe_act:.4f}  survival='
                  f'{np.mean(np.array(Fs) > F_MIN):.3f}', flush=True)
            np.savetxt(out, np.array(rows), fmt='%.6f', delimiter=', ',
                       header='W0, SFE_target, SFE_actual, seed, F_bound, '
                              'n_iter, N_part', comments='')
    print(f'\nWrote {out} ({len(rows)} rows). Compare with:\n'
          f'  python mc_thresholds_n1e5.py            # reads {ref}\n'
          f'  python mc_thresholds_n1e5.py {out}')


if __name__ == '__main__':
    main()
