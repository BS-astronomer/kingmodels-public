#!/usr/bin/env python3
"""
mc_refine_threshold.py
======================
Adds finely spaced SFE points around each already-located transition, so that
the threshold's uncertainty is set by sampling rather than by grid spacing.

This is needed because the stochastic zone narrows as N grows. At N = 1e4 the
survival fraction climbs from 0 to 1 over several SFE grid steps, so a coarse
grid still resolves the crossing; at N = 1e5 the same transition can complete
inside a single 0.005-wide step, in which case a bootstrap over realizations
would report an error much smaller than the interval the crossing is actually
known to lie in. Refining until several grid points fall inside the transition
removes that mismatch.

Procedure: read the existing run, locate the bracketing pair where the
survival fraction crosses 1/2, and resample that bracket with N_SUB interior
points at the same seed count. Rows are appended to the input CSV, so
mc_thresholds_n1e5.py can simply be re-run afterwards.

Usage:  python mc_refine_threshold.py [csv] [N_part]
Output: appends to the input CSV (default mc_bound_fractions_n1e5.csv)
"""
import os
import sys
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
sys.path.append('../01_king_sfe_analysis')
import agama
from mc_bound_fraction_agama import (build_model, bound_fraction_particles,
                                     tsf_for_sfe)

F_MIN = 0.02
N_SUB = 7          # interior points inserted across the bracketing interval
N_SEEDS = 150


def bracket(sfes, p):
    """(lo, hi) SFE pair straddling the last upward crossing of p = 1/2."""
    hit = None
    for i in range(1, len(sfes)):
        if p[i - 1] < 0.5 <= p[i]:
            hit = (sfes[i - 1], sfes[i])
    return hit


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else 'mc_bound_fractions_n1e5.csv'
    N_PART = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    d = np.loadtxt(csv, delimiter=',', skiprows=1, ndmin=2)
    seeds = np.random.default_rng(4242).integers(1_000_000, 99_999_999,
                                                 size=N_SEEDS)
    added = []
    for W0 in np.unique(d[:, 0]):
        sub = d[d[:, 0] == W0]
        sfes = np.unique(np.round(sub[:, 2], 6))
        p = np.array([np.mean(sub[np.round(sub[:, 2], 6) == s, 4] > F_MIN)
                      for s in sfes])
        br = bracket(sfes, p)
        if br is None:
            print(f'W0={W0}: no crossing on the sampled grid, skipping')
            continue
        lo, hi = br
        grid = np.linspace(lo, hi, N_SUB + 2)[1:-1]
        print(f'W0={W0:4.1f}: refining ({lo:.4f}, {hi:.4f}) with {len(grid)} '
              f'points, spacing {grid[1]-grid[0]:.5f}', flush=True)
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
                added.append([W0, sfe_t, sfe_act, s, F, nit, N_PART])
                Fs.append(F)
            print(f'    SFE={sfe_act:.4f}  survival={np.mean(np.array(Fs)>F_MIN):.3f}',
                  flush=True)
    if not added:
        print('nothing to add')
        return
    out = np.vstack([d, np.array(added)])
    out = out[np.lexsort((out[:, 2], out[:, 0]))]
    np.savetxt(csv, out, fmt='%.6f', delimiter=', ',
               header='W0, SFE_target, SFE_actual, seed, F_bound, n_iter, N_part',
               comments='')
    print(f'\nAppended {len(added)} realizations to {csv} ({len(out)} rows). '
          f'Re-run mc_thresholds_n1e5.py.')


if __name__ == '__main__':
    main()
