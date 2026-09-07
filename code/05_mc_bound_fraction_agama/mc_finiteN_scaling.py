#!/usr/bin/env python3
"""
mc_finiteN_scaling.py
=====================
How the frozen-DF survival probability depends on particle number.

Near the transition the outcome is bimodal, so whether a given realization
survives is decided by shot noise. Two particle counts would show that the
outcome is stochastic but would not constrain how the survival probability
scales; this runs a configuration at six values of N spanning more than two
decades, so the trend is measured rather than inferred.

The SFE for each W0 is chosen to sit inside the stochastic zone, i.e. where
p(N) sweeps from near 1 at small N to 0 by N = 1e5. Away from that zone the
outcome is deterministic at every N and the scan is uninformative.

For each N the survival probability p = P(F_b > F_MIN) is measured over
N_SEEDS realizations, with a binomial standard error sqrt(p(1-p)/N_SEEDS).

Usage:  python mc_finiteN_scaling.py [W0 SFE]     (default: 3.0 0.30)
Output: appends to mc_finiteN_scaling.csv
        (W0, N_part, SFE, n_seeds, p_survive, p_err, mean_Fb, std_Fb)
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

N_LIST = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
N_SEEDS = 150
F_MIN = 0.02
OUT = 'mc_finiteN_scaling.csv'
HEADER = ('W0, N_part, SFE, n_seeds, p_survive, p_err, mean_Fb, std_Fb')


def run(W0, SFE):
    tsf = tsf_for_sfe(W0, SFE)
    pot_total, king_pot, df, sfe_act = build_model(W0, tsf)
    gm = agama.GalaxyModel(potential=pot_total, df=df)
    seeds = np.random.default_rng(777).integers(1_000_000, 99_999_999,
                                                size=N_SEEDS)
    rows = []
    print(f'W0={W0}, SFE_actual={sfe_act:.4f}, {N_SEEDS} realizations per N\n')
    print(f'{"N":>8} {"p_survive":>11} {"binom err":>10} {"<F_b>":>8} {"sd":>8}')
    for N in N_LIST:
        Fs = []
        for s in seeds:
            agama.setRandomSeed(int(s))
            np.random.seed(int(s))
            xv, _ = gm.sample(N)
            F, _ = bound_fraction_particles(xv[:, :3], xv[:, 3:])
            Fs.append(F)
        Fs = np.array(Fs)
        p = float(np.mean(Fs > F_MIN))
        err = float(np.sqrt(max(p * (1 - p), 0.0) / N_SEEDS))
        rows.append([W0, N, sfe_act, N_SEEDS, p, err, Fs.mean(), Fs.std()])
        print(f'{N:8d} {p:11.3f} {err:10.3f} {Fs.mean():8.4f} {Fs.std():8.4f}',
              flush=True)
    return np.array(rows)


def main():
    W0 = float(sys.argv[1]) if len(sys.argv) > 2 else 3.0
    SFE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    new = run(W0, SFE)
    if os.path.exists(OUT):
        old = np.loadtxt(OUT, delimiter=',', skiprows=1, ndmin=2)
        old = old[np.abs(old[:, 0] - W0) > 1e-9]        # replace this W0
        new = np.vstack([old, new]) if old.size else new
    new = new[np.lexsort((new[:, 1], new[:, 0]))]
    np.savetxt(OUT, new, fmt='%.6g', delimiter=', ', header=HEADER, comments='')
    print(f'\nWrote {OUT} ({len(new)} rows)')


if __name__ == '__main__':
    main()
