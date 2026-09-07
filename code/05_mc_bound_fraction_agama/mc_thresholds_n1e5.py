#!/usr/bin/env python3
"""
mc_thresholds_n1e5.py
=====================
Frozen-DF Monte Carlo thresholds at N = 1e5, WITH a sampling uncertainty.

The threshold is the SFE at which the survival fraction

    p(SFE) = P(F_b > F_MIN)        F_MIN = 0.02

crosses 1/2, and the uncertainty comes from a BOOTSTRAP over realizations:
at each SFE the N_SEEDS outcomes are resampled with replacement, p(SFE) is
re-formed, the crossing is re-interpolated, and the 16th-84th percentile
spread of the resulting distribution is quoted. This propagates the binomial
sampling noise of every grid point through the interpolation in one step.

Note that half the SFE-grid gap, which is the easier quantity to compute, is a
grid-resolution statement rather than a statistical one; both are written to
the output so they can be compared. Survival probability is used rather than
mean F_b because the cells near the transition are strongly bimodal, so a mean
F_b has no meaningful centre there.

Usage:  python mc_thresholds_n1e5.py [input_csv]
Input:  mc_bound_fractions_n1e5.csv   (mc_bound_fraction_n1e5.py), or any CSV
        in the same schema -- e.g. the higher-N run written by
        mc_threshold_convergence.py, for a convergence comparison.
Output: mc_thresholds_<stem>.csv      (W0, SFE_thr, err_lo, err_hi, err_sym,
                                       grid_halfwidth, n_seeds)
"""
import sys
import numpy as np

F_MIN = 0.02
N_BOOT = 4000
RNG = np.random.default_rng(12345)


def crossing(sfes, p):
    """Last upward crossing of p = 1/2, linearly interpolated. None if absent."""
    hit = None
    for i in range(1, len(sfes)):
        if p[i - 1] < 0.5 <= p[i]:
            hit = (sfes[i - 1], sfes[i], p[i - 1], p[i])
    if hit is None:
        return None
    s0, s1, p0, p1 = hit
    return s0 + (0.5 - p0) * (s1 - s0) / max(p1 - p0, 1e-12)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'mc_bound_fractions_n1e5.csv'
    out = 'mc_thresholds_' + src.split('mc_bound_fractions_')[-1]
    d = np.loadtxt(src, delimiter=',', skiprows=1, ndmin=2)
    print(f'source: {src}   ({len(d)} realizations)')
    rows = []
    print(f'{"W0":>5} {"SFE_thr":>9} {"-err":>7} {"+err":>7} '
          f'{"grid/2":>8} {"seeds":>6}')
    for w in np.unique(d[:, 0]):
        sub = d[d[:, 0] == w]
        sfes = np.unique(np.round(sub[:, 2], 6))
        # survived[i] = boolean outcomes at SFE i
        outcomes = [sub[np.round(sub[:, 2], 6) == s, 4] > F_MIN for s in sfes]
        p = np.array([o.mean() for o in outcomes])
        sc = crossing(sfes, p)
        if sc is None:
            print(f'{w:5.1f}   no 0.5 crossing on the sampled grid')
            continue
        boots = []
        for _ in range(N_BOOT):
            pb = np.array([RNG.choice(o, size=o.size, replace=True).mean()
                           for o in outcomes])
            c = crossing(sfes, pb)
            if c is not None:
                boots.append(c)
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [16, 84])
        # grid half-width, for comparison with the old quoted error
        i = int(np.searchsorted(sfes, sc))
        gh = 0.5 * (sfes[min(i, len(sfes) - 1)] - sfes[max(i - 1, 0)])
        nseed = outcomes[0].size
        rows.append([w, sc, sc - lo, hi - sc, 0.5 * (hi - lo), gh, nseed])
        print(f'{w:5.1f} {sc:9.4f} {sc-lo:7.4f} {hi-sc:7.4f} {gh:8.4f} {nseed:6d}')
    np.savetxt(out, np.array(rows), fmt='%.6f', delimiter=', ',
               header='W0, SFE_thr, err_lo, err_hi, err_sym, grid_halfwidth, '
                      'n_seeds', comments='')
    print(f'\nWrote {out}   (bootstrap over realizations, {N_BOOT} resamples)')


if __name__ == '__main__':
    main()
