#!/usr/bin/env python3
"""
mc_diagnose_w3.py -- pinpoint the W0=3 anomaly (F_b=0 at SFE=0.31).

Runs the model construction of mc_bound_fraction_agama.py at W0=3 for
SFE targets around the anomaly, with BOTH multipole settings (old fixed
rmax=1e4 vs new adaptive rmax), and prints per-configuration:
  - tsf from the inversion (must be smooth & monotone in SFE)
  - gas potential at r = 0.1, 1, 3 from the multipole vs the direct
    integral of the gas density (must agree; discrepancy = fit artifact)
  - single-pass unbound fraction of one sampled realization
    (fraction with v^2/2 > Psi_star(r) before any iteration; must be
    smooth in SFE -- a jump at 0.31 pins the artifact)

Usage: python mc_diagnose_w3.py     (needs agama; ~2-3 minutes)
"""
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import agama
from mc_bound_fraction_agama import (gas_density, tsf_for_sfe, RMIN, RMAX,
                                     bound_fraction_particles)
from scipy.interpolate import interp1d

W0 = 3.0
TARGETS = [0.29, 0.30, 0.31, 0.32, 0.33, 0.35]
SEED = 41607809
N = 10_000


def direct_gas_potential(r_grid, rho_g, r_eval):
    """Psi_gas by direct quadrature (G=1)."""
    from scipy.integrate import cumulative_trapezoid
    M = 4*np.pi*cumulative_trapezoid(rho_g*r_grid**2, r_grid, initial=0.0)
    outer = 4*np.pi*cumulative_trapezoid((rho_g*r_grid)[::-1], r_grid[::-1],
                                         initial=0.0)[::-1]*(-1.0)
    psi = M/r_grid + outer
    return np.interp(r_eval, r_grid, psi)


def build(tsf, adaptive):
    king = agama.Potential(type='king', W0=W0, scaleRadius=1.0, mass=1.0)
    n = 3000
    r = np.logspace(np.log10(RMIN), np.log10(RMAX), n)
    xyz = np.zeros((n, 3)); xyz[:, 0] = r
    rho_s = king.density(xyz)
    rho_g = gas_density(rho_s, tsf)
    gi = interp1d(r, rho_g, kind='linear', bounds_error=False,
                  fill_value=(rho_g[0], 0.0))
    def gd(xyz):
        xyz = np.asarray(xyz)
        rr = np.linalg.norm(xyz, axis=1) if xyz.ndim == 2 else np.linalg.norm(xyz)
        return gi(rr)
    if adaptive:
        r_gas = r[rho_g > 0]
        rmax_fit = min(RMAX, 2.0*r_gas[-1]) if len(r_gas) else RMAX
        pg = agama.Potential(type='multipole', symmetry='spherical',
                             density=gd, rmin=RMIN, rmax=rmax_fit, gridSizeR=60)
    else:
        pg = agama.Potential(type='multipole', symmetry='spherical',
                             density=gd, rmin=RMIN, rmax=RMAX)
    return king, pg, r, rho_s, rho_g


for adaptive in (False, True):
    print(f"\n=== multipole: {'ADAPTIVE rmax (fixed script)' if adaptive else 'fixed rmax=1e4 (old script)'} ===")
    print(f"{'SFE':>6} {'tsf':>10} {'dPsi/Psi(0.1)':>14} {'(1)':>9} {'(3)':>9} "
          f"{'1-pass unbound':>15} {'F_b':>7}")
    for s in TARGETS:
        tsf = tsf_for_sfe(W0, s)
        king, pg, r, rho_s, rho_g = build(tsf, adaptive)
        # multipole vs direct gas potential
        pts = np.array([[0.1, 0, 0], [1, 0, 0], [3, 0, 0]])
        psi_mp = -pg.potential(pts)
        psi_dir = direct_gas_potential(r, rho_g, pts[:, 0])
        rel = (psi_mp - psi_dir)/psi_dir
        # sample and single-pass + full criterion
        pot_tot = agama.Potential(pg, king)
        df = agama.DistributionFunction(type='QuasiSpherical',
                                        potential=pot_tot, density=king)
        gm = agama.GalaxyModel(potential=pot_tot, df=df)
        agama.setRandomSeed(SEED); np.random.seed(SEED)
        xv, _ = gm.sample(N)
        rr = np.linalg.norm(xv[:, :3], axis=1)
        v2 = np.einsum('ij,ij->i', xv[:, 3:], xv[:, 3:])
        psi_star = -king.potential(xv[:, :3])
        onepass = np.mean(0.5*v2 >= psi_star)
        F, nit = bound_fraction_particles(xv[:, :3], xv[:, 3:])
        print(f"{s:6.2f} {tsf:10.4f} {rel[0]:14.2e} {rel[1]:9.2e} {rel[2]:9.2e} "
              f"{onepass:15.4f} {F:7.4f}")
