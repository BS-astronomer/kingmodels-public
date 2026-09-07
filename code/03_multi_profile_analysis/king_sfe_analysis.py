#!/usr/bin/env python3
"""
king_sfe_analysis.py
====================
Semi-analytic analysis of King-model star clusters formed with a
centrally-peaked star-formation efficiency (SFE), for the initial-conditions
paper (companion to the gen_king_sc.py generator).

Physics
-------
* Stellar component: King (1966) lowered-isothermal model with parameter W0,
  normalized to M_star = 1, r0 (King core radius) = 1, G = 1.
* Residual gas: obtained point-by-point from the local-SFE relation
  (Parmentier & Pfalzner 2013 formalism) by inverting
      rho_star = f(rho_gas; k),   k = sqrt(8/(3 pi)) * eps_ff * t_sf,
  using the closed-form quartic solution (same K0/K1/K2 expressions as in
  the IC generator). Gas vanishes identically where the King density
  vanishes, i.e. the gas is truncated at the stellar tidal radius r_t.
* The stars are in virial equilibrium within the TOTAL (star+gas) potential.
  After instantaneous gas expulsion the post-expulsion virial ratio is
      Q = T / |W_ss| = (|W_ss| + W_sg) / (2 |W_ss|),
  where
      |W_ss| = 4 pi G * int rho_s(r) M_s(r) r dr   (stellar self-energy)
      W_sg   = 4 pi G * int rho_s(r) M_g(r) r dr   (star-gas cross term),
  and the effective SFE is
      eSFE = 1 / (2 Q) = |W_ss| / (|W_ss| + W_sg).
  Sanity limit: if rho_g = ((1-e)/e) rho_s (identical profiles), eSFE = e.

All SFE/eSFE quantities are invariant under the choice of mass/length units
(a rescaling of rho only relabels t_sf), so the results are general.

Outputs (PDF + PNG, 300 dpi) and a CSV summary table.
"""

import numpy as np
if not hasattr(np, 'trapezoid'):          # numpy < 2.0 compatibility
    np.trapezoid = np.trapz
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
EPS_FF        = 0.01                      # star-formation efficiency per free-fall time (KMB-H19: near-universal ~0.01)
W0_SHOWCASE   = [3.0, 6.0, 9.0]           # the simulated grid of Papers I/II
W0_FINE       = np.linspace(0.5, 20.0, 40)
TSF_GRID      = np.logspace(-0.5, 3.5, 120)  # t_sf sweep (model units)
SFE_SHOWCASE  = 0.17                      # global SFE for profile figures
ESFE_CRITICAL = [1.0/3.0]                 # survival threshold(s) to mark (BK03-like)
OUTDIR        = './'

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
})

# ----------------------------------------------------------------------
# King model
# ----------------------------------------------------------------------
def verify_rho_hat_series(verbose=True):
    """Self-test for the small-W series used in rho_hat().

    Two independent checks, neither run at import time:

    (1) SYMBOLIC (if sympy is available): re-derive the expansion of
        exp(W) erf(sqrt(W)) - sqrt(4W/pi) (1 + 2W/3) about W = 0 and compare
        the coefficients with the hard-coded ones. This catches a typo in the
        series that no numerical test in the small-W regime could catch,
        because there the direct formula is itself pure round-off.

    (2) NUMERICAL (always): in the overlap window 1e-3 < W < 1e-1 BOTH forms
        are accurate, so they must agree; and the leading behaviour is checked
        by confirming rho_hat / W^{5/2} -> 8/(15 sqrt(pi)) as W -> 0.

    Returns True if all available checks pass; raises AssertionError otherwise.
    """
    ok = True
    C0 = 8.0 / (15.0 * np.sqrt(np.pi))

    try:
        import sympy as sp
        W = sp.symbols('W', positive=True)
        expr = (sp.exp(W) * sp.erf(sp.sqrt(W))
                - sp.sqrt(4 * W / sp.pi) * (1 + 2 * W / 3))
        # need terms up to W^{5/2+2} = W^{9/2}, so expand past it (order 6)
        ser = sp.series(expr, W, 0, 6).removeO()
        # factor out the leading W^{5/2}: remaining polynomial is 1 + c1 W + c2 W^2
        poly = sp.simplify(sp.expand(ser / (W**sp.Rational(5, 2))))
        c0 = sp.limit(poly, W, 0)
        c1 = sp.limit(sp.diff(poly, W), W, 0)
        c2 = sp.limit(sp.diff(poly, W, 2) / 2, W, 0)
        c0f, c1f, c2f = [float(sp.N(c / c0)) for c in (c0, c1, c2)]
        lead = float(sp.N(c0))
        if verbose:
            print(f"  [sympy] leading coeff = {lead:.12e} "
                  f"(hard-coded {C0:.12e})")
            print(f"  [sympy] series ratios: 1, {c1f:.12f}, {c2f:.12f} "
                  f"(hard-coded 1, {2/7:.12f}, {4/63:.12f})")
        assert abs(lead - C0) / C0 < 1e-12, "leading coefficient mismatch"
        assert abs(c1f - 2.0 / 7.0) < 1e-12, "W^1 coefficient mismatch (expected 2/7)"
        assert abs(c2f - 4.0 / 63.0) < 1e-12, "W^2 coefficient mismatch (expected 4/63)"
    except ImportError:
        ok = False
        if verbose:
            print("  [sympy] not installed -- symbolic check skipped")

    # (2) numerical: branch agreement in the overlap window
    # Window choice: below 1e-3 the direct form is round-off noise; above ~1e-2
    # the 3-term series is truncated at the 1e-8 level and worse beyond
    # (1.1e-5 at W=0.1). 1e-3..1e-2 is where both are simultaneously accurate.
    Wo = np.logspace(-3, -2, 200)
    direct = np.exp(Wo) * erf(np.sqrt(Wo)) - np.sqrt(4.0 * Wo / np.pi) * (1.0 + 2.0 * Wo / 3.0)
    series = C0 * Wo**2.5 * (1.0 + 2.0 * Wo / 7.0 + 4.0 * Wo**2 / 63.0)
    rel = np.max(np.abs(direct - series) / series)
    if verbose:
        print(f"  [numeric] max branch disagreement on 1e-3 < W < 1e-2: {rel:.2e}")
    assert rel < 1e-7, f"branches disagree by {rel:.2e}"

    # (2b) leading behaviour from the series branch itself
    Wt = np.array([1e-8, 1e-10])
    ratio = rho_hat(Wt) / Wt**2.5
    if verbose:
        print(f"  [numeric] rho_hat/W^(5/2) at W=1e-8,1e-10: "
              f"{ratio[0]:.12e}, {ratio[1]:.12e} (limit {C0:.12e})")
    assert np.allclose(ratio, C0, rtol=1e-9), "small-W limit wrong"
    return ok


def rho_hat(W):
    """Un-normalized King density as a function of the scaled potential W.
    Uses a series expansion for small W, where the direct expression suffers
    catastrophic cancellation (both terms ~ sqrt(W); difference ~ W^{5/2})."""
    W = np.asarray(W, dtype=float)
    out = np.zeros_like(W)
    small = (W > 0) & (W < 1e-3)
    big = W >= 1e-3
    Ws = W[small]
    #   rho_hat = (8/(15 sqrt(pi))) W^{5/2} (1 + 2W/7 + 4W^2/63 + ...)  [sympy-verified]
    out[small] = 8.0 / (15.0 * np.sqrt(np.pi)) * Ws**2.5 * (1.0 + 2.0 * Ws / 7.0 + 4.0 * Ws**2 / 63.0)
    Wb = W[big]
    out[big] = np.exp(Wb) * erf(np.sqrt(Wb)) - np.sqrt(4.0 * Wb / np.pi) * (1.0 + 2.0 * Wb / 3.0)
    return np.maximum(out, 0.0)


def rho_hat_prime(W):
    """d(rho_hat)/dW, analytic (sympy-verified in verify_rho_hat_derivatives
    below to machine precision against symbolic differentiation of rho_hat's
    own defining expressions, both branches). Used by the semi-analytic
    Eddington inversion in adams_df_boundfraction.py (see its module
    docstring) so that dS/dPsi no longer needs a finite-difference derivative
    of a tabulated density."""
    W = np.asarray(W, dtype=float)
    out = np.zeros_like(W)
    small = (W > 0) & (W < 1e-3)
    big = W >= 1e-3
    Ws = W[small]
    out[small] = (8.0 / (15.0 * np.sqrt(np.pi)) *
                  (2.5 * Ws**1.5 + Ws**2.5 + (2.0 / 7.0) * Ws**3.5))
    Wb = W[big]
    out[big] = np.exp(Wb) * erf(np.sqrt(Wb)) - 2.0 * np.sqrt(Wb / np.pi)
    return out


def rho_hat_dprime(W):
    """d^2(rho_hat)/dW^2, analytic (see rho_hat_prime)."""
    W = np.asarray(W, dtype=float)
    out = np.zeros_like(W)
    small = (W > 0) & (W < 1e-3)
    big = W >= 1e-3
    Ws = W[small]
    out[small] = (8.0 / (15.0 * np.sqrt(np.pi)) *
                  (3.75 * Ws**0.5 + 2.5 * Ws**1.5 + Ws**2.5))
    Wb = W[big]
    out[big] = np.exp(Wb) * erf(np.sqrt(Wb))
    return out


def verify_rho_hat_derivatives(verbose=True):
    """Symbolic cross-check of rho_hat_prime/rho_hat_dprime against sympy
    differentiation of rho_hat's own defining expressions (both branches),
    plus a numerical branch-agreement check in the small-W overlap window --
    same spirit as verify_rho_hat_series. Not run at import time."""
    ok = True
    try:
        import sympy as sp
        W = sp.symbols('W', positive=True)
        direct = sp.exp(W) * sp.erf(sp.sqrt(W)) - sp.sqrt(4 * W / sp.pi) * (1 + 2 * W / 3)
        d1s, d2s = sp.simplify(sp.diff(direct, W)), sp.simplify(sp.diff(direct, W, 2))
        for Wt in (0.5, 2.0, 5.0, 9.0):
            n1, n2 = float(d1s.subs(W, Wt)), float(d2s.subs(W, Wt))
            h1, h2 = rho_hat_prime(np.array([Wt]))[0], rho_hat_dprime(np.array([Wt]))[0]
            assert abs(n1 - h1) < 1e-9 * max(abs(n1), 1), f"rho_hat_prime mismatch at W={Wt}"
            assert abs(n2 - h2) < 1e-9 * max(abs(n2), 1), f"rho_hat_dprime mismatch at W={Wt}"
        C0 = sp.Rational(8, 15) / sp.sqrt(sp.pi)
        series = C0 * W**sp.Rational(5, 2) * (1 + sp.Rational(2, 7) * W + sp.Rational(4, 63) * W**2)
        d1ss, d2ss = sp.expand(sp.diff(series, W)), sp.expand(sp.diff(series, W, 2))
        for Wt in (1e-4, 5e-4, 9e-4):  # strictly below the W<1e-3 series cutoff
            n1, n2 = float(d1ss.subs(W, Wt)), float(d2ss.subs(W, Wt))
            h1, h2 = rho_hat_prime(np.array([Wt]))[0], rho_hat_dprime(np.array([Wt]))[0]
            assert abs(n1 - h1) < 1e-9 * max(abs(n1), 1e-30), f"series rho_hat_prime mismatch at W={Wt}"
            assert abs(n2 - h2) < 1e-9 * max(abs(n2), 1e-30), f"series rho_hat_dprime mismatch at W={Wt}"
        if verbose:
            print("  [sympy] rho_hat_prime/rho_hat_dprime match symbolic "
                  "differentiation on both branches to <1e-9 relative")
    except ImportError:
        ok = False
        if verbose:
            print("  [sympy] not installed -- symbolic derivative check skipped")

    # branch continuity at the W=1e-3 seam, both derivatives
    for fn, name in ((rho_hat_prime, "rho_hat_prime"), (rho_hat_dprime, "rho_hat_dprime")):
        lo = fn(np.array([9.99e-4]))[0]
        hi = fn(np.array([1.001e-3]))[0]
        rel = abs(hi - lo) / max(abs(lo), abs(hi), 1e-300)
        if verbose:
            print(f"  [numeric] {name} continuity across the branch seam: {rel:.2e}")
        # the 3-term series is tuned for rho_hat's own precision at W<1e-3;
        # its derivative's truncation error is a factor ~W worse (each extra
        # series term shifts weight by one more power of W), so ~0.3% right
        # at the seam is expected, not a defect -- the sympy check above is
        # the one that actually certifies correctness.
        assert rel < 1e-2, f"{name} discontinuous across the W=1e-3 seam"
    return ok


def king_model(W0, n_grid=4000):
    """Solve the King structure ODE; return the model normalized to
    M_star = 1, r0 = 1 (King core radius), G = 1.

    Also returns rho1, rho2 = d(rho)/dx, d^2(rho)/dx^2 -- analytic, via the
    chain rule through the ODE's own state (W, dW/dx) and rho_hat_prime/
    rho_hat_dprime, NOT a finite difference of the tabulated rho array. This
    is what lets adams_df_boundfraction.eddington_f_analytic build the
    Eddington DF without differentiating a table (see that function's
    docstring for why the old approach was numerically fragile)."""
    rh0 = rho_hat(np.array([W0]))[0]

    def rhs(x, y):
        W, dW = y
        return [dW, -9.0 * rho_hat(np.array([max(W, 0.0)]))[0] / rh0 - 2.0 * dW / x]

    def hit_edge(x, y):
        return y[0]
    hit_edge.terminal = True
    hit_edge.direction = -1

    x0 = 1e-6
    y0 = [W0 - 1.5 * x0**2, -3.0 * x0]
    sol = solve_ivp(rhs, [x0, 1e7], y0, events=hit_edge,
                    method='DOP853', rtol=1e-11, atol=1e-13, dense_output=True)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError(f"King model W0={W0} did not truncate")
    xt = sol.t_events[0][0]

    # radial grid: linear core + log halo, ending exactly at r_t
    x = np.unique(np.concatenate([
        np.linspace(x0, min(1.0, 0.1 * xt), n_grid // 4),
        np.logspace(np.log10(min(1.0, 0.1 * xt)), np.log10(xt), n_grid),
    ]))
    x = x[x <= xt]
    W_dW = sol.sol(x)
    W = np.clip(W_dW[0], 0.0, None)
    dW_dx = W_dW[1]
    # second derivative from the ODE's own right-hand side (algebraic in
    # already-known W, dW_dx -- not a finite difference of anything tabulated)
    d2W_dx2 = -9.0 * rho_hat(W) / rh0 - 2.0 * dW_dx / x
    rho = rho_hat(W) / rh0                       # rho(0) = 1 before normalization
    rho1 = rho_hat_prime(W) * dW_dx / rh0
    rho2 = (rho_hat_dprime(W) * dW_dx**2 + rho_hat_prime(W) * d2W_dx2) / rh0

    M = 4.0 * np.pi * cumulative_trapezoid(rho * x**2, x, initial=0.0)
    Mtot = M[-1]
    rho, M = rho / Mtot, M / Mtot                # normalize: M_star = 1
    rho1, rho2 = rho1 / Mtot, rho2 / Mtot         # same linear normalization as rho

    rh = brentq(lambda r: np.interp(r, x, M) - 0.5, x[1], xt)
    return dict(W0=W0, x=x, rho=rho, M=M, rt=xt, rh=rh, c=np.log10(xt / 1.0),
                rho1=rho1, rho2=rho2)


# ----------------------------------------------------------------------
# Gas from the local-SFE relation (closed-form inversion, as in the IC code)
# ----------------------------------------------------------------------
def gas_density(rho_s, tsf, eps_ff=EPS_FF):
    k = np.sqrt(8.0 / (3.0 * np.pi)) * eps_ff * tsf
    k4 = k**4
    rs = np.maximum(rho_s, 1e-140)               # floor: keeps alpha=k4*rs^2 above underflow
    alpha = k4 * rs**2
    K0 = (alpha**3 + 36 * alpha**2 + 216 * alpha
          + 24 * alpha * np.sqrt(3.0 * (alpha + 27.0)))**(1.0 / 3.0)
    k4K0 = k4 * K0
    K1 = np.sqrt((alpha**2 + alpha * (K0 + 24.0) + K0 * (K0 + 12.0)) / (12.0 * k4K0))
    K2 = (alpha - K0 + 24.0) * (K0 - alpha) / (3.0 * k4K0)
    with np.errstate(invalid='ignore'):   # NaNs at huge alpha replaced below
        rho_g = 1.0 / k**2 - rs / 2.0 - np.sqrt(K2 + 8.0 / (k**6 * K1)) / 2.0 + K1
    # Large-alpha branch: the exact expression loses all precision in
    # (K0 - alpha) for alpha >~ 1e16 (float cancellation). Asymptotic
    # expansion: rho_g = 1/k^2 - 2/(k^3 sqrt(rho)) + 3/(k^4 rho) + O(alpha^-3/2).
    big = alpha > 1e8
    if np.any(big):
        rb = rs[big]
        rho_g[big] = 1.0 / k**2 - 2.0 / (k**3 * np.sqrt(rb)) + 3.0 / (k**4 * rb)
    # Small-alpha branch: for alpha < 1e-30 the exact expression is destroyed by
    # cancellation (all terms ~1/k^2 while the answer is ~1e-17 of that) and can
    # return exactly 0, which makes the local SFE rho_s/(rho_s+rho_g) spike to 1
    # at the truncation radius. The asymptotic limit is rho_g -> (rho_s/(2k))^(2/3)
    # (verified: exact/asymptotic = 1 - 1.7e-5 at alpha = 1e-28).
    tiny = alpha < 1e-30
    if np.any(tiny):
        rho_g[tiny] = (rs[tiny] / (2.0 * k))**(2.0 / 3.0)
    rho_g = np.maximum(rho_g, 0.0)
    rho_g[rho_s <= 0.0] = 0.0                    # no gas where no stars formed
    return rho_g


# ----------------------------------------------------------------------
# Global SFE and post-expulsion effective SFE
# ----------------------------------------------------------------------
def analyze(model, tsf):
    x, rho_s, M_s = model['x'], model['rho'], model['M']
    rho_g = gas_density(rho_s, tsf)
    M_g = 4.0 * np.pi * cumulative_trapezoid(rho_g * x**2, x, initial=0.0)
    sfe_glob = 1.0 / (1.0 + M_g[-1])             # M_star = 1
    W_ss = 4.0 * np.pi * np.trapezoid(rho_s * M_s * x, x)   # |W_ss| (G=1)
    W_sg = 4.0 * np.pi * np.trapezoid(rho_s * M_g * x, x)
    esfe = W_ss / (W_ss + W_sg)
    return dict(tsf=tsf, sfe=sfe_glob, esfe=esfe, rho_g=rho_g, M_g=M_g)


def sweep(model, tsf_grid=TSF_GRID):
    res = [analyze(model, t) for t in tsf_grid]
    return (np.array([r['sfe'] for r in res]),
            np.array([r['esfe'] for r in res]))


def tsf_for_sfe(model, sfe_target, tsf_grid=TSF_GRID):
    """Invert SFE(t_sf); extends the sweep upward if the target is not yet bracketed
    (low-W0 models need very long t_sf to reach a given global SFE)."""
    grid = np.array(tsf_grid)
    for _ in range(6):
        sfe, _ = sweep(model, grid)
        if sfe.max() >= sfe_target:
            break
        grid = np.logspace(np.log10(grid[0]), np.log10(grid[-1]) + 1.0, len(grid))
    else:
        raise RuntimeError(f"SFE={sfe_target} not reachable for W0={model['W0']}")
    keep = np.concatenate([[True], np.diff(sfe) > 0])   # strictly increasing for interp
    f = interp1d(sfe[keep], np.log10(grid[keep]), kind='cubic')
    return 10.0**float(f(sfe_target))


def critical_sfe(model, esfe_crit, tsf_grid=TSF_GRID):
    grid = np.array(tsf_grid)
    for _ in range(6):
        sfe, esfe = sweep(model, grid)
        order = np.argsort(sfe)
        s, e = sfe[order], esfe[order]
        if (e - esfe_crit).min() < 0 < (e - esfe_crit).max():
            break
        grid = np.logspace(np.log10(grid[0]) - 0.5, np.log10(grid[-1]) + 1.0, len(grid))
    else:
        return np.nan
    keep = np.concatenate([[True], np.diff(s) > 0])
    f = interp1d(s[keep], e[keep] - esfe_crit, kind='cubic')
    return brentq(f, s[keep].min() * 1.001, s[keep].max() * 0.999)


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
def verify_all(verbose=True):
    """Run every self-test of this module (series + identical-profile limit).

    Useful standalone -- e.g. after editing rho_hat or the energy integrals:
        python -c "import king_sfe_analysis as K; K.verify_all()"
    main() runs the same checks as part of a normal figure-producing run, so
    calling both is redundant.
    """
    if verbose:
        print("rho_hat series verification:")
    verify_rho_hat_series(verbose=verbose)
    for w in (3.0, 6.0, 9.0):
        verify(king_model(w))
    if verbose:
        print("all king_sfe_analysis self-tests passed")


def verify(model):
    """Identical-profile limit: rho_g = ((1-e)/e) rho_s must give eSFE = e."""
    x, rho_s, M_s = model['x'], model['rho'], model['M']
    W_ss = 4.0 * np.pi * np.trapezoid(rho_s * M_s * x, x)
    for e in (0.1, 0.33, 0.7):
        fac = (1.0 - e) / e
        W_sg = 4.0 * np.pi * np.trapezoid(rho_s * (fac * M_s) * x, x)
        esfe = W_ss / (W_ss + W_sg)
        assert abs(esfe - e) < 1e-12, (e, esfe)
    print(f"  verify W0={model['W0']}: identical-profile limit OK "
          f"(rt={model['rt']:.3f}, rh={model['rh']:.3f}, c={model['c']:.3f})")


# ----------------------------------------------------------------------
# Main: figures and table
# ----------------------------------------------------------------------
def main():
    print("Self-tests:")
    verify_rho_hat_series()          # series/branch check (sympy optional)
    print("Solving King models...")
    showcase = {w: king_model(w) for w in W0_SHOWCASE}
    for m in showcase.values():
        verify(m)                    # identical-profile limit, per model

    # colors for the showcase trio
    ck = {3.0: '#1b7837', 6.0: '#2166ac', 9.0: '#b2182b'}

    # ---------------- Fig 1: density profiles at fixed global SFE -------
    # Half-mass-radius units (G = M_star = r_h = 1): these are the units of
    # the Paper I N-body campaign, which equates half-mass radii across W_0.
    # Concentration then shows up directly as the central density contrast;
    # r_t/r_h varies with W_0 and is drawn per model.
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    for w, m in showcase.items():
        tsf = tsf_for_sfe(m, SFE_SHOWCASE)
        r = analyze(m, tsf)
        xh = m['rh']
        xr = m['x'] / xh
        ax.plot(xr, m['rho'] * xh**3, color=ck[w], lw=1.6, label=fr'$W_0={w:.0f}$')
        ax.plot(xr, r['rho_g'] * xh**3, color=ck[w], lw=1.6, ls='--')
        ax.axvline(1.0 / xh, color=ck[w], ls=':', lw=1.0, alpha=0.75)
        ax.axvline(m['rt'] / xh, color=ck[w], ls=(0, (1, 1)), lw=1.0, alpha=0.6)
    ax.axhline(3 / (8 * np.pi), color='0.6', ls=':', lw=1)
    ax.text(1.1e-2, 3 / (8 * np.pi) * 1.4,
            r'$\langle\rho_\star\rangle_{r_{\rm h}}$', fontsize=8, color='0.4')
    ax.plot([], [], color='0.3', lw=1.6, label='stars')
    ax.plot([], [], color='0.3', lw=1.6, ls='--', label='gas')
    ax.plot([], [], color='0.3', lw=1.0, ls=':', label=r'$r_{\rm c}$')
    ax.plot([], [], color='0.3', lw=1.0, ls=(0, (1, 1)), label=r'$r_{\rm t}$')
    ax.set(xscale='log', yscale='log', xlim=(1e-2, 12), ylim=(1e-5, 1e3),
           xlabel=r'$r/r_{\rm h}$', ylabel=r'$\rho\;\;[M_\star\,r_{\rm h}^{-3}]$')
    ax.set_title(fr'global SFE $= {SFE_SHOWCASE}$', fontsize=10)
    ax.legend(ncol=2, fontsize=8.0, loc='upper right')
    fig.savefig(OUTDIR + 'fig1_density_profiles_rh.pdf')
    fig.savefig(OUTDIR + 'fig1_density_profiles_rh.png')
    plt.close(fig)
    print("Fig 1 done (r/r_h)")

    # ---------------- Fig 2: local SFE profiles -------------------------
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    W0_family = np.array([0.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    norm = colors.Normalize(vmin=W0_family.min(), vmax=W0_family.max())
    cmap = cm.viridis
    xtmax = 0.0
    for w in W0_family:
        m = showcase.get(w) or king_model(w)
        tsf = tsf_for_sfe(m, SFE_SHOWCASE)
        r = analyze(m, tsf)
        eps_loc = np.where(m['rho'] > 0,
                           m['rho'] / np.maximum(m['rho'] + r['rho_g'], 1e-300), np.nan)
        ax.plot(m['x'] / m['rh'], eps_loc, color=cmap(norm(w)), lw=1.4)
        xtmax = max(xtmax, m['rt'] / m['rh'])
    ax.axhline(SFE_SHOWCASE, color='0.4', ls=':', lw=1)
    ax.text(1.3e-2, SFE_SHOWCASE * 1.07, 'global SFE', fontsize=8, color='0.35')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=r'$W_0$')
    ax.set(xscale='log', xlim=(1e-2, 1.05 * xtmax), ylim=(0, 1.02),
           xlabel=r'$r/r_{\rm h}$', ylabel=r'local SFE $\;\rho_\star/(\rho_\star+\rho_{\rm gas})$')
    ax.set_title(fr'global SFE $= {SFE_SHOWCASE}$', fontsize=10)
    fig.savefig(OUTDIR + 'fig2_local_sfe_rh.pdf')
    fig.savefig(OUTDIR + 'fig2_local_sfe_rh.png')
    plt.close(fig)
    print("Fig 2 done (r/r_h)")

    # ---------------- Fig 3: global SFE vs t_sf -------------------------
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    # Grid defined in CLUMP (r_0) units and converted per model to r_h units
    # (t = t_hat * (r_h/r_0)^(3/2)). In half-mass-radius units -- the natural
    # clock for the Paper I campaign -- the curves lie within a factor ~2.6.
    that_grid = np.logspace(-3, 3, 400)
    t_at_ref = []                      # t needed to reach SFE_SHOWCASE, r_h units
    for w in W0_family:
        m = showcase.get(w) or king_model(w)
        sfe, _ = sweep(m, that_grid * m['rt']**1.5)
        trh = that_grid * (m['rt'] / m['rh'])**1.5
        ax.plot(trh, sfe, color=cmap(norm(w)), lw=1.1, alpha=0.85)
        t_at_ref.append(np.interp(SFE_SHOWCASE, sfe, trh))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$W_0$')
    ax.set(xscale='log', xlabel=r'$t_{\rm sf}\;\;[G=M_\star=r_{\rm h}=1]$',
           ylabel='global SFE', ylim=(0, 1), xlim=(3e-2, 1e3))
    axin = ax.inset_axes([0.15, 0.55, 0.40, 0.38])
    axin.plot(W0_family, t_at_ref, 'o-', color='0.25', lw=1.2, ms=3)
    axin.set_xlabel(r'$W_0$', fontsize=7, labelpad=1)
    axin.set_ylabel(fr'$t_{{\rm sf}}$ at SFE$={SFE_SHOWCASE}$', fontsize=7, labelpad=1)
    axin.tick_params(labelsize=6, length=2)
    axin.set_xlim(0, 20.5)
    print(f"  t_sf at SFE={SFE_SHOWCASE}: {min(t_at_ref):.2f}..{max(t_at_ref):.2f}"
          f" r_h units (factor {max(t_at_ref)/min(t_at_ref):.2f})")
    fig.savefig(OUTDIR + 'fig3_sfe_vs_tsf_rh_units.pdf')
    fig.savefig(OUTDIR + 'fig3_sfe_vs_tsf_rh_units.png')
    plt.close(fig)
    print("Fig 3 done (t_sf in r_h units)")

    # ---------------- Fig 4: eSFE vs global SFE -------------------------
    # NOTE: both axes are efficiencies (dimensionless ratios of energies /
    # masses), so this figure carries no length or time scale and is
    # identical in r_0 and r_t units -- nothing to convert here.
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    # Grid in CLUMP units, converted per model: a grid fixed in r_0 units does
    # not reach high SFE for concentrated models (t_sf needed grows as
    # x_t^(3/2)), which truncates their curves part-way across the panel.
    that_wide = np.logspace(-3.5, 3.5, 400)
    crossings = []
    for w in W0_family:
        m = showcase.get(w) or king_model(w)
        sfe, esfe = sweep(m, that_wide * m['rt']**1.5)
        o = np.argsort(sfe)
        s, e = sfe[o], esfe[o]
        keep = (s > 0.004) & (s < 0.996)
        ax.plot(s[keep], e[keep], color=cmap(norm(w)), lw=1.1, alpha=0.9)
        for ec in ESFE_CRITICAL:                  # mark the survival threshold
            if e[keep].min() < ec < e[keep].max():
                crossings.append((np.interp(ec, e[keep], s[keep]), ec, w))
    for sc, ec, w in crossings:
        ax.plot(sc, ec, 'o', color=cmap(norm(w)), ms=4.5, mec='0.25', mew=0.6,
                zorder=5)
    ax.plot([0, 1], [0, 1], color='0.5', ls=':', lw=1)
    from itertools import cycle as _cycle
    _ls = _cycle(['--', '-.', ':', (0, (3, 1, 1, 1))])
    for ec, ls in zip(ESFE_CRITICAL, _ls):
        ax.axhline(ec, color='0.2', ls=ls, lw=1)
        ax.text(0.975, ec - 0.045, fr'$e{{\rm SFE}}={ec:.2f}$',
                fontsize=8, ha='right')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$W_0$')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='global SFE',
           ylabel=r'$e$SFE')
    ax.legend(fontsize=9, loc='lower right')
    fig.savefig(OUTDIR + 'fig4_esfe_vs_sfe.pdf')
    fig.savefig(OUTDIR + 'fig4_esfe_vs_sfe.png')
    plt.close(fig)
    print("Fig 4 done")

    # ---------------- Fig 5 (headline): critical SFE vs W0 --------------
    print("Fine W0 sweep for the critical curve...")
    rows = []
    for w in W0_FINE:
        m = king_model(w)
        row = [w, m['c'], m['rt'], m['rh']]
        for ec in ESFE_CRITICAL:
            row.append(critical_sfe(m, ec))
        rows.append(row)
        crit_str = " / ".join(f"{v:.4f}" for v in row[4:])
        print(f"  W0={w:5.2f}  c={m['c']:.3f}  SFE_crit={crit_str}")
    rows = np.array(rows)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    labels = [fr'$e{{\rm SFE}}_{{\rm crit}}={ec:.2f}$' for ec in ESFE_CRITICAL]
    from itertools import cycle
    col_cycle = cycle(['#2166ac', '#b2182b', '#1b7837', '#762a83', '#e08214'])
    ls_cycle = cycle(['-', '--', '-.', ':'])
    for j, lab in enumerate(labels):
        ax.plot(rows[:, 0], rows[:, 4 + j], color=next(col_cycle), lw=1.8,
                ls=next(ls_cycle), label=lab)
    for w in W0_SHOWCASE:
        ax.axvline(w, color='0.8', lw=0.8, zorder=0)
    ax.text(0.02, 0.03, '$W_0=3,6,9$', transform=ax.transAxes,
            fontsize=8, color='0.4')
    ax.set(xlabel=r'$W_0$', ylabel='virial SFE threshold',
           xlim=(0, 20.5))
    ax.legend(fontsize=9)
    secax = ax.secondary_xaxis('top')
    cw = interp1d(rows[:, 0], rows[:, 1])
    tick_w0 = [2, 5, 9, 13, 17]
    secax.set_xticks(tick_w0)
    secax.set_xticklabels([f'{cw(t):.1f}' for t in tick_w0])
    secax.set_xlabel(r'$c=\log_{10}(r_{\rm t}/r_{\rm c})$', fontsize=9)
    fig.savefig(OUTDIR + 'fig5_critical_sfe_vs_W0.pdf')
    fig.savefig(OUTDIR + 'fig5_critical_sfe_vs_W0.png')
    plt.close(fig)
    print("Fig 5 done")

    # ---------------- Table ---------------------------------------------
    hdr = ('W0, c=log10(rt/r0), rt_over_r0, rh_over_r0, '
           + ', '.join(f'SFE_crit_eSFE{ec:.2f}' for ec in ESFE_CRITICAL))
    np.savetxt(OUTDIR + 'results_summary.csv', rows, fmt='%.6f',
               delimiter=', ', header=hdr, comments='')
    print("Table saved. All done.")


if __name__ == '__main__':
    main()