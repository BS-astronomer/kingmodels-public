import agama
import numpy as np
import argparse
import sys
import os
import re
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Astrophysical Constants
G      = 6.67430E-11
Msol   = 1.98841E+30
Rsol   = 6.957E+08
AU     = 1.49597870700E+11
pc     = 3.08567758149E+16
Year   = 31556925.1
km     = 1.0E+03

parser = argparse.ArgumentParser()
parser.add_argument('--N', default=10455, help='TOTAL number of stars, held constant across models [integer]', type=int)
parser.add_argument('--sfe', default=0.17, help='Target SFE (e.g., 0.15, 0.17, 0.20, 0.25, etc.)', type=float)
parser.add_argument('--rnd_ps', default=0, choices=range(10), help='random seed [integer]', type=int)
parser.add_argument('--rnd_imf', default=0, choices=range(10), help='random seed [integer]', type=int)
parser.add_argument('--W0', default=7.0, help='King model W0 [float]', type=float)
parser.add_argument('--rc', default=None, help='King model core radius rc [pc]; used only if neither --lam nor --rh is given [float]', type=float)
parser.add_argument('--rh', default=None, help='half-mass radius [pc]; used only if --lam is not given (default 1.0 if nothing is set) [float]', type=float)
parser.add_argument('--fb', default=0.0, help='binary fraction (fraction of SYSTEMS that are binaries) [float]', type=float)
parser.add_argument('--cm', nargs='+', default=[-8178.0, 0, 0, 0, 234.73697522559405, 0], help='position and velocity of the center of mass in galactic frame [list of 6 float values]', type=float)
parser.add_argument('--cm_file', default='', help='filename containing the position and velocity of the center of mass in galactic frame', type=str)
parser.add_argument('--lam', default=0.0, help='lambda = r_h/r_J [float]', type=float)
parser.add_argument('--eff', default=0.01, help='SFE per free-fall time eps_ff; '
                    'NOTE: t_SF and eps_ff are exactly degenerate (only their '
                    'product enters), so this choice does not change the ICs '
                    'for a given target SFE [float]', type=float)
parser.add_argument('--tsf', default=None, help='override: star-formation duration '
                    'in model units; if given, --sfe is only used for the '
                    'consistency check (must match the eps_ff convention of '
                    'whatever produced this value!) [float]', type=float)

args = parser.parse_args()
print("The following args are used:\n", args)

# Extract CM coordinates
if args.cm_file:
    try:
        orb = np.genfromtxt(args.cm_file, max_rows=1, unpack=True)
        X_sc = orb[:3]
        V_sc = orb[3:]
        print('Position and velocity of the cluster center of mass were obtained from file: %s' % (args.cm_file))
    except Exception as e:
        print(f"An unexpected error occurred reading cm_file: {e}")
        sys.exit(1)
else:
    print('Position and velocity of the cluster center of mass:')
    print(args.cm)
    X_sc = np.array(args.cm)[:3]  # pc
    V_sc = np.array(args.cm)[3:]  # km/s

# Size precedence: lam > rh > rc > (default rh = 1 pc)
if args.lam > 0.0 and (args.rh is not None or args.rc is not None):
    print('Warning! --lam is set: half-mass radius will be derived from lambda; --rh/--rc are ignored.')
elif args.rh is not None and args.rc is not None:
    print('Warning! Both --rh and --rc are set: using --rh; --rc is ignored.')


def r_tid(M, R, V, g=G):
    # NOTE: assumes a circular orbit (Omega = V/R) and uses the STELLAR mass only.
    # For eccentric --cm inputs the instantaneous V/R is not the circular frequency.
    beta = 1.37
    ome = 1.0 * V / R
    return (g * M / ((4 - beta**2) * ome**2))**(1. / 3.)


rnd_seed = np.array([41607809, 20601384, 56226667, 17886841, 79145889,
                     82447603, 99438324, 94536542,  9673519, 12345678])
rnd_ps  = args.rnd_ps
rnd_imf = args.rnd_imf
rnd_seed_ps  = rnd_seed[args.rnd_ps]
rnd_seed_imf = rnd_seed[args.rnd_imf]

Ms    = 1.0
a     = 1.0  # Agama dimensionless scale
W0    = args.W0
metallicity = 0.02
# Grid bounds are ADAPTIVE per model (see king_grid_bounds below):
#   rmin = 1e-3 * r_c (King core radius = agama scaleRadius)
#   rmax = 1.1  * r_t (King truncation radius, detected from the potential)
# The constants below are only the fallback probe range for r_t detection.
rmin_fallback = 1e-3
rmax_probe    = 1e6


def king_rt(king_pot, r_probe_max=rmax_probe):
    """Truncation radius of an agama King potential, found as the outermost
    radius with nonzero density (log-grid probe + bisection; unit-safe)."""
    import numpy as _np
    rp = _np.logspace(-3, _np.log10(r_probe_max), 400)
    xyz = _np.zeros((rp.size, 3)); xyz[:, 0] = rp
    dens = king_pot.density(xyz)
    pos = _np.where(dens > 0)[0]
    if len(pos) == 0 or pos[-1] == rp.size - 1:
        raise RuntimeError('king_rt: could not bracket the truncation radius')
    lo, hi = rp[pos[-1]], rp[pos[-1] + 1]
    for _ in range(60):
        mid = _np.sqrt(lo * hi)
        if king_pot.density([[mid, 0, 0]])[0] > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def king_grid_bounds(king_pot, a):
    """Adaptive (rmin, rmax) for the gas grid and multipole solver:
    rmin = 1e-3 * core radius, rmax = 1.1 * truncation radius."""
    return 1e-3 * a, 1.1 * king_rt(king_pot)

# =========================================================================
# PARTICLE BOOKKEEPING (FIX: total STAR count is exactly N_tot for any fb)
#
# fb is the fraction of SYSTEMS that are binaries. With N_sys sampled
# centers of mass and N_bin of them split into two components, the total
# number of stars is N_sys + N_bin. We choose integers so that this sum
# equals N_tot exactly, keeping total mass / mean density comparable
# across models with different fb.
# =========================================================================
fb    = args.fb
N_tot = args.N
if fb > 0.0:
    N_bin   = int(np.floor(N_tot * fb / (1.0 + fb) + 0.5))  # explicit round-half-up
    N_stars = N_tot - N_bin          # number of SYSTEMS (sampled phase-space points)
    fb_eff  = N_bin / N_stars        # effective binary fraction actually realized
    print(f"Requested fb={fb:.3f}: using {N_stars} systems, {N_bin} binaries "
          f"-> {N_stars + N_bin} stars total (= N_tot), effective fb={fb_eff:.4f}")
else:
    N_bin   = 0
    N_stars = N_tot
    fb_eff  = 0.0

# =========================================================================
# DYNAMIC t_sf LOADING & NaN FILTERING
# =========================================================================
# SFE -> t_sf inversion is done INTERNALLY (no lookup tables): the gas mass
# is integrated on the model grid and brentq solves SFE(t_sf) = target with
# THIS script's eps_ff. This removes the failure mode of lookup tables
# generated under a different eps_ff convention (t_sf scales exactly as
# 1/eps_ff at fixed SFE; tables produced with eps_1 convert to eps_2 via
# t_sf * eps_1/eps_2). The legacy target_sfe_vs_tsf_W0_*.txt files are no
# longer read.
def _gas_profile(rho_stars, tsf_, e_ff_):
    """PP13 residual-gas density for a stellar density array (same formula as
    in gen_king_sc, incl. the large-alpha asymptotic branch); zero where no
    stars formed."""
    rs = np.maximum(rho_stars, 1e-10)
    k_ = (8. / 3. / np.pi)**0.5 * e_ff_ * tsf_
    k4_ = k_**4
    al = k4_ * rs**2
    K0_ = (al**3 + 36*al**2 + 216*al + 24*al*(3.*(al+27.))**0.5)**(1./3.)
    k4K0_ = k4_ * K0_
    K1_ = ((al**2 + al*(K0_+24.) + K0_*(K0_+12.)) / 12. / k4K0_)**0.5
    K2_ = (al - K0_ + 24) * (K0_ - al) / 3. / k4K0_
    with np.errstate(invalid='ignore'):
        rg = 1./k_**2 - rs/2 - (K2_ + 8./k_**6/K1_)**0.5/2 + K1_
    big = al > 1e8
    if np.any(big):
        rb = rs[big]
        rg[big] = 1./k_**2 - 2./(k_**3*np.sqrt(rb)) + 3./(k_**4*rb)
    rg = np.maximum(rg, 0.0)
    rg[rho_stars <= 0.0] = 0.0
    return rg


def _invert_sfe_to_tsf(target_sfe, W0_, a_, Ms_, e_ff_):
    from scipy.optimize import brentq
    kp = agama.Potential(type='king', W0=W0_, scaleRadius=a_, mass=Ms_)
    rmin_, rmax_ = king_grid_bounds(kp, a_)
    rr = np.logspace(np.log10(rmin_), np.log10(rmax_), 3000)
    xyz_ = np.zeros((rr.size, 3)); xyz_[:, 0] = rr
    rho_s_ = kp.density(xyz_)

    def sfe_of_logt(lt):
        rho_g_ = _gas_profile(rho_s_, 10.0**lt, e_ff_)
        Mg_ = np.trapezoid(4.0*np.pi*rho_g_*rr**2, rr)
        return Ms_/(Ms_+Mg_) - target_sfe
    lt = brentq(sfe_of_logt, -4.0, 9.0, xtol=1e-8)
    return 10.0**lt

if args.tsf is not None:
    tsf_matched, sfe_matched = args.tsf, args.sfe
    print(f"t_sf OVERRIDE: using t_sf={tsf_matched:.6f}; the built model's actual "
          f"SFE will be checked against --sfe={args.sfe} below.")
else:
    tsf_matched = _invert_sfe_to_tsf(args.sfe, W0, a, Ms, args.eff)
    sfe_matched = args.sfe
print(f"Parameters: W0={W0}, target SFE={sfe_matched:.4f} -> t_sf={tsf_matched:.6f} "
      f"(eps_ff={args.eff})")

# =========================================================================
# UTILITIES
# =========================================================================
FLOAT_RE = r'([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)'  # FIX: handles scientific notation


def get_king_rh_over_rc(W0_val):
    """Ratio rh/rc for a given King W0 (rc = Agama's scaleRadius).

    FIX: uses brentq on enclosedMass(r) with scalar radii instead of a cubic
    interpolation of enclosedMass evaluated on an (N,3) coordinate array.
    The old version (a) passed xyz arrays where radii are expected and
    (b) fed a plateaued (non-monotonic) m_enc curve to a cubic interpolator,
    since the King model truncates well inside the r=1000 grid.
    """
    temp_king = agama.Potential(type='king', W0=W0_val, scaleRadius=1.0, mass=1.0)
    mtot = float(temp_king.enclosedMass(1e3))
    f = lambda r: float(temp_king.enclosedMass(r)) - 0.5 * mtot
    return brentq(f, 1e-3, 1e3, xtol=1e-10)


def extract_value(label_pattern, text):
    """FIX: raises on failure instead of silently returning None."""
    match = re.search(label_pattern % FLOAT_RE, text)
    if match is None:
        print(f"CRITICAL ERROR: could not parse '{label_pattern}' from mcluster output.")
        sys.exit(1)
    return float(match.group(1).replace('D', 'E').replace('d', 'e'))


def gen_imf_K2001(m, ml, mu):
    np.random.seed(rnd_seed_imf)
    AN = 1.0
    N = len(m)
    m1, m2 = 0.5, 1.0
    a1, a2, a3 = 1.3, 2.3, 2.3
    A1, A2, A3 = 2.0 / AN, 1.0 / AN, 1.0 / AN
    tmp_N  = (A1 / (1.0 - a1)) * ((m1**(1.0 - a1)) - (ml**(1.0 - a1)))
    tmp_N += (A2 / (1.0 - a2)) * ((m2**(1.0 - a2)) - (m1**(1.0 - a2)))
    tmp_N += (A3 / (1.0 - a3)) * ((mu**(1.0 - a3)) - (m2**(1.0 - a3)))
    AN = tmp_N
    A1, A2, A3 = 2.0 / AN, 1.0 / AN, 1.0 / AN
    g1 = (A1 / (1 - a1)) * (m1**(1 - a1) - ml**(1 - a1))
    g2 = g1 + (A2 / (1 - a2)) * ((m2**(1.0 - a2)) - (m1**(1.0 - a2)))
    tmp = np.random.uniform(size=N)

    cut1 = np.where(tmp <= g1)
    m[cut1] = (ml**(1 - a1) + (1 - a1) * tmp[cut1] / A1)**(1. / (1 - a1))
    cut2 = np.where((tmp > g1) & (tmp <= g2))
    m[cut2] = (m1**(1 - a2) + (1 - a2) * (tmp[cut2] - g1) / A2)**(1. / (1 - a2))
    cut3 = np.where(tmp > g2)
    m[cut3] = (m2**(1 - a3) + (1 - a3) * (tmp[cut3] - g2) / A3)**(1. / (1 - a3))


def gen_binaries(params, params_val):
    with open('mcluster.ini', 'w') as mcinit:
        mcinit.write('[Mcluster]\n')
        for p, pv in zip(params, params_val):
            mcinit.write(p + pv + '\n')

    # FIX: check the mcluster exit code and fail fast instead of continuing
    # with mnorm/rnorm possibly undefined or None.
    rc = os.system('time ./mcluster.exe 1> mcluster.out 2> mcluster.err')
    if rc != 0:
        print(f"CRITICAL ERROR: mcluster.exe exited with code {rc}. See mcluster.err.")
        sys.exit(1)

    with open('mcluster.out', 'r') as outfile:
        output = outfile.read()
    mnorm = extract_value(r'Total mass %s', output)
    rnorm = extract_value(r'rvir = %s', output)
    print('Mcluster: Total mass is %.4f [Msol]' % mnorm)
    print('Mcluster: rvir is %.10E [pc]' % rnorm)

    mnorm *= Msol
    rnorm *= pc
    vnorm = np.sqrt(G * mnorm / rnorm)

    dat10f = 'dat.10'
    if not os.path.isfile(dat10f):
        print("CRITICAL ERROR: dat.10 was not generated by mcluster.")
        sys.exit(1)

    dat10 = np.genfromtxt(dat10f, unpack=True)
    dat10[0]   *= mnorm / Msol
    dat10[1:4] *= rnorm / pc
    dat10[4:7] *= vnorm / km

    # FIX: verify mcluster produced exactly the expected number of stars,
    # otherwise masses and positions would be silently misaligned later.
    n_from_mc = dat10.shape[1]
    n_expected = N_stars + N_bin
    if n_from_mc != n_expected:
        print(f"CRITICAL ERROR: mcluster returned {n_from_mc} stars but "
              f"{n_expected} were expected ({N_stars} systems, {N_bin} binaries). "
              f"Check how your mcluster build rounds 'fracb * n'.")
        sys.exit(1)

    binaries = dat10.T[:N_bin * 2]
    m  = dat10[0]
    b1 = binaries[np.arange(N_bin) * 2].T[1:7].T
    b2 = binaries[np.arange(N_bin) * 2 + 1].T[1:7].T
    m1 = binaries[np.arange(N_bin) * 2].T[0].T
    m2 = binaries[np.arange(N_bin) * 2 + 1].T[0].T

    cm = (m1[:, None] * b1 + m2[:, None] * b2) / ((m1 + m2)[:, None])
    b1 -= cm
    b2 -= cm
    return [m, m1, b1, m2, b2]


# =========================================================================
# MAIN CLUSTER GENERATION
# =========================================================================
def gen_king_sc(tsf, sfe):
    np.random.seed(rnd_seed_ps)
    agama.setRandomSeed(int(rnd_seed_ps))

    king_pot = agama.Potential(type='king', W0=W0, scaleRadius=a, mass=Ms)

    rmin, rmax = king_grid_bounds(king_pot, a)
    npts = 2000
    r_grid = np.logspace(np.log10(rmin), np.log10(rmax), npts)
    xyz_grid = np.zeros((npts, 3))
    xyz_grid[:, 0] = r_grid

    rho_stars_grid = king_pot.density(xyz_grid)
    rho_s_clamped = np.maximum(rho_stars_grid, 1e-10)

    e_ff = args.eff
    k = (8. / 3. / np.pi)**0.5 * e_ff * tsf
    k4 = k**4
    alpha = k4 * rho_s_clamped**2

    K_0 = (alpha**3 + 36 * alpha**2 + 216 * alpha + 24 * alpha * (3. * (alpha + 27.))**0.5)**(1. / 3.)
    k4K0 = k4 * K_0
    K_1 = ((alpha**2 + alpha * (K_0 + 24.) + K_0 * (K_0 + 12.)) / 12. / k4K0)**0.5
    K_2 = (alpha - K_0 + 24) * (K_0 - alpha) / 3. / k4K0

    with np.errstate(invalid='ignore'):   # NaNs at huge alpha replaced below
        rho_gas_grid = 1. / k**2 - rho_s_clamped / 2 - (K_2 + 8. / k**6 / K_1)**0.5 / 2 + K_1
    # Large-alpha branch: the exact expression loses precision in (K_0 - alpha)
    # for alpha >~ 1e16; use the asymptotic expansion above alpha = 1e8
    # (identical to king_sfe_analysis.gas_density).
    big = alpha > 1e8
    if np.any(big):
        rb = rho_s_clamped[big]
        rho_gas_grid[big] = 1. / k**2 - 2. / (k**3 * np.sqrt(rb)) + 3. / (k**4 * rb)
    # FIX: numerical cancellation in the outskirts can leave tiny negative
    # densities; clamp to zero before handing to the multipole solver.
    rho_gas_grid = np.maximum(rho_gas_grid, 0.0)
    # FIX: the rho_s clamp (1e-10 floor) leaks a spurious gas floor
    # (~2e-7, since rho_gas ~ rho_s^(2/3)) beyond the King tidal radius,
    # integrating to ~1 model unit of fake gas out to rmax. The model has
    # NO gas where no stars formed, so enforce rho_gas = 0 exactly where
    # the true (unclamped) King density vanishes.
    rho_gas_grid[rho_stars_grid <= 0.0] = 0.0

    gas_density_interp = interp1d(
        r_grid, rho_gas_grid,
        kind='linear',
        bounds_error=False,
        fill_value=(rho_gas_grid[0], 0.0)
    )

    def fast_gas_density(xyz):
        xyz = np.asarray(xyz)
        r = np.linalg.norm(xyz, axis=1) if xyz.ndim == 2 else np.linalg.norm(xyz)
        return gas_density_interp(r)

    # SELF-CHECK: the actual global SFE of the built model must match the
    # requested target; any eps_ff/table/unit inconsistency is caught here.
    M_gas_actual = np.trapezoid(4.0*np.pi*rho_gas_grid*r_grid**2, r_grid)
    sfe_actual = Ms/(Ms + M_gas_actual)
    print(f"SELF-CHECK: actual SFE of built model = {sfe_actual:.5f} "
          f"(target {sfe:.5f})")
    if abs(sfe_actual - sfe) > 5e-3:
        raise RuntimeError(
            f"SFE mismatch: built {sfe_actual:.5f} vs target {sfe:.5f}. "
            f"If you passed --tsf from an external table, its eps_ff "
            f"convention differs from --eff={args.eff} "
            f"(t_sf scales as 1/eps_ff; rescale by eps_table/eps_here).")

    pot_gas = agama.Potential(type='multipole', symmetry='spherical', density=fast_gas_density, rmin=rmin, rmax=rmax)
    pot_total = agama.Potential(pot_gas, king_pot)
    df_stars = agama.DistributionFunction(type='QuasiSpherical', potential=pot_total, density=king_pot)
    galmod = agama.GalaxyModel(potential=pot_total, df=df_stars)

    # Sample system centers of mass (N_stars systems; N_bin of them will be split)
    particles = galmod.sample(N_stars)
    pa = particles[0]

    m_low, m_up = 0.08, 150.0

    if fb > 0.0:
        params = ['n = ', 'fracb_reference = ', 'fracb = ', 'mfunc = ', 'mlow = ', 'mup = ',
                  'pairing = ', 'adis = ', 'amin = ', 'amax = ', 'zini = ', 'seedmc = ', 'outputf = ']
        # FIX: pass the effective binary fraction so mcluster produces exactly
        # N_bin binaries out of N_stars systems (verified by the count check).
        params_val = ['%d' % N_stars, '%.6f' % fb_eff, '%.6f' % fb_eff, '1',
                      '%.3f' % m_low, '%.3f' % m_up, '3', '4',
                      '%.3f' % (1 * AU / Rsol), '%.3f' % (1000 * AU / Rsol),
                      '%.4f' % metallicity, '%d' % rnd_seed_imf, '2']
        m, m1, b1, m2, b2 = gen_binaries(params, params_val)
    else:
        m = np.ones(len(pa), dtype=float)
        gen_imf_K2001(m, m_low, m_up)

    m_norm = np.sum(m) * Msol
    rh_over_rc = get_king_rh_over_rc(W0)

    # Physical scale, precedence: lam > rh > rc > default (rh = 1 pc).
    # --rc sets the King scale radius; it must be applied here, not just parsed.
    if args.lam > 0.0:
        Rgal = np.sqrt(np.sum(X_sc**2)) * pc
        Vgal = np.sqrt(np.sum(V_sc**2)) * km
        r_J = r_tid(m_norm, Rgal, Vgal) / pc
        rh_phys = args.lam * r_J
        print(f"Size from lambda={args.lam}: r_J={r_J:.4f} pc -> rh={rh_phys:.4f} pc")
    elif args.rh is not None:
        rh_phys = args.rh
        print(f"Size from --rh: rh={rh_phys:.4f} pc")
    elif args.rc is not None:
        rh_phys = args.rc * rh_over_rc
        print(f"Size from --rc: rc={args.rc:.4f} pc -> rh={rh_phys:.4f} pc (rh/rc={rh_over_rc:.4f} for W0={W0})")
    else:
        rh_phys = 1.0
        print("Size: no --lam/--rh/--rc given, using default rh = 1.0 pc")

    rc_phys = rh_phys / rh_over_rc
    r_norm = rc_phys * pc
    v_norm = np.sqrt(G * m_norm / r_norm)

    # Scale coordinates
    pa[:, :3] *= r_norm / pc
    pa[:, 3:] *= v_norm / km

    # Process Binaries
    if fb > 0.0:
        bcm = pa[:N_bin]
        singles = pa[N_bin:]
        b1 += bcm
        b2 += bcm

        newpa = np.zeros(((N_stars + N_bin), 6))
        print("Generated %d single stars, %d binaries, %d stars in total"
              % (N_stars - N_bin, N_bin, len(newpa)))

        newpa[np.arange(N_bin) * 2]     += b1
        newpa[np.arange(N_bin) * 2 + 1] += b2
        newpa[N_bin * 2:] += singles
        pa = newpa

    n_out = len(pa)
    if n_out != N_tot:
        # Should be impossible after the bookkeeping fix, but never write a
        # file whose header disagrees with its contents.
        print(f"CRITICAL ERROR: output has {n_out} stars but N_tot={N_tot}.")
        sys.exit(1)

    # Output formatting
    if args.lam > 0.0:
        filenm = 'king_sc-%07d-%.2f-%d%d-%.2f-%.2f-%.3f-%.4f_W0-%.1f' % (
            N_tot, sfe, rnd_ps, rnd_imf, m_norm / Msol, rh_phys, fb, args.lam, W0)
    else:
        filenm = 'king_sc-%07d-%.2f-%d%d-%.2f-%.2f-%.3f_W0-%.1f' % (
            N_tot, sfe, rnd_ps, rnd_imf, m_norm / Msol, rh_phys, fb, W0)

    if fb > 0.0:
        for f in ['binary_nbody.dat', 'dat.10', 'mcluster.ini', 'mcluster.out', 'mcluster.err', 'single_nbody.dat']:
            try:
                os.rename(f, filenm + '_' + f)
            except OSError as e:
                print(f"Warning: could not rename {f}: {e}")

    inds  = np.arange(n_out)
    ZZZ   = np.ones_like(m) * metallicity
    ones  = np.ones_like(m).astype(int)
    zeros = np.zeros_like(m)

    # Columns: index, mass, x,y,z,vx,vy,vz, mass0, Z, + SSE bookkeeping zeros/flags.
    # (The mass appears twice on purpose: current mass and initial mass.)
    data = np.concatenate((
        inds, m, pa.T.reshape(pa.size), m, ZZZ, zeros, ones,
        zeros, zeros, zeros, zeros, zeros, zeros, zeros
    )).reshape(19, n_out).T

    # NOTE: the PeTar file intentionally stays in the CLUSTER frame (no
    # X_sc/V_sc shift) — the galactic orbit is handled by PeTar's external
    # (galpot) configuration. Only the .ini file below is shifted.
    data4petar = np.concatenate((m, pa.T.reshape(pa.size))).reshape(7, n_out).T

    data.T[2:5].T[:] += X_sc
    data.T[5:8].T[:] += V_sc

    print(f"Saving to {filenm}.ini and {filenm}-petar.init ...")

    # FIX: header records the ACTUAL number of rows written
    header = f"0000000\n{n_out:07d}\n0.0000000000e+00"
    fmt = ' '.join(['%06i'] + [' %.12E\t'] + ['% .16e'] * 6 + ['\t % .12E %.4f %.1f %i\t'] + ['%.1f'] * 7)
    fmt4petar = ' '.join([' %.12E\t'] + ['% .16e'] * 6)

    np.savetxt(f"{filenm}.ini", data, fmt=fmt, header=header, comments='')
    np.savetxt(f"{filenm}-petar.init", data4petar, fmt=fmt4petar)

    print("Done!")


if __name__ == "__main__":
    gen_king_sc(tsf_matched, sfe_matched)
