"""
runInflow.py -- OPEN-CHANNEL variant of runDraft.py: inflow at the LEFT boundary, outflow at
the RIGHT boundary, free-stream (inflow AND outflow) TOP/BOTTOM. Exact adjoint gradient of the streamwise impulse on the
body over a window in the statistically steady state.
=======================================================================================

Run:
    python runInflow.py                                   # optimise
    python -c "import runInflow as ri; ri.validate_adjoint()"          # pathwise check (small box)
    python -c "import runInflow as ri; ri.validate_adjoint(collisions=True)"
    python -c "import runInflow as ri; ri.steady_state_report()"       # is the window really steady?
"""
import os
import shutil
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from scipy.special import erfinv, erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adjoint.boundary_geometry import radius_r, radius_r_theta, normal_n, _radius_r_vec, solve_theta_inter_batch
from adjoint.adjoint_jacobians import (dn_dtheta, dtheta_dv_prime, dtheta_dx,
                                       compute_M_ki, compute_N_ki, compute_G_ki, compute_H_ki)
from adjoint.shape_gradient import (perimeter, perimeter_rescale, rescaled_shape_gradient,
                                    dv_reflected_dC, dx_reflected_dC, _dn_dC_total, area,
                                    area_rescale, rescaled_shape_gradient_area,
                                    curvature_penalty, curvature_penalty_gradient)
from shape_optimizer import backtracking_gd, stochastic_endgame

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_NAME = "runInflow_convergence_perimeter_1mil_10coef"
SHAPE_GIF_NAME = "shape_evolution_perimeter_1mil_10coef.gif"

# one-hue sequential ramp (same as video_density.py's SEQ_BLUE): light = early iteration, dark = late
SHAPE_SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                  "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SHAPE_CMAP = colors.LinearSegmentedColormap.from_list("shape_seq_blue", SHAPE_SEQ_BLUE)

# =============================================================================
# Parameters (physics as runDraft; times are the open-channel choices)
# =============================================================================
LX, LY = 16.0, 8.0
U0, T0, RHO0, KN = 1.0, 0.5, 1.0, 1.0
DT = 0.1
T_START = 0.0             # window start. The warm state IS the steady state, so counting starts at once (T_START = 0):
                          # measured 2026-09-13 (12 seeds) the first two time units after the sensitivity starts are the
                          # least noisy pieces of the gradient and the closest to the steady-drag derivative; (0,5] has
                          # half the gradient variance of the old (2,5] at equal cost and no lower edge term.
TOBS = 5.0                # window end; F_x = I_x / (TOBS - T_START), W = 5. Measured with W = 3: the per-seed gradient spread
                          # GROWS with the window end time (0.19 / 0.23 / 0.38 / 0.54 for W = 3 / 6 / 12 / 18 at
                          # 12 seeds: the adjoint of later hits carries longer, re-amplified histories; see the
                          # docstring and window_noise_report()), so a short window with more seeds is the
                          # better use of compute.
N_REF = 1_000_000           # particles that represent the free-stream mass rho0*|Omega_gas,ref| (sets m_p)
CAP = 1_650_000             # pool capacity (alive count fluctuates around ~N_REF * |Omega_gas(box)|/|Omega_gas,ref|)
PREFILL = True            # start from a uniform free stream (shortens the spin-up); False: start empty
WARM_START = True         # every evaluation starts from a cached state obtained by running the channel for
T_WARM = 60.0             # T_WARM time units at C_init (once per seed; stored in OUT_DIR/warm_cache). With
                          # free-stream sides the flow is steady after ~20 time units (the displaced gas leaves
                          # sideways); with periodic sides the upstream reservoir of the choked strip needed ~60.
                          # The cached state is a fixed, C-independent initial condition (then advanced at each
                          # accepted shape, see REWARM_T), exactly like the release of runDraft.
N_COLL_CELLS = (128, 64)  # 0.125 x 0.125 cells = 1/8 mean free path (Kn=1); at N_REF=500_000 this holds
                          # ~61 particles/cell in the steady state, vs ~1950 for the old 32x16 grid at this
                          # count (measured: F_x and |g| agree with 32x16 to <1%, cost unchanged -- forward
                          # 31s, adjoint 54s, warm start 6min, peak RSS ~1GB per seed at N_REF=500_000).
COLLISIONS = True
N_FOURIER = 10
C_init = np.zeros(2 * N_FOURIER + 1); C_init[0] = 1.0; C_init[1] = 0.0   # near-circular start
SIDE_BC = "freestream"    # top/bottom: "freestream" = BOTH inflow and outflow of the undisturbed gas (remove
                          # crossers, inject the thermal half-flux n0*LX*sqrt(T0/2pi) per side; zero net flux in
                          # undisturbed gas, so displaced gas can escape sideways and the far field stays at n0);
                          # "periodic" = the old confined strip (an infinite lattice of bodies -> choked channel).
RIGHT_BACKFLOW = False    # True: also inject at the outlet the backward-moving tail of the free-stream Maxwellian
                          # (~2.4% of the inlet flux at U0=1, T0=0.5; negligible at hypersonic speed ratios).
VEL_DIM = 2               # velocity components. 2 (default): a PLANAR gas, v = (v_x, v_y), Maxwell-molecule collisions
                          # scatter the relative velocity uniformly on the unit circle, the adjoint beta has two
                          # components. 3: the 2D3V model of runDraft (a passive v_z that takes part in collisions),
                          # with which the measurements quoted in this docstring and in tex Sec. 6.6 up to the
                          # "two velocity components" paragraph were made. Enters the warm-cache key.

N_ITER = 20

# ---- window / adjoint / optimiser settings (same values as runDraft.py; this module owns them) ----
N_STEPS = int(round(TOBS / DT))
WEIGHT = 1.0                  # W * m combined
ADD_EDGE_CORRECTION = False   # horizon (edge) estimator: statistically zero in the steady window, adds noise; reported only
N_AVG = 1                     # initial number of CRN seeds; grows adaptively up to N_AVG_MAX
ADAPTIVE_SEEDS = True
N_AVG_MAX = 1
GRAD_SIGNIFICANCE = 2.0       # add seeds while ||g|| < GRAD_SIGNIFICANCE * s.e.(g)
MIN_DISPLACEMENT = 1e-8       # stop when an accepted step moves C by less than this
ARMIJO_C = 0.1                # sufficient-decrease fraction of the noise-aware Armijo test
ENDGAME = False              # curvature-scaled gradient steps after the line search stalls ...
ENDGAME_ITERS = 6
ENDGAME_SIGNIF = 1.5          # ... until ||g||/s.e. < ENDGAME_SIGNIF (gradient statistically zero)
NOISE_SNR = 2.0               # a trial step is unresolvable when its predicted decrease < NOISE_SNR x paired loss s.e.
# ---- objective (PDF: J = I_x + lambda_R oint kappa^2 ds, "or a fixed-area constraint"), constraint, normalisation ----
OBJECTIVE = "drag"            # "drag": J_flow = I_x / (TOBS - T_START), the MEAN DRAG FORCE over the window. The open
                              #   channel has a steady state, so the force (not the window impulse of the closed box) is
                              #   the natural objective: independent of the window length, comparable across windows.
                              # "impulse": I_x itself (the closed-box objective; = W x drag).
CONSTRAINT = "perimeter"      # what is held fixed by rescaling the simulated shape (PDF Sec. 7.2 for the perimeter):
                              #   "perimeter": C_sim = (L0/perimeter(C)) C with L0 = perimeter(C_init) (numerical quadrature);
                              #   "area":      C_sim = sqrt(A0/area(C)) C with A0 = area(C_init), area = pi a0^2 + pi/2 sum C_k^2
                              #                (ANALYTIC; the PDF's alternative to the curvature penalty);
                              #   None:        C_sim = C (size free -> the optimiser just shrinks the body).
CURV_LAMBDA = 1.0             # lambda_R: J = J_flow + lambda_R * oint kappa^2 ds on the SIMULATED shape (0 = off). A circle of
                              #   radius R has oint kappa^2 ds = 2 pi / R (6.3 at R = 1) and J_flow(C_init) ~ 520 (N_REF 20k,
                              #   raw force units), so lambda_R ~ 1-4 makes the penalty 1-5% of the initial drag; the
                              #   gradient of the penalty is analytic (curvature_penalty_gradient). Optional in the PDF.
NORMALIZE_MODE = "both"       # "both" (default): TWO independent normalisations, both fixed at the first evaluation:
                              #   loss = J / J(C_init)  (starts at 1)   and   gradient = dJ/dC / ||dJ/dC (C_init)||  (starts
                              #   at norm 1, so ||g|| -> 0 is read in units of its initial size). The gradient is then a
                              #   constant multiple 1/kappa of the derivative of the printed loss, kappa = ||dJ/dC(C_init)||/J(C_init);
                              #   kappa is exported as evaluate.state["slope_scale"] and the optimiser uses kappa*(g.d) as the
                              #   true slope in its Armijo and noise tests, so the objective is untouched and the line search
                              #   stays exact. The first trial moves C by GRAD_INIT_FRAC*||C_init|| (init_displacement).
                              # "objective": loss J / J(C_init) only; the gradient is its exact derivative.
                              # "gradient": loss AND gradient divided by ONE factor so that ||g(C_init)|| = GRAD_INIT_FRAC*||C_init||.
                              # None: raw units.
GRAD_INIT_FRAC = 0.05        # the FIRST line-search trial moves C by GRAD_INIT_FRAC * ||C_init|| in every mode
GRAD_TOL_REL = 0.0            # optional stop on the noise-corrected ||g|| (0 = off)
STOP_ON_SIGNIFICANCE = True
STEP_INC, STEP_DEC = 1.5, 0.5
FD_H = 1e-4
A_MAX_FRAC = 0.6
# Steady window: the two edge terms cancel in expectation (measured mean 0.6% explicit, 4.5% +- 5.1% implicit
# at C_init) while the estimator adds ~12% noise per seed -> off by default here; steady_state_report()
# still prints it as a diagnostic.
# (ADD_EDGE_CORRECTION = False is set in the settings block above)

GAS_AREA_REF = LX * LY - area(C_init)          # |Omega_gas,ref|: fixed reference (production box, initial body)
M_P = RHO0 * GAS_AREA_REF / N_REF              # mass represented by one simulator particle
N0 = N_REF / GAS_AREA_REF                      # free-stream number density (particles per unit area)


def set_particle_count(n_ref, cap=None):
    """Change the resolution consistently: N_REF, CAP and the DERIVED m_p and n0 (setting ri.N_REF alone
    after import leaves m_p and n0 stale, which silently corrupts the collision rate and the inflow)."""
    global N_REF, CAP, M_P, N0
    N_REF = int(n_ref); CAP = int(cap) if cap is not None else int(round(1.3 * N_REF))
    M_P = RHO0 * GAS_AREA_REF / N_REF; N0 = N_REF / GAS_AREA_REF


def n_steps():
    # steps of the counted simulation (validate_adjoint and the diagnostics override N_STEPS temporarily)
    return int(N_STEPS)


# =============================================================================
# Geometry, reflection Jacobians, exact adjoint, edge (horizon) estimator, evaluate() and shape
# helpers. These are the same routines as in runDraft.py (identical text, checked against the
# closed-box suite), but this module is STANDALONE: it does not import runDraft, so nothing here
# can be overridden by, or override, that file. Only the adjoint/ helper modules and
# shape_optimizer.py are shared.
# =============================================================================
def _k_start():
    """First step index whose impulse counts: step k covers (k dt, (k+1) dt]."""
    return int(round(T_START / DT))


def _step_weight(k):
    """Per-step objective weight (W inside the window (T_START, TOBS], 0 before it)."""
    return WEIGHT if k >= _k_start() else 0.0


def _batch_radius_and_slope(theta, C):
    C = np.asarray(C, dtype=float)
    Nf = (len(C) - 1) // 2
    k = np.arange(1, Nf + 1)
    ph = 2 * np.pi * np.outer(theta, k)
    a, b = C[1:Nf + 1], C[Nf + 1:]
    r = C[0] + np.cos(ph) @ a + np.sin(ph) @ b
    r_th = (-2 * np.pi * k * np.sin(ph)) @ a + (2 * np.pi * k * np.cos(ph)) @ b
    return r, r_th


def body_outward_normal(theta, C):
    r, r_th = _batch_radius_and_slope(theta, C)
    tw = 2 * np.pi * theta
    c, s = np.cos(tw), np.sin(tw)
    fx = r_th * s + 2 * np.pi * r * c
    fy = -r_th * c + 2 * np.pi * r * s
    f = np.stack([fx, fy], axis=1)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def inside_body(x, C):
    r = np.linalg.norm(x, axis=1)
    th = (np.arctan2(x[:, 1], x[:, 0]) / (2 * np.pi)) % 1.0
    return r < _radius_r_vec(th, C)


def selftest_geometry(n=500, seed=0):
    rng = np.random.default_rng(seed)
    C = np.array([1.0, 0.2, -0.15, 0.1, 0.12, -0.08, 0.05])
    thetas = rng.uniform(0, 1, n)
    r_vec, rth_vec = _batch_radius_and_slope(thetas, C)
    n_vec = body_outward_normal(thetas, C)
    for i, th in enumerate(thetas):
        assert abs(r_vec[i] - radius_r(th, C)) < 1e-10
        assert abs(rth_vec[i] - radius_r_theta(th, C)) < 1e-10
        assert np.allclose(n_vec[i], normal_n(th, C), atol=1e-10)


def detect_hit_theta(x_k, v_prime, C, dt):
    x_prime = x_k + dt * v_prime
    return solve_theta_inter_batch(x_prime, -v_prime, C, dt)


def _resample_outside(C, n, rng):
    """Rare fallback (tangent-line reflection still lands inside the body): redraw a
    fresh position uniformly in the box (excluding the body) and a fresh random
    direction at the same speed -- mirrors the main problem's established convention.
    These hits are EXCLUDED from the deterministic adjoint propagation."""
    xs = np.empty((n, 2))
    filled = 0
    while filled < n:
        cand = np.column_stack([rng.uniform(-LX / 2, LX / 2, n),
                                 rng.uniform(-LY / 2, LY / 2, n)])
        cand = cand[~inside_body(cand, C)]
        take = min(len(cand), n - filled)
        xs[filled:filled + take] = cand[:take]
        filled += take
    return xs


_GAS_FRAC_CACHE = {}


def _reflection_rng(seed, k):
    """Generator for specular_step's rare resampling draws at step k: keyed per (seed, step)
    so that a shape change cannot shift the main stream (which then only feeds
    sample_initial, whose draw count is fixed -> bitwise-identical initial condition)."""
    return np.random.default_rng([int(seed), 104729, int(k)])


def _cell_gas_fraction(C, n_sub=16):
    """Fraction of every collision cell's area that is gas (outside the body), from a fixed
    n_sub x n_sub sub-grid per cell; index = ix*ny + iy like `cell` in collide_gas. Without it
    the cells cut by the body use the FULL cell area in rho^j = N^j/(N|Omega_j|), i.e. the gas
    in the layer that produces the impulse collides at gas_fraction x the nominal rate
    (measured 0.65x within 0.1 of the surface). Cached per shape."""
    key = (np.asarray(C, float).tobytes(), LX, LY, N_COLL_CELLS, n_sub)
    f = _GAS_FRAC_CACHE.get(key)
    if f is None:
        nx, ny = N_COLL_CELLS
        u = (np.arange(n_sub) + 0.5) / n_sub
        cx = -LX / 2 + (np.arange(nx)[:, None] + u[None, :]) * (LX / nx)      # (nx, n_sub)
        cy = -LY / 2 + (np.arange(ny)[:, None] + u[None, :]) * (LY / ny)      # (ny, n_sub)
        X, Y = np.broadcast_arrays(cx[:, None, :, None], cy[None, :, None, :])  # (nx, ny, n_sub, n_sub)
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        inside = inside_body(pts, C).reshape(nx, ny, n_sub * n_sub)
        f = (1.0 - inside.mean(axis=2)).reshape(-1)
        _GAS_FRAC_CACHE.clear()
        _GAS_FRAC_CACHE[key] = f
    return f


def _dh_terms(theta, x_k, v_prime, C, wgt=None):
    wgt = WEIGHT if wgt is None else wgt
    n = normal_n(theta, C)
    dn_dth = dn_dtheta(theta, C)
    dth_dv = dtheta_dv_prime(theta, x_k, v_prime, C)
    dth_dx = dtheta_dx(theta, v_prime, C)
    A = np.outer(dn_dth, dth_dv)   # == paper's A = (dn~/dtheta~)(dtheta~/dv')^T
    B = np.outer(dn_dth, dth_dx)   # == the matrix already used inside compute_H_ki

    s = float(np.dot(n, v_prime))
    ex = np.array([1.0, 0.0])
    w = n[0] * v_prime + s * ex

    dh_dvp = 2.0 * wgt * (n[0] * n + A.T @ w)
    dh_dx = 2.0 * wgt * (B.T @ w)
    return dh_dvp, dh_dx


def _dh_dC(theta, v_prime, C, wgt=None):
    wgt = WEIGHT if wgt is None else wgt
    n = normal_n(theta, C)
    s = float(np.dot(n, v_prime))
    w = n[0] * v_prime + s * np.array([1.0, 0.0])
    dn_dC = _dn_dC_total(theta, v_prime, C)
    return 2.0 * wgt * (w @ dn_dC)


def backward_pass_impulse(C, steps, edge=None):
    """Exact gradient of Ix(C). alpha is 2D (positions live in the plane); beta is 3D:
    reflections and the impulse never touch v_z, so no source term or reflection Jacobian
    ever acts on beta_z, but the 3D collision transpose beta_i = (r_i+r_j)/2 + e^(omega.(r_i-r_j))/2
    CREATES a z-component from the xy ones (e^_z != 0) and, at that particle's earlier
    collisions, feeds it back into xy through omega_z. Tracking beta in 2D was measured to
    be 0.5% off the pathwise FD at ~10 collisions/particle; 3D is exact. A record with "nv" = 2
    (the planar-velocity variant of runInflow.py) gets a 2D beta and omega on the unit circle; the
    formulas are the same."""
    N = steps[0].get("n", CAP) if steps else CAP
    nv = steps[0].get("nv", 3) if steps else 3      # velocity components: 3 (2D3V, this file), 2 (planar variant, runInflow)
    beta = np.zeros((N, nv))
    alpha = np.zeros((N, 2))
    grad = np.zeros(2 * N_FOURIER + 1)

    # Horizon (window-edge) term, IMPLICIT part. The jump term at an edge is
    # -/+ (1/dt) sum_i h_i dt*_i/dC with the exact hit-time derivative
    #   dt*/dC = [<e_r,n> dr/dC - <n, dx_k/dC + tau dv'/dC>] / <n,v'>,   tau = <gamma(theta)-x_k, v'>/|v'|^2.
    # edge_term() supplies the first (geometric) part; the second, through the pre-hit
    # trajectory (earlier reflections, collisions), is a linear functional of the state at
    # the edge-step hits and is therefore injected here as adjoint SOURCES at those hits:
    #   alpha_k += c (2W/dt) n_x n,   beta_k += c (2W/dt) n_x tau n,
    # c = -1/2 for each of the two steps adjacent to TOBS (last counted + extra step, weight
    # 0, appended below), +1/2 for the two steps adjacent to T_START (centred estimator).
    # Complete only for first hits of released particles otherwise; measured ~3% of the
    # gradient at production settings, several s.e. in longer collision-free windows.
    steps_all = list(steps)
    edge_c = {}
    if edge is not None:
        (before, after), (last, extra) = edge
        M_ = len(steps_all)
        if last is not None:
            edge_c[M_ - 1] = edge_c.get(M_ - 1, 0.0) - 0.5
        if extra is not None:
            steps_all.append(extra)
            edge_c[M_] = edge_c.get(M_, 0.0) - 0.5
        k0 = _k_start()
        if k0 >= 1:
            if before is not None:
                edge_c[k0 - 1] = edge_c.get(k0 - 1, 0.0) + 0.5
            if after is not None and k0 < M_:
                edge_c[k0] = edge_c.get(k0, 0.0) + 0.5

    for k_rev, rec in enumerate(reversed(steps_all)):
        k_step = len(steps_all) - 1 - k_rev
        c_edge = edge_c.get(k_step, 0.0)
        w_k = rec.get("w", WEIGHT)        # objective weight of this step (0 before the window)
        beta_next = beta
        alpha_next = alpha
        # Open-boundary variants (runInflow.py): slots whose particle at k+1 is not the particle
        # at k -- it left the domain during step k, or a new particle was injected into the slot
        # at the end of step k (with a C-independent state). No sensitivity flows across such a
        # boundary, so the incoming adjoint of those slots is zeroed before step k is processed.
        bnd = rec.get("boundary")
        if bnd is not None and len(bnd):
            beta_next[bnd] = 0.0
            alpha_next[bnd] = 0.0
        beta_after_refl = beta_next.copy()              # default: passthrough
        beta_after_refl[:, :2] += DT * alpha_next       # (x depends on v_xy only)
        alpha_k = alpha_next.copy()                       # default: passthrough

        for hb in rec["hits"]:
            idx = hb["idx"]
            bnext = beta_next[idx][:, :2]                 # reflection Jacobians act on xy only
            anext = alpha_next[idx]
            bnew = np.empty_like(bnext)
            anew = np.empty_like(anext)
            inc_mask = hb.get("incoming")
            for j in range(len(idx)):
                th = hb["theta"][j]; xk = hb["x_k"][j]
                vp = hb["v_prime"][j]; xp = hb["x_prime"][j]

                # Running-cost source terms exist iff the forward counted this hit's
                # impulse (same n.v' < 0 mask as specular_step).
                if w_k != 0.0 and (inc_mask is None or inc_mask[j]):
                    dh_dvp, dh_dx = _dh_terms(th, xk, vp, C, w_k)
                    dh_dC = _dh_dC(th, vp, C, w_k)
                else:
                    dh_dvp = np.zeros(2); dh_dx = np.zeros(2); dh_dC = 0.0
                if c_edge != 0.0 and (inc_mask is None or inc_mask[j]):
                    n_hat = normal_n(th, C)
                    gam = radius_r(th, C) * np.array([np.cos(2 * np.pi * th), np.sin(2 * np.pi * th)])
                    tau = float((gam - xk) @ vp) / float(vp @ vp)
                    src = c_edge * (2.0 * WEIGHT / DT) * n_hat[0] * n_hat
                    dh_dx = dh_dx - src              # alpha_k += src
                    dh_dvp = dh_dvp - tau * src      # beta_k  += tau src

                if hb["was_resampled"][j]:
                    # The post-hit state was redrawn (C-independent given the RNG stream),
                    # so NO downstream sensitivity flows back through the reflection map
                    # (M, N, G, H, dv/dC, dx/dC all dropped). But this hit's OWN impulse
                    # h(x_k, v', C) was still added to Ix in the forward pass, so its
                    # source terms and direct dC term must stay. (Zeroing everything here
                    # dropped them; one such hit gave a 13% gradient error in validation.)
                    bnew[j] = -dh_dvp
                    # the redrawn velocity keeps the SPEED |v'| (direction and position are
                    # RNG-only): d v_new / d v' = (v_new/|v'|)(v'/|v'|)^T, nonzero once
                    # collisions make |v'| depend on C
                    vnew = hb.get("v_new")
                    sp = float(np.linalg.norm(vp))
                    if vnew is not None and sp > 0.0:
                        bnew[j] = bnew[j] + (vp / sp) * (float(vnew[j] @ bnext[j]) / sp)
                    anew[j] = -dh_dx
                    grad += dh_dC
                    continue

                M = compute_M_ki(th, xk, vp, C, np.eye(2))
                Nm = compute_N_ki(th, xp, xk, vp, C, DT)
                G = compute_G_ki(th, xp, vp, C)
                H = compute_H_ki(th, vp, C)

                bnew[j] = M.T @ bnext[j] + Nm.T @ anext[j] - dh_dvp
                anew[j] = G.T @ anext[j] + H.T @ bnext[j] - dh_dx

                dv_dC = dv_reflected_dC(th, vp, C)
                dx_dC = dx_reflected_dC(th, xp, vp, C)
                grad += -dv_dC.T @ bnext[j] - dx_dC.T @ anext[j] + dh_dC
            beta_after_refl[idx, :2] = bnew               # beta_z passes through the reflection
            alpha_k[idx] = anew

        frozen = rec.get("frozen")
        if frozen is not None and len(frozen) > 0:
            # theta solver failed for these: held at x_k with v unchanged, so
            # d x_{k+1} / d v' = 0 -- undo the default free-flight passthrough.
            beta_after_refl[frozen] = beta_next[frozen]

        coll = rec["coll"]
        beta_k = beta_after_refl
        if coll is not None:
            ii, jj = coll["idx_i"], coll["idx_j"]
            u3d = coll["v_i"] - coll["v_j"]
            un3d = np.linalg.norm(u3d, axis=1)
            safe = un3d > 1e-14
            ehat = np.zeros((len(ii), nv))
            ehat[safe] = u3d[safe] / un3d[safe, None]     # e^ = (v_i - v_j)/|v_i - v_j|, all nv components
            omega = coll["omega"]                          # omega with nv components (S^2 or S^1)
            ri, rj = beta_after_refl[ii], beta_after_refl[jj]
            mean = 0.5 * (ri + rj)
            coeff = 0.5 * np.sum(omega * (ri - rj), axis=1)
            s_vec = ehat * coeff[:, None]
            beta_k = beta_after_refl.copy()
            beta_k[ii] = mean + s_vec
            beta_k[jj] = mean - s_vec
        beta = beta_k
        alpha = alpha_k

    return grad


def edge_term(rec, C):
    """Horizon term of ONE window edge estimated from the incoming hits of ONE step adjacent
    to that edge:  -(2W/dt) * sum n_x * (dr/dC_m)(theta) * <e_r, n>  (module docstring).
    Resampled hits are included: the forward pass counted their impulse (and the adjoint
    keeps their source terms); the hit-time shift the term estimates is purely geometric."""
    g = np.zeros(2 * N_FOURIER + 1)
    if rec is None:
        return g
    k = np.arange(1, N_FOURIER + 1)
    for h in rec["hits"]:
        inc = h["incoming"]
        th = h["theta"][inc]
        if len(th) == 0:
            continue
        n = body_outward_normal(th, C)
        e_r = np.stack([np.cos(2 * np.pi * th), np.sin(2 * np.pi * th)], axis=1)
        ern = np.sum(e_r * n, axis=1)
        ph = 2 * np.pi * np.outer(th, k)
        phi = np.column_stack([np.ones_like(th), np.cos(ph), np.sin(ph)])   # dr/dC_m
        g += -(2.0 * WEIGHT / DT) * np.sum(n[:, 0:1] * phi * ern[:, None], axis=0)
    return g


def edge_residual(edge, C):
    """E_TOBS - E_T_START: what d E[Ix]/dC contains beyond the pathwise adjoint (~0 when the
    hit rate is stationary across the window). Each edge term is CENTRED: the average of the
    estimates from the two steps adjacent to the edge (last counted step + the extra step
    after TOBS; the steps before and after T_START). A one-sided estimate is biased by
    ~(dt/2) d(rate)/dt when the hit rate ramps across the edge (measured ~5% here)."""
    start_pair, end_pair = edge
    def centred(pair):
        terms = [edge_term(r, C) for r in pair if r is not None]
        return np.mean(terms, axis=0) if terms else np.zeros(2 * N_FOURIER + 1)
    return centred(end_pair) - centred(start_pair)


def _effective(C):
    """The SIMULATED shape for parameters C: rescaled to the fixed perimeter L0 or the fixed area A0
    (CONSTRAINT), or C itself."""
    C = np.asarray(C, float)
    if CONSTRAINT == "perimeter":
        return perimeter_rescale(C, _L0())
    if CONSTRAINT == "area":
        return area_rescale(C, _A0())
    return C


def _constrained_grad(C, g_sim):
    """Chain rule of a gradient w.r.t. the simulated shape back to the parameters C through the
    rescaling (linear in g_sim, so per-seed gradients may be transformed and then averaged)."""
    if CONSTRAINT == "perimeter":
        return rescaled_shape_gradient(C, _L0(), g_sim)
    if CONSTRAINT == "area":
        return rescaled_shape_gradient_area(C, _A0(), g_sim)
    return np.asarray(g_sim, float)


def _window():
    return n_steps() * DT - T_START


def objective_from_impulse(C_sim, Ix, g_sim=None):
    """The objective J and (if g_sim = d Ix/d C_sim is given) its gradient w.r.t. the SIMULATED shape:
    J = I_x/W (OBJECTIVE "drag") or I_x, plus CURV_LAMBDA * oint kappa^2 ds (analytic penalty and gradient)."""
    W = _window() if OBJECTIVE == "drag" else 1.0
    J = float(Ix) / W
    g = None if g_sim is None else np.asarray(g_sim, float) / W
    if CURV_LAMBDA:
        J += CURV_LAMBDA * curvature_penalty(C_sim)
        if g is not None:
            g = g + CURV_LAMBDA * curvature_penalty_gradient(C_sim)
    return J, g


def set_reference_shape(C):
    """Fix the constraint targets L0 = perimeter(C) and A0 = area(C) (called with C_init by main() and
    the diagnostics; validate_adjoint uses the tested shape)."""
    global _L0_VALUE, _A0_VALUE
    _L0_VALUE = perimeter(C); _A0_VALUE = area(C)


_L0_VALUE = None
_A0_VALUE = None


def _L0():
    return _L0_VALUE


def _A0():
    return _A0_VALUE


def make_evaluate_adjoint(seeds):
    """Primary evaluate() for the optimiser: exact adjoint gradient (+ edge term), no FD.

    Seeds are common random numbers: every call -- current point, gradient, line-search
    trials -- uses the same list, so losses are comparable. The list only GROWS: (i) inside
    a gradient call while ||mean g|| < GRAD_SIGNIFICANCE*||s.e.(g)|| (ADAPTIVE_SEEDS), (ii) via
    evaluate.grow(), which the noise-aware line search calls when a trial's predicted decrease
    is unresolvable against the paired loss noise. Both up to N_AVG_MAX.
    CONTRACT: the returned loss is the common-random-number sample mean of the OBJECTIVE
    J = J_flow + CURV_LAMBDA * oint kappa^2 ds on the simulated (constraint-rescaled) shape,
    J_flow = I_x/W ("drag") or I_x, divided by the normalisation scale; the returned gradient is
    dJ/dC through the rescaling, in the same units: the exact pathwise derivative of the printed
    loss. With ADD_EDGE_CORRECTION (off by default here) the horizon-term estimator is added and the
    gradient estimates d E[J]/dC to first order instead, which is NOT the derivative of the sample
    loss (piecewise smooth with O(1) jumps); the Armijo prediction then uses a slightly different
    slope than the sample loss has, which is intended (see module docstring).
    evaluate.state (refreshed on every call):
      losses      per-seed losses of the LAST call (the optimiser pairs them across C)
      n_seeds, loss_se
      grad_se     norm of the s.e. vector of the mean gradient (C coordinates)
      grad_signif ||g|| / grad_se
      gnorm_corr  sqrt(max(||g||^2 - grad_se^2, 0)): noise-corrected gradient norm -- the
                  raw norm has E||g_hat||^2 = ||g||^2 + grad_se^2 and cannot fall below grad_se
      edge, edge_rel   edge-term residual (C_sim coords) and its size over ||d Ix/d C_sim||"""
    seeds = list(seeds)
    state = {"losses": None, "n_seeds": len(seeds), "grad_se": float("nan"),
             "grad_signif": float("inf"), "gnorm_corr": float("nan"),
             "edge": None, "edge_rel": float("nan"), "loss_se": float("nan"),
             "scale": None,      # loss divisor: "both"/"objective": J(C_init) on the initial seed list; "gradient": ||g||/(frac ||C||)
             "gscale": None,     # "both": extra gradient divisor ||dJ/dC(C_init)||/J(C_init), so that ||g(C_init)|| = 1
             "slope_scale": 1.0, # true slope of the returned loss along d = slope_scale * (g . d)  (= gscale in "both" mode)
             "gnorm0": None}     # ||g|| at the first gradient evaluation (normalised units)

    def _scale(L_all, g_raw):
        """Common normalisation factor of loss and gradient, fixed at the FIRST gradient
        evaluation (the optimiser starts at C_init); 1.0 when NORMALIZE_MODE is None.
          "gradient" : s = ||g_raw(C_init)|| / (GRAD_INIT_FRAC * ||C_init||)  -> ||g/s|| = frac*||C||
          "objective": s = J(C_init) (drag force [+ penalty], raw units)         -> loss/s = 1"""
        if NORMALIZE_MODE is None:
            return 1.0
        if state["scale"] is None:
            if NORMALIZE_MODE == "gradient":
                state["scale"] = float(np.linalg.norm(g_raw)) / (GRAD_INIT_FRAC * float(np.linalg.norm(C_init)))
            else:
                state["scale"] = float(L_all.mean())
            if NORMALIZE_MODE == "both":                      # second, independent factor: ||g(C_init)|| = 1
                state["gscale"] = float(np.linalg.norm(g_raw)) / state["scale"]
                state["slope_scale"] = state["gscale"]
        return state["scale"]
    trS_hist = []      # (dof, tr S^2) of the last few gradient evaluations: a 3-seed covariance
                       # has 2 dof, so the seed-variance is pooled (dof-weighted) over the last 3

    def _pooled_se(G):
        n = len(G)
        if n < 2:
            return float("inf")
        trS_hist.append((n - 1, float(G.var(axis=0, ddof=1).sum())))
        del trS_hist[:-3]
        dof = sum(w for w, _ in trS_hist)
        trS = sum(w * v for w, v in trS_hist) / dof
        return float(np.sqrt(trS / n))

    def grow():
        """Append seeds (nested CRN). Returns True if any were added."""
        if len(seeds) >= N_AVG_MAX:
            return False
        new = list(range(max(seeds) + 1, max(seeds) + 1 + min(len(seeds), N_AVG_MAX - len(seeds))))
        seeds.extend(new)
        state["n_seeds"] = len(seeds)
        return True

    def _scale_or_raw(C):
        if NORMALIZE_MODE is None:
            return 1.0
        if state["scale"] is None:               # loss-only call before any gradient call:
            evaluate(C, want_grad=True)          # define the scale the same way (one gradient evaluation)
        return state["scale"]

    def _per_seed(C, C_sim, s):
        Ix, steps, edge = simulate_and_record(C_sim, s)
        if ADD_EDGE_CORRECTION:
            g_tilde = backward_pass_impulse(C_sim, steps, edge)   # pathwise + implicit edge part
            e = edge_residual(edge, C_sim)                        # explicit (geometric) edge part
            g_sim = g_tilde + e
        else:
            g_tilde = backward_pass_impulse(C_sim, steps)
            e = np.zeros_like(g_tilde)
            g_sim = g_tilde
        # objective on the simulated shape (drag force or impulse, + curvature penalty), then the
        # chain rule through the constraint rescaling (linear in g, so per-seed transformed
        # gradients average to the transformed mean)
        J, g_obj = objective_from_impulse(C_sim, Ix, g_sim)
        g = _constrained_grad(C, g_obj)
        return J, g, g_tilde, e

    def evaluate(C, want_grad=True):
        C_sim = _effective(C)
        if not want_grad:
            sc = _scale_or_raw(C)                # may run one gradient evaluation and GROW `seeds`: fix it first
            L_all = np.array([objective_from_impulse(C_sim, simulate_impulse(C_sim, s))[0] for s in seeds]) / sc
            state.update(losses=L_all, n_seeds=len(seeds),
                         loss_se=float(L_all.std(ddof=1) / np.sqrt(len(L_all))) if len(L_all) > 1 else float("nan"))
            return float(L_all.mean()), None
        res = [_per_seed(C, C_sim, s) for s in seeds]
        while True:
            n = len(res)
            G = np.array([r[1] for r in res])
            g = G.mean(axis=0)
            se = _pooled_se(G)
            if not (ADAPTIVE_SEEDS and n < N_AVG_MAX and np.linalg.norm(g) < GRAD_SIGNIFICANCE * se):
                break
            if n >= 2:
                trS_hist.pop()          # this evaluation will be redone with more seeds
            n_before = len(seeds)
            grow()
            res += [_per_seed(C, C_sim, s) for s in seeds[n_before:]]
        L_all = np.array([r[0] for r in res])
        sc = _scale(L_all, g)                    # fixes the scale(s) at the first (C_init) call
        gs = state["gscale"] if (NORMALIZE_MODE == "both" and state["gscale"]) else 1.0
        L_all = L_all / sc
        g = g / (sc * gs)                        # "both": g / ||dJ/dC(C_init)||, a constant 1/kappa times dL/dC
        se = se / (sc * gs)
        g_tilde = np.mean([r[2] for r in res], axis=0) / (sc * gs)
        e = np.mean([r[3] for r in res], axis=0) / (sc * gs)
        gn = float(np.linalg.norm(g))
        state.update(
            losses=L_all, n_seeds=len(seeds), grad_se=se,
            grad_signif=gn / se if 0 < se < np.inf else float("inf"),
            gnorm_corr=float(np.sqrt(max(gn * gn - se * se, 0.0))) if np.isfinite(se) else float("nan"),
            edge=e, edge_rel=float(np.linalg.norm(e) / (np.linalg.norm(g_tilde) + 1e-300)),
            loss_se=float(L_all.std(ddof=1) / np.sqrt(len(L_all))) if len(L_all) > 1 else float("nan"),
        )
        if state["gnorm0"] is None:
            state["gnorm0"] = gn
        return float(L_all.mean()), g

    evaluate.state = state
    evaluate.seeds = seeds
    evaluate.grow = grow
    return evaluate


def make_evaluate_fd(seeds, H=FD_H):
    """Finite-difference evaluate(), kept ONLY to cross-check the exact adjoint above
    (see validate_adjoint())."""
    def objective(C, seeds_):
        C_sim = _effective(C)
        return float(np.mean([objective_from_impulse(C_sim, simulate_impulse(C_sim, s))[0] for s in seeds_]))

    def evaluate(C, want_grad=True):
        L = objective(C, seeds)
        if not want_grad:
            return L, None
        g = np.zeros_like(C)
        for i in range(len(C)):
            Cp = C.copy(); Cp[i] += H
            Cm = C.copy(); Cm[i] -= H
            g[i] = (objective(Cp, seeds) - objective(Cm, seeds)) / (2 * H)
        return L, g
    return evaluate


def project_C(C):
    C = np.array(C, float)
    C[0] = max(C[0], 1e-3)
    Nf = (len(C) - 1) // 2
    amax = A_MAX_FRAC * C[0]
    for k in range(1, Nf + 1):
        amp = np.hypot(C[k], C[Nf + k])
        if amp > amax:
            C[k] *= amax / amp
            C[Nf + k] *= amax / amp
    return C


def is_valid(C):
    th = np.linspace(0, 1, 200, endpoint=False)
    r = _radius_r_vec(th, C)
    if r.min() <= 0.05:
        return False
    if r.max() >= 0.9 * min(LX, LY) / 2:
        return False
    return True


def body_outline(C, n=400):
    th = np.linspace(0, 1, n)
    r = _radius_r_vec(th, C)
    return r * np.cos(2 * np.pi * th), r * np.sin(2 * np.pi * th)


# =============================================================================
# Deterministic randomness from integer labels (common random numbers)
# =============================================================================
_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)
TAG_COLL_KEY, TAG_COLL_Z, TAG_COLL_PHI, TAG_CELL_U, TAG_NINJ = 1, 2, 3, 4, 5
TAG_INJ_VX, TAG_INJ_VY, TAG_INJ_VZ, TAG_INJ_Y, TAG_INJ_T = 11, 12, 13, 14, 15
TAG_PRE_X, TAG_PRE_Y, TAG_PRE_VX, TAG_PRE_VY, TAG_PRE_VZ = 21, 22, 23, 24, 25
TAG_SIDE_N, TAG_SIDE_VN, TAG_SIDE_VX, TAG_SIDE_VZ, TAG_SIDE_S, TAG_SIDE_T = 31, 32, 33, 34, 35, 36   # + side index*100
TAG_BACK_N, TAG_BACK_VX, TAG_BACK_VY, TAG_BACK_VZ, TAG_BACK_Y, TAG_BACK_T = 41, 42, 43, 44, 45, 46


def u01(*labels):
    """Uniform(0,1) numbers that are a pure function of the integer labels (vectorised
    splitmix64-style hash). Broadcasts over array labels."""
    arrs = [np.atleast_1d(np.asarray(l, dtype=np.int64)).astype(np.uint64) for l in labels]
    shape = np.broadcast(*arrs).shape
    h = np.zeros(shape, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for a in arrs:
            h = (h ^ (a + _M1)) * _M2
            h ^= h >> np.uint64(29)
        z = (h ^ (h >> np.uint64(30))) * _M2
        z = (z ^ (z >> np.uint64(27))) * _M3
        z ^= z >> np.uint64(31)
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 2 ** 53)


def _normal(u, mean, var):
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return mean + np.sqrt(2.0 * var) * erfinv(2.0 * u - 1.0)


_FLUX_TABLE = None


def _flux_vx(u):
    """Inverse CDF of the flux-weighted inflow distribution  v f(v) 1{v>0},  f = N(U0, T0)."""
    global _FLUX_TABLE
    if _FLUX_TABLE is None:
        vg = np.linspace(0.0, U0 + 9.0 * np.sqrt(T0), 20001)
        pdf = vg * np.exp(-(vg - U0) ** 2 / (2.0 * T0))
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(vg))))
        _FLUX_TABLE = (cdf / cdf[-1], vg)
    cdf, vg = _FLUX_TABLE
    return np.interp(u, cdf, vg)


def mean_inward_speed():
    """<v_x^+> = E[max(v_x,0)] for v_x ~ N(U0, T0)."""
    s = np.sqrt(T0); a = U0 / s
    Phi = 0.5 * (1.0 + erf(a / np.sqrt(2.0))); phi = np.exp(-0.5 * a * a) / np.sqrt(2.0 * np.pi)
    return U0 * Phi + s * phi


def inflow_rate():
    """Particles per unit time entering through the left boundary of height LY."""
    return N0 * LY * mean_inward_speed()


# =============================================================================
# Pool operations: prefill, inject, remove
# =============================================================================
def prefill(x, v, alive, pid, seed, C):
    """Uniform free stream in the gas region at t=0 (C-independent count; positions inside the
    body redrawn PER PARTICLE from fresh labelled uniforms so a shape change re-positions only
    the particles whose own candidate flipped)."""
    n_init = int(round(N0 * (LX * LY - area(C_init))))
    if n_init > CAP:
        raise RuntimeError(f"CAP={CAP} too small for the pre-fill ({n_init})")
    j = np.arange(n_init)
    xs = np.column_stack([-LX / 2 + LX * u01(seed, TAG_PRE_X, 0, j), -LY / 2 + LY * u01(seed, TAG_PRE_Y, 0, j)])
    inside = inside_body(xs, C)
    for rnd in range(1, 200):
        if not inside.any():
            break
        cand = np.column_stack([-LX / 2 + LX * u01(seed, TAG_PRE_X, rnd, j), -LY / 2 + LY * u01(seed, TAG_PRE_Y, rnd, j)])
        xs[inside] = cand[inside]
        inside = inside_body(xs, C)
    x[:n_init] = xs
    v[:n_init, 0] = _normal(u01(seed, TAG_PRE_VX, j), U0, T0)
    v[:n_init, 1] = _normal(u01(seed, TAG_PRE_VY, j), 0.0, T0)
    if VEL_DIM == 3:
        v[:n_init, 2] = _normal(u01(seed, TAG_PRE_VZ, j), 0.0, T0)
    alive[:n_init] = True
    pid[:n_init] = j
    return n_init


def inject(x, v, alive, pid, seed, k, counter):
    """Inflow through x = -LX/2 during step k (placed at their end-of-step positions). The
    count and every state are functions of (seed, k, particle id) only. Returns (slots, counter)."""
    expected = inflow_rate() * DT
    lo = int(np.floor(expected))
    n_new = lo + int(u01(seed, TAG_NINJ, k)[0] < expected - lo)
    if n_new == 0:
        return np.empty(0, dtype=int), counter
    free = np.nonzero(~alive)[0]
    if len(free) < n_new:
        raise RuntimeError(f"pool capacity CAP={CAP} exhausted at step {k}: raise CAP")
    slots = free[:n_new]
    p = counter + np.arange(n_new)
    vx = _flux_vx(u01(seed, TAG_INJ_VX, p))
    vy = _normal(u01(seed, TAG_INJ_VY, p), 0.0, T0)
    vz = _normal(u01(seed, TAG_INJ_VZ, p), 0.0, T0) if VEL_DIM == 3 else None
    tf = u01(seed, TAG_INJ_T, p)                      # fraction of the step already travelled
    y0 = -LY / 2 + LY * u01(seed, TAG_INJ_Y, p)
    x[slots, 0] = -LX / 2 + tf * DT * vx
    x[slots, 1] = (y0 + tf * DT * vy + LY / 2) % LY - LY / 2
    v[slots, 0] = vx; v[slots, 1] = vy
    if VEL_DIM == 3:
        v[slots, 2] = vz
    alive[slots] = True
    pid[slots] = p
    return slots, counter + n_new


def side_inflow_rate():
    """Particles per unit time entering through ONE lateral (top or bottom) boundary of length LX
    from undisturbed gas: n0 * LX * E[max(v_n, 0)] with v_n ~ N(0, T0), i.e. n0 LX sqrt(T0/(2 pi))."""
    return N0 * LX * np.sqrt(T0 / (2.0 * np.pi))


def backflow_rate():
    """Particles per unit time entering through the outlet against the drift: n0 LY E[max(-v_x, 0)]."""
    s = np.sqrt(T0); a = U0 / s
    Phi_m = 0.5 * (1.0 - erf(a / np.sqrt(2.0))); phi = np.exp(-0.5 * a * a) / np.sqrt(2.0 * np.pi)
    return N0 * LY * (-U0 * Phi_m + s * phi)


_BACK_TABLE = None


def _back_vx(u):
    """Inverse CDF of the flux-weighted BACKWARD tail  |v| f(v) 1{v<0},  f = N(U0, T0); returns v_x < 0."""
    global _BACK_TABLE
    if _BACK_TABLE is None:
        vg = np.linspace(-(U0 + 9.0 * np.sqrt(T0)), 0.0, 20001)
        pdf = -vg * np.exp(-(vg - U0) ** 2 / (2.0 * T0))
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(vg))))
        _BACK_TABLE = (cdf / cdf[-1], vg)
    cdf, vg = _BACK_TABLE
    return np.interp(u, cdf, vg)


def _place(x, v, alive, pid, slots, p, xs, ys, vx, vy, vz=None):
    x[slots, 0] = xs; x[slots, 1] = ys
    v[slots, 0] = vx; v[slots, 1] = vy
    if VEL_DIM == 3:
        v[slots, 2] = vz
    alive[slots] = True; pid[slots] = p


def _take_slots(alive, n_new, k):
    free = np.nonzero(~alive)[0]
    if len(free) < n_new:
        raise RuntimeError(f"pool capacity CAP={CAP} exhausted at step {k}: raise CAP")
    return free[:n_new]


def inject_sides(x, v, alive, pid, seed, k, counter):
    """Free-stream inflow through the bottom (side 0) and top (side 1) boundaries: undisturbed gas
    at (n0, U0, T0) sends its thermal half-flux inward through each side. Normal speed from the
    flux-weighted half-Maxwellian v_n = sqrt(-2 T0 ln u) (Rayleigh), v_x drifting Maxwellian,
    x uniform along the side, entry time uniform within the step. Particles whose end-of-step
    position already lies beyond the inlet/outlet are not created (they would have left again
    within the step). Returns (slots, counter)."""
    out = []
    for side in (0, 1):                                   # 0: bottom (y = -LY/2, moving +y), 1: top
        base = 100 * (side + 1)
        expected = side_inflow_rate() * DT
        lo = int(np.floor(expected))
        n_new = lo + int(u01(seed, base + TAG_SIDE_N, k)[0] < expected - lo)
        if n_new == 0:
            continue
        p = counter + np.arange(n_new); counter += n_new
        vn = np.sqrt(-2.0 * T0 * np.log(np.clip(u01(seed, base + TAG_SIDE_VN, p), 1e-300, 1.0)))
        vx = _normal(u01(seed, base + TAG_SIDE_VX, p), U0, T0)
        vz = _normal(u01(seed, base + TAG_SIDE_VZ, p), 0.0, T0) if VEL_DIM == 3 else None
        tf = u01(seed, base + TAG_SIDE_T, p)
        x0 = -LX / 2 + LX * u01(seed, base + TAG_SIDE_S, p)
        sgn = 1.0 if side == 0 else -1.0                   # inward normal direction
        vy = sgn * vn
        xs = x0 + tf * DT * vx
        ys = (-sgn) * LY / 2 + tf * DT * vy
        keep = (xs > -LX / 2) & (xs < LX / 2)
        if not keep.any():
            continue
        slots = _take_slots(alive, int(keep.sum()), k)
        _place(x, v, alive, pid, slots, p[keep], xs[keep], ys[keep], vx[keep], vy[keep], vz[keep] if VEL_DIM == 3 else None)
        out.append(slots)
    return (np.concatenate(out) if out else np.empty(0, dtype=int)), counter


def inject_back(x, v, alive, pid, seed, k, counter):
    """Optional backward inflow through the outlet x = +LX/2 (RIGHT_BACKFLOW)."""
    expected = backflow_rate() * DT
    lo = int(np.floor(expected))
    n_new = lo + int(u01(seed, TAG_BACK_N, k)[0] < expected - lo)
    if n_new == 0:
        return np.empty(0, dtype=int), counter
    p = counter + np.arange(n_new); counter += n_new
    vx = _back_vx(u01(seed, TAG_BACK_VX, p))
    vy = _normal(u01(seed, TAG_BACK_VY, p), 0.0, T0); vz = _normal(u01(seed, TAG_BACK_VZ, p), 0.0, T0) if VEL_DIM == 3 else None
    tf = u01(seed, TAG_BACK_T, p); y0 = -LY / 2 + LY * u01(seed, TAG_BACK_Y, p)
    xs = LX / 2 + tf * DT * vx
    ys = y0 + tf * DT * vy
    if SIDE_BC == "periodic":
        ys = (ys + LY / 2) % LY - LY / 2
        keep = np.ones(n_new, bool)
    else:
        keep = np.abs(ys) < LY / 2
    if not keep.any():
        return np.empty(0, dtype=int), counter
    slots = _take_slots(alive, int(keep.sum()), k)
    _place(x, v, alive, pid, slots, p[keep], xs[keep], ys[keep], vx[keep], vy[keep], vz[keep] if VEL_DIM == 3 else None)
    return slots, counter


def inject_all(x, v, alive, pid, seed, k, counter):
    """All inflows of one step: left inlet, lateral free-stream sides (SIDE_BC == "freestream"),
    optional outlet backflow. Returns (injected slots, counter)."""
    parts = []
    s_, counter = inject(x, v, alive, pid, seed, k, counter); parts.append(s_)
    if SIDE_BC == "freestream":
        s_, counter = inject_sides(x, v, alive, pid, seed, k, counter); parts.append(s_)
    if RIGHT_BACKFLOW:
        s_, counter = inject_back(x, v, alive, pid, seed, k, counter); parts.append(s_)
    return np.concatenate(parts), counter


# =============================================================================
# Collisions (Algorithm 1, Maxwell molecules) on the alive particles, labelled randomness
# =============================================================================
def collide_open(x, v, alive, pid, seed, k, C):
    idx = np.nonzero(alive)[0]
    N = len(idx)
    if N < 2:
        return None
    nx, ny = N_COLL_CELLS
    n_cells = nx * ny
    key = u01(seed, TAG_COLL_KEY, k, pid[idx])
    z = 2.0 * u01(seed, TAG_COLL_Z, k, pid[idx]) - 1.0 if VEL_DIM == 3 else None
    phi = 2.0 * np.pi * u01(seed, TAG_COLL_PHI, k, pid[idx])
    u_cell = u01(seed, TAG_CELL_U, k, np.arange(n_cells))
    xa = x[idx]
    ix = np.clip(((xa[:, 0] + LX / 2) / LX * nx).astype(int), 0, nx - 1)
    iy = np.clip(((xa[:, 1] + LY / 2) / LY * ny).astype(int), 0, ny - 1)
    cell = ix * ny + iy
    order_local = np.lexsort((key, cell))
    counts = np.bincount(cell, minlength=n_cells)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    cell_area = (LX / nx) * (LY / ny)
    gas_frac = _cell_gas_fraction(C)
    gas_area = cell_area * np.where(gas_frac > 0, gas_frac, 1.0)
    N_j = counts.astype(float)
    rho_j = np.maximum(N_j - 1.0, 0.0) * M_P / gas_area      # physical partner density
    mu_j = rho_j / KN                                        # Maxwell molecules: velocity independent
    expected = N_j * DT * mu_j / 2.0
    lo = np.floor(expected)
    n_pairs = np.minimum((lo + (u_cell < expected - lo)).astype(int), counts // 2)
    tot = int(n_pairs.sum())
    if tot == 0:
        return None
    cells_with = np.nonzero(n_pairs)[0]
    reps = n_pairs[cells_with]
    within = np.arange(tot) - np.repeat(np.cumsum(reps) - reps, reps)
    a_pos = starts[np.repeat(cells_with, reps)] + 2 * within
    a_loc = order_local[a_pos]; b_loc = order_local[a_pos + 1]
    a = idx[a_loc]; b = idx[b_loc]                           # slot indices
    vi_pre = v[a].copy(); vj_pre = v[b].copy()
    vcm = 0.5 * (vi_pre + vj_pre)
    gmag = np.linalg.norm(vi_pre - vj_pre, axis=1)
    if VEL_DIM == 2:                                         # planar gas: omega uniform on the unit circle
        omega = np.column_stack([np.cos(phi[a_loc]), np.sin(phi[a_loc])])
    else:                                                    # 2D3V: omega uniform on the unit sphere
        st = np.sqrt(1.0 - z[a_loc] ** 2)
        omega = np.column_stack([st * np.cos(phi[a_loc]), st * np.sin(phi[a_loc]), z[a_loc]])
    v[a] = vcm + 0.5 * gmag[:, None] * omega
    v[b] = vcm - 0.5 * gmag[:, None] * omega
    return dict(idx_i=a, idx_j=b, v_i=vi_pre, v_j=vj_pre, omega=omega)


# =============================================================================
# Free flight + one-shot specular reflection on the alive particles; x open; y per SIDE_BC
# =============================================================================
def flight_step(x, v, alive, C, refl_rng, record, weight):
    """Mirror of runDraft.specular_step for a particle pool. Returns (Ix_step, exited_slots)."""
    idx = np.nonzero(alive)[0]
    x_k = x[idx].copy()
    v_prime = v[idx, :2].copy()
    x_prime = x_k + DT * v_prime
    x[idx] = x_prime
    if record is not None:
        record["frozen"] = np.empty(0, dtype=int)
    Ix_step = 0.0
    hit = inside_body(x_prime, C)
    if hit.any():
        hl = np.nonzero(hit)[0]                              # local (within idx)
        theta = detect_hit_theta(x_k[hl], v_prime[hl], C, DT)
        valid = np.isfinite(theta)
        hl_v, th_v = hl[valid], theta[valid]
        if len(hl_v) > 0:
            n = body_outward_normal(th_v, C)
            r_v, _ = _batch_radius_and_slope(th_v, C)
            e_r = np.stack([np.cos(2 * np.pi * th_v), np.sin(2 * np.pi * th_v)], axis=1)
            c_val = r_v * np.sum(n * e_r, axis=1)
            xp_v = x_prime[hl_v]; vp_v = v_prime[hl_v]
            n_dot_v = np.sum(n * vp_v, axis=1); n_dot_x = np.sum(n * xp_v, axis=1)
            incoming = n_dot_v < 0.0
            if incoming.any():
                Ix_step = weight * float(np.sum(2.0 * n_dot_v[incoming] * n[incoming, 0]))
            v_tilde = vp_v - 2.0 * n_dot_v[:, None] * n
            x_tilde = xp_v - 2.0 * (n_dot_x - c_val)[:, None] * n
            still = inside_body(x_tilde, C)
            was_resampled = still.copy()
            if still.any():
                ns = int(still.sum())
                x_tilde[still] = _resample_outside(C, ns, refl_rng)
                speed = np.linalg.norm(v_tilde[still], axis=1)
                ang = refl_rng.uniform(0, 2 * np.pi, ns)
                v_tilde[still] = speed[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=1)
            slots = idx[hl_v]
            x[slots] = x_tilde
            v[slots, :2] = v_tilde
            if record is not None:
                record["hits"].append(dict(
                    idx=slots.copy(), theta=th_v.copy(), x_k=x_k[hl_v].copy(),
                    v_prime=vp_v.copy(), x_prime=xp_v.copy(), was_resampled=was_resampled.copy(),
                    incoming=incoming.copy(), v_new=v_tilde.copy()))
        invalid = hl[~valid]
        if len(invalid) > 0:
            x[idx[invalid]] = x_k[invalid]
            if record is not None:
                record["frozen"] = idx[invalid].copy()
    # x open at both ends; y periodic or free-stream (crossers leave)
    gone = (x[idx, 0] > LX / 2) | (x[idx, 0] < -LX / 2)
    if SIDE_BC == "periodic":
        x[idx, 1] = (x[idx, 1] + LY / 2) % LY - LY / 2
    else:
        gone |= np.abs(x[idx, 1]) > LY / 2
    exited = idx[gone]
    alive[exited] = False
    return Ix_step, exited


# =============================================================================
# Simulation
# =============================================================================
LAST_DIAG = None       # (alive count per step, incoming hits per step) of the last simulation
_WARM_CACHE = {}
_WARM_STEP = {}        # per seed: next (negative) step label available for warm-up / re-warm steps
WARM_VERSION = "2026-09-13d"   # bump whenever u01 / inject / prefill / collide_open / flight_step change
REWARM_T = 10.0        # moving warm start: at every gradient evaluation of the optimiser the cached state is
                       # first advanced by REWARM_T time units AT THE CURRENT SHAPE (see main). The counted
                       # run then starts from (approximately) the steady state of the shape being evaluated
                       # instead of that of C_init (measured 3% drag bias for an optimiser-sized move).


def _warm_key(seed):
    """Everything the warm-up depends on. Compared EXACTLY on load (a stale state of different
    physics would silently corrupt every evaluation)."""
    return (int(seed), float(LX), float(LY), float(T_WARM), int(N_REF), int(CAP), bool(COLLISIONS),
            float(DT), tuple(int(c) for c in N_COLL_CELLS), float(U0), float(T0), float(RHO0), float(KN),
            bool(PREFILL), WARM_VERSION, SIDE_BC, bool(RIGHT_BACKFLOW), int(VEL_DIM), tuple(float(c) for c in C_init))


def warm_state(seed):
    """State (x, v, alive, pid, counter) after running the channel for T_WARM at C_init, computed
    once per (seed, box, ...) and cached in memory and in OUT_DIR/warm_cache/. Step labels of the
    warm-up are negative (-K_warm..-1), so they never collide with the counted run's labels."""
    import hashlib
    key = _warm_key(seed)
    if key in _WARM_CACHE:
        return _WARM_CACHE[key]
    cdir = os.path.join(OUT_DIR, "warm_cache"); os.makedirs(cdir, exist_ok=True)
    tag = hashlib.sha1(repr(key).encode()).hexdigest()[:12]
    fname = os.path.join(cdir, f"warm_s{seed}_{tag}.npz")
    K = int(round(T_WARM / DT))
    if os.path.exists(fname):
        d = np.load(fname, allow_pickle=False)
        if str(d["key"]) == repr(key):
            st = (d["x"], d["v"], d["alive"], d["pid"], int(d["counter"]))
            _WARM_CACHE[key] = st
            _WARM_STEP[seed] = -K - 1
            return st
    x = np.zeros((CAP, 2)); v = np.zeros((CAP, VEL_DIM)); alive = np.zeros(CAP, dtype=bool); pid = np.zeros(CAP, dtype=np.int64)
    counter = prefill(x, v, alive, pid, seed, C_init) if PREFILL else 0
    for kk in range(-K, 0):
        _advance_one(x, v, alive, pid, seed, kk, C_init)
        counter = _advance_inject(x, v, alive, pid, seed, kk, counter)
    st = (x, v, alive, pid, counter)
    np.savez(fname, x=x, v=v, alive=alive, pid=pid, counter=counter, key=np.array(repr(key)))
    _WARM_CACHE[key] = st
    _WARM_STEP[seed] = -K - 1
    return st


def _advance_one(x, v, alive, pid, seed, kk, C):
    """One unrecorded, unweighted step with labels kk (negative for warm-up / re-warm steps)."""
    if COLLISIONS:
        collide_open(x, v, alive, pid, seed, kk, C)
    flight_step(x, v, alive, C, _reflection_rng(seed, 2_000_000 - kk), None, 0.0)   # distinct non-negative rng label


def _advance_inject(x, v, alive, pid, seed, kk, counter):
    _, counter = inject_all(x, v, alive, pid, seed, kk, counter)
    return counter


def rewarm(seed, C, T=None):
    """Moving warm start: advance the cached state of `seed` by T time units at shape C (in place,
    memory cache only; the file cache keeps the C_init state). Labels continue downwards from the
    last warm-up label, so no random label is ever reused. Particles swallowed by the body are
    dropped first (see _simulate)."""
    T = REWARM_T if T is None else T
    x, v, alive, pid, counter = warm_state(seed)
    alive &= ~inside_body(x, C)
    K = int(round(T / DT))
    k_next = _WARM_STEP.get(seed, -int(round(T_WARM / DT)) - 1)
    for i in range(K):
        kk = k_next - i
        _advance_one(x, v, alive, pid, seed, kk, C)
        counter = _advance_inject(x, v, alive, pid, seed, kk, counter)
    _WARM_STEP[seed] = k_next - K
    _WARM_CACHE[_warm_key(seed)] = (x, v, alive, pid, counter)


def _simulate(C, seed, record):
    global LAST_DIAG
    M = n_steps()
    if WARM_START:
        x0, v0, a0, p0, counter = warm_state(seed)
        x, v, alive, pid = x0.copy(), v0.copy(), a0.copy(), p0.copy()
    else:
        x = np.zeros((CAP, 2)); v = np.zeros((CAP, VEL_DIM)); alive = np.zeros(CAP, dtype=bool); pid = np.zeros(CAP, dtype=np.int64)
        counter = prefill(x, v, alive, pid, seed, C) if PREFILL else 0
    # The initial state was prepared for another body (C_init, or the previous accepted shape):
    # particles now INSIDE the body would sit there frozen and still collide. Drop them. The
    # dropped set is piecewise constant in C, so the pathwise adjoint is unaffected.
    alive &= ~inside_body(x, C)
    Ix = 0.0; steps = []; n_alive = np.zeros(M + 1, dtype=int); n_hits = np.zeros(M + 1, dtype=int)
    for k in range(M + 1):                                    # +1: extra unweighted step for the edge estimator
        w = _step_weight(k) if k < M else 0.0
        rec = {"hits": [], "coll": None, "n": CAP, "nv": VEL_DIM, "w": w, "boundary": None} if record else None
        if COLLISIONS:
            coll = collide_open(x, v, alive, pid, seed, k, C)
            if rec is not None:
                rec["coll"] = coll
        Ix_step, exited = flight_step(x, v, alive, C, _reflection_rng(seed, k), rec, w)
        injected, counter = inject_all(x, v, alive, pid, seed, k, counter)
        Ix += Ix_step
        n_alive[k] = int(alive.sum())
        if rec is not None:
            n_hits[k] = sum(int(h["incoming"].sum()) for h in rec["hits"])
            rec["boundary"] = np.unique(np.concatenate([exited, injected]))
            steps.append(rec)
    LAST_DIAG = (n_alive, n_hits)
    if not record:
        return Ix, None, None
    extra = steps.pop()
    k0 = _k_start()
    before = steps[k0 - 1] if 1 <= k0 <= M else None
    after = steps[k0] if 0 <= k0 < M else None
    return Ix, steps, ((before, after), (steps[-1], extra))


def simulate_impulse(C, seed):
    return _simulate(C, seed, record=False)[0]


def simulate_and_record(C, seed):
    return _simulate(C, seed, record=True)




def drag_force(Ix):
    return Ix / (n_steps() * DT - T_START)


def loss_to_force(loss, scale, C=None):
    """Mean drag force from a (normalised) loss: undo the normalisation, subtract the curvature penalty
    (needs C; if C is None the penalty stays in and the value is the full objective), and convert an
    impulse objective to a force."""
    J = float(loss) * float(scale)
    if CURV_LAMBDA and C is not None:
        J -= CURV_LAMBDA * curvature_penalty(_effective(C))
    return J if OBJECTIVE == "drag" else J / _window()


# =============================================================================
# Diagnostics and checks
# =============================================================================
def steady_state_report(C=None, seed=0, block=1.0):
    """Alive-particle count and incoming-hit rate per time block: the window must sit where both
    have levelled off. Also prints the edge-term residual relative to the pathwise gradient."""
    global _L0_VALUE
    C = C_init if C is None else np.asarray(C, float)
    set_reference_shape(C_init)
    t0 = time.time()
    Ix, steps, edge = simulate_and_record(C, seed)
    n_alive, n_hits = LAST_DIAG
    nb = int(round(block / DT)); M = n_steps()
    print(f"open channel {LX}x{LY}, N_REF={N_REF} (m_p={M_P:.4g}), inflow {inflow_rate():.0f} particles/unit time "
          f"({inflow_rate()*DT:.1f}/step), warm start {'T_WARM=' + str(T_WARM) if WARM_START else 'off'}, "
          f"window ({T_START}, {n_steps()*DT}]  [{time.time()-t0:.0f}s]")
    print(f"{'t':>6} {'alive':>7} {'hits/unit time':>15}")
    for b in range(0, M, nb):
        print(f"{(b+nb)*DT:6.1f} {n_alive[b:b+nb].mean():7.0f} {n_hits[b:b+nb].sum()/(nb*DT):15.0f}")
    g_path = backward_pass_impulse(C, steps)
    e_expl = edge_residual(edge, C)
    g_full = backward_pass_impulse(C, steps, edge) + e_expl
    print(f"Ix over window = {Ix:.1f}  ->  drag force F_x = {drag_force(Ix):.1f}")
    print(f"|edge term| / |pathwise gradient| = {np.linalg.norm(g_full - g_path)/np.linalg.norm(g_path):.3f}  "
          f"(explicit part {np.linalg.norm(e_expl)/np.linalg.norm(g_path):.3f}, the rest is the implicit, collision-mediated part)")
    gas_state_report(seed, C)
    return Ix, g_path, g_full


def gas_state_report(seed=0, C=None):
    """Local gas state in the warm state. With free-stream sides (default) the rows next to the
    top/bottom boundaries should sit at (n0, U0, T0) and only the neighbourhood of the body is
    disturbed (compression upstream, wake downstream); with periodic sides the channel chokes
    (upstream gas compressed and decelerated, part of the injected gas returns through the inlet,
    depleted downstream) and the counted drag is the drag in THAT upstream state."""
    C = C_init if C is None else np.asarray(C, float)
    x, v, alive, pid, _ = warm_state(seed)
    xa, va = x[alive], v[alive]
    def col(x_lo, x_hi, y_half=None):
        m = (xa[:, 0] >= x_lo) & (xa[:, 0] < x_hi)
        if y_half is not None:
            m &= np.abs(xa[:, 1]) < y_half
        area_ = (x_hi - x_lo) * (LY if y_half is None else 2 * y_half)
        n = m.sum() / area_ / N0
        u = va[m, 0].mean(); Tm = np.mean(np.sum((va[m] - va[m].mean(0)) ** 2, axis=1)) / VEL_DIM
        back = np.mean(va[m, 0] < 0)
        return n, u, Tm, back
    print(f"boundaries: left inflow {inflow_rate():.0f}/unit time; top/bottom {SIDE_BC}"
          + (f" (each side: {side_inflow_rate():.0f}/unit time in, same out in undisturbed gas)" if SIDE_BC == "freestream" else "")
          + (f"; outlet backflow {backflow_rate():.0f}/unit time" if RIGHT_BACKFLOW else "; outlet: outflow only"))
    print("gas state (warm state; n/n0, mean u_x, T, fraction with v_x<0; free stream would be 1, 1.0, 0.5, 0.08):")
    for name, args in (("inlet column      ", (-LX / 2, -LX / 2 + 0.5)),
                       ("upstream of body  ", (-3.0, -2.0, 1.0)),
                       ("downstream of body", (2.0, 3.0, 1.0)),
                       ("outlet column     ", (LX / 2 - 0.5, LX / 2))):
        n, u, Tm, back = col(*args)
        print(f"   {name}: n/n0={n:5.2f}  u_x={u:5.2f}  T={Tm:5.2f}  P(v_x<0)={back:4.2f}")
    # far-field rows next to the lateral boundaries (should be ~1 if the box is large enough)
    for name, lo, hi in (("bottom row (y<-LY/2+0.5)", -LY / 2, -LY / 2 + 0.5), ("top row    (y> LY/2-0.5)", LY / 2 - 0.5, LY / 2)):
        m = (xa[:, 1] >= lo) & (xa[:, 1] < hi)
        print(f"   {name}: n/n0={m.sum()/(0.5*LX)/N0:5.2f}  u_x={va[m,0].mean():5.2f}  <v_y>={va[m,1].mean():+5.2f}")


def check_expectation_gradient(C=None, seeds=range(6), h=0.05, verbose=True):
    """The pathwise adjoint is exact for each sample path (validate_adjoint) but is NOT an
    unbiased estimate of d E[Ix]/dC: the collision pairing depends on the local densities and
    is not differentiated (Section 6.2 of the tex). In the closed box this bias was 20-40% in
    single components; here it is a factor ~1.5 along the mean direction (1.7 in the periodic
    strip, whose upstream gas was 1.8x denser). This check compares, along the mean pathwise direction d, the pathwise directional
    derivative <g_path, d> with the centred finite-displacement derivative of the seed-averaged Ix
    (h ~ 0.05 keeps the FD noise well below the difference). Ratio ~1 = unbiased."""
    global _L0_VALUE
    C = C_init if C is None else np.asarray(C, float)
    set_reference_shape(C_init)
    seeds = list(seeds)
    G = np.array([backward_pass_impulse(C, simulate_and_record(C, s)[1]) for s in seeds])
    g = G.mean(0); d = g / np.linalg.norm(g)
    dd_path = G @ d
    Ip = np.array([simulate_impulse(C + h * d, s) for s in seeds]); Im = np.array([simulate_impulse(C - h * d, s) for s in seeds])
    dd_fd = (Ip - Im) / (2 * h)
    if verbose:
        print(f"directional derivative of Ix along the mean pathwise direction (C-units, {len(seeds)} seeds):")
        print(f"   pathwise adjoint  <g,d> = {dd_path.mean():9.0f} +- {dd_path.std(ddof=1)/np.sqrt(len(seeds)):.0f}")
        print(f"   centred FD (h={h})      = {dd_fd.mean():9.0f} +- {dd_fd.std(ddof=1)/np.sqrt(len(seeds)):.0f}")
        print(f"   ratio pathwise/FD       = {dd_path.mean()/dd_fd.mean():.2f}   (1 = unbiased; the excess is the undifferentiated collision-pairing dependence)")
    return dd_path, dd_fd




def check_steady_gradient(C=None, seeds=range(12), h=0.05, T_RE=20.0, T_AVG=30.0, components=None, verbose=True):
    """Derivative of the STEADY drag by central finite differences: each shape C +- h e_i is
    re-adapted for T_RE time units from the cached steady state (rewarm, in memory) and its force
    averaged over the next T_AVG; then averaged over the seeds. This is what the optimiser should
    follow. The pathwise adjoint of a short window is instead the response of the drag to a shape
    change applied at t=0 to the frozen steady state, estimated with all collision partners held
    fixed. Measured 2026-09-13 at C_init (12 seeds, per unit time): (413, 51, -651, 167, 25, 4, 2)
    +- (23, 26, 20, 21, 19, 16, 10); pathwise (0,5]: (449, 210, -1060, 657, ...): cos 0.94, a0
    agrees, a2 overstated 1.6x, a1/a3 ~4x. Cost ~12 s per seed and shape. Returns (mean, se)."""
    global _L0_VALUE, T_START, TOBS, N_STEPS
    C = C_init.copy() if C is None else np.asarray(C, float); seeds = list(seeds)
    set_reference_shape(C_init)
    comps = list(range(len(C))) if components is None else list(components)
    saved = (T_START, TOBS, N_STEPS)

    def steady_force(Cs, seed):
        _WARM_CACHE.pop(_warm_key(seed), None)              # back to the cached steady state of C_init
        rewarm(seed, Cs, T=T_RE)                            # adapt the flow to Cs (memory only)
        return simulate_impulse(Cs, seed) / T_AVG
    mean = np.full(len(C), np.nan); se = np.full(len(C), np.nan)
    try:
        T_START = 0.0; TOBS = float(T_AVG); N_STEPS = int(round(T_AVG / DT))
        for i in comps:
            e = np.zeros(len(C)); e[i] = h; t1 = time.time()
            d = np.array([(steady_force(C + e, s_) - steady_force(C - e, s_)) / (2 * h) for s_ in seeds])
            mean[i] = d.mean(); se[i] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            if verbose:
                print(f"   steady dF_x/dC[{i}] = {mean[i]:8.1f} +- {se[i]:5.1f}   ({len(seeds)} seeds, {time.time()-t1:.0f}s)")
    finally:
        T_START, TOBS, N_STEPS = saved
        for s_ in seeds:
            _WARM_CACHE.pop(_warm_key(s_), None)            # leave no re-adapted state behind
    return mean, se

def window_noise_report(seeds=range(12), T=20.0, piece=1.5, t0=2.0, verbose=True):
    """Why the gradient noise does not fall with the window length. Runs T time units per seed
    from the warm state, splits (t0, T] into pieces of length `piece` and computes each piece's
    exact gradient by a separate backward pass (the adjoint is linear in the step weights rec["w"]
    and only looks backwards in time, so the pieces sum exactly to the window gradient). Prints
    per piece the mean gradient norm, the per-seed deviation norm and the impulse statistics,
    then the per-seed relative spread of the nested windows (t0, t0 + k*piece] and of windows of
    two pieces placed later and later. Measured 2026-09-13 (12 seeds, collisions on): the piece
    deviation grows ~t^1.7 with the piece time (330 at t=2.75 -> 11400 at t=19.25) while the mean
    grows <= 2x and the impulse noise is flat; nested spread 0.19 / 0.23 / 0.38 / 0.54 for
    W = 3 / 6 / 12 / 18. Collisions off: flat pieces, spread ~ W^-1/2. Returns (G, I, edges) with
    G[seed, piece, :] the piece gradients and I[seed, piece] the piece impulses."""
    global flight_step, _L0_VALUE, T_START, TOBS, N_STEPS
    seeds = list(seeds); C = C_init.copy(); set_reference_shape(C_init)
    for s_ in seeds:
        warm_state(s_)                                   # build the caches before wrapping flight_step
    saved = (T_START, TOBS, N_STEPS); orig = flight_step; IX = []
    def wrapped(*a, **kw):
        r = orig(*a, **kw); IX.append(r[0]); return r
    edges = np.arange(t0, T + 1e-9, piece); k_edges = np.round(edges / DT).astype(int); P = len(edges) - 1
    G = np.zeros((len(seeds), P, len(C))); I = np.zeros((len(seeds), P))
    try:
        T_START = 0.0; TOBS = float(T); N_STEPS = int(round(T / DT)); flight_step = wrapped
        for si, s_ in enumerate(seeds):
            t1 = time.time(); IX.clear(); _, steps, _ = simulate_and_record(C, s_); ix = np.array(IX[:N_STEPS])
            for j in range(P):
                lo, hi = k_edges[j], k_edges[j + 1]
                for k, rec in enumerate(steps):
                    rec["w"] = 1.0 if lo <= k < hi else 0.0
                G[si, j] = backward_pass_impulse(C, steps[:hi]); I[si, j] = ix[lo:hi].sum()
            del steps
            if verbose:
                print(f"   seed {s_}: {P} piece gradients in {time.time()-t1:.0f}s")
    finally:
        flight_step = orig; T_START, TOBS, N_STEPS = saved
    if verbose:
        m = G.mean(0); dev = np.linalg.norm(G - m, axis=2).mean(0)
        print(f"pieces of {piece} time units, {len(seeds)} seeds (mean gradient norm | per-seed deviation norm | impulse mean +- std):")
        for j in range(P):
            print(f"   t in ({edges[j]:5.2f},{edges[j+1]:5.2f}]: |m|={np.linalg.norm(m[j]):8.0f}  dev={dev[j]:8.0f}  "
                  f"I={I[:, j].mean():7.1f} +- {I[:, j].std(ddof=1) if len(seeds) > 1 else 0:5.1f}")
        def spread(g):
            gm = g.mean(0); return np.linalg.norm(g - gm, axis=1).mean() / np.linalg.norm(gm)
        print("nested windows (per-seed relative spread of the gradient; W^-1/2 prediction from the first; impulse rel. std):")
        s0 = None
        for n in range(1, P + 1):
            W = n * piece; sp = spread(G[:, :n].sum(1)); s0 = sp if s0 is None else s0
            Iw = I[:, :n].sum(1)
            print(f"   ({t0:.0f},{t0+W:5.1f}]  W={W:4.1f}: spread {sp:.3f}   W^-1/2 would give {s0*np.sqrt(piece/W):.3f}   "
                  f"impulse {Iw.std(ddof=1)/Iw.mean() if len(seeds) > 1 else 0:.4f}")
        print("shifted windows of two pieces (same length, later start):")
        for j in range(0, P - 1, 2):
            print(f"   ({edges[j]:5.2f},{edges[j+2]:5.2f}]: spread {spread(G[:, j:j+2].sum(1)):.3f}   "
                  f"|mean|/W {np.linalg.norm(G[:, j:j+2].sum(1).mean(0))/(2*piece):7.0f}")
    return G, I, edges

def validate_adjoint(n_steps_=80, n_avg=2, seed0=3, collisions=False, box=(6.0, 4.0), rescale=None, C=None,
                     t_start=2.0, fd_h=1e-8, verbose=True, constraint=None, curv_lambda=None, reference=None):
    """Pathwise check: exact adjoint vs common-random-number finite differences on the SAME forward
    map, in a small open channel (default 6x4, so that the inflow reaches the body within a short
    run; the free-stream alive count there is ~3300). t_start=2 gives the inflow time to reach
    the body. constraint=None uses the module CONSTRAINT; "perimeter", "area" or False (no
    rescaling; then the raw d Ix/d C_sim IS d Ix/d C) override it for this call; rescale=True/False
    is the old spelling of "perimeter"/False. curv_lambda overrides CURV_LAMBDA for the call, so the
    analytic penalty gradient is checked through the same chain. reference (default: the tested C) sets
    the constraint targets L0/A0; a reference != C makes the rescale factor != 1, exercising the s (or c)
    prefactor of the chain rule. ALWAYS also test a strongly deformed C: the
    near-circular default exercises none of the rare paths (resampled/frozen hits). The edge
    residual printed is a separate diagnostic, not part of the pathwise check. Module state is
    restored afterwards (the small-box warm states are cached under their own key)."""
    global N_STEPS, COLLISIONS, LX, LY, CONSTRAINT, CURV_LAMBDA, _L0_VALUE, _A0_VALUE, T_START
    global ADD_EDGE_CORRECTION, ADAPTIVE_SEEDS, NORMALIZE_MODE
    saved = (N_STEPS, COLLISIONS, LX, LY, CONSTRAINT, CURV_LAMBDA, _L0_VALUE, _A0_VALUE, T_START,
             ADD_EDGE_CORRECTION, ADAPTIVE_SEEDS, NORMALIZE_MODE)
    ADD_EDGE_CORRECTION = False   # the PATHWISE derivative is checked (CRN FD)
    ADAPTIVE_SEEDS = False        # adjoint and FD must average over the SAME seed list
    NORMALIZE_MODE = None         # compare in raw units (the FD path is raw)
    C_test = C_init.copy() if C is None else np.asarray(C, float)
    N_STEPS, COLLISIONS = int(n_steps_), bool(collisions)
    LX, LY = box
    T_START = float(t_start)
    assert _k_start() < N_STEPS, "empty impulse window: t_start >= n_steps*DT"
    if rescale is not None:                                  # old spelling
        constraint = "perimeter" if rescale else False
    if constraint is not None:
        CONSTRAINT = None if constraint is False else constraint
    if curv_lambda is not None:
        CURV_LAMBDA = float(curv_lambda)
    set_reference_shape(C_test if reference is None else np.asarray(reference, float))
    try:
        seeds = list(range(seed0, seed0 + n_avg))
        ev_adj = make_evaluate_adjoint(seeds)
        ev_fd = make_evaluate_fd(seeds, H=fd_h)
        L_a, g_a = ev_adj(C_test, want_grad=True)
        L_f, g_f = ev_fd(C_test, want_grad=True)
        cos = float(g_a @ g_f / (np.linalg.norm(g_a) * np.linalg.norm(g_f) + 1e-300))
        rel = float(np.linalg.norm(g_a - g_f) / (np.linalg.norm(g_f) + 1e-300))
        if verbose:
            print(f"L (adjoint)={L_a:.6f}  L (fd)={L_f:.6f}")
            print(f"adjoint grad: {g_a}")
            print(f"fd      grad: {g_f}")
            print(f"cosine={cos:.6f}  rel={rel:.4f}   (edge residual |E_T-E_T0|/|g| = "
                  f"{ev_adj.state['edge_rel']:.3f}, diagnostic only)")
        return cos, rel
    finally:
        (N_STEPS, COLLISIONS, LX, LY, CONSTRAINT, CURV_LAMBDA, _L0_VALUE, _A0_VALUE, T_START,
         ADD_EDGE_CORRECTION, ADAPTIVE_SEEDS, NORMALIZE_MODE) = saved


# =============================================================================
# Main
# =============================================================================
def plot_results(C_init_eff, C_opt_eff, hist, scale):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot([loss_to_force(l, scale) for l in hist["loss"]], "b-o", ms=3)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel(("objective J = F_x + lambda_R P" if CURV_LAMBDA else "drag force F_x") + f"  (window ({T_START},{n_steps()*DT}])")
    ax1.set_title("Drag convergence"); ax1.grid(True)
    ax2.semilogy(hist["gnorm"], "m-o", ms=3); ax2.set_xlabel("iteration"); ax2.set_ylabel("||grad|| (normalised units)")
    ax2.set_title("Gradient norm"); ax2.grid(True, which="both")
    ax3.plot(*body_outline(C_init_eff), "firebrick", lw=2, label="initial body")
    ax3.plot(*body_outline(C_opt_eff), "royalblue", lw=2.5, label="optimised body")
    ax3.annotate("inflow  U0 ->", xy=(0, 1.4), ha="center", color="green")
    ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3); ax3.legend(); ax3.set_title("Body shape (open channel)")
    plt.tight_layout(); out = os.path.join(OUT_DIR, PLOT_NAME); plt.savefig(out, dpi=140); print(f"Saved {out}")


def shape_evolution_gif(C_hist, force_hist=None, out=None, fps=2, dpi=110, effective=True, hold_frames=6):
    """Animate the body shape through the accepted iterations of an optimisation run.

    C_hist   : list of raw shape-coefficient vectors, one per accepted iterate, index 0 = the
               starting shape (C_init). main() collects this automatically (one entry per call
               of on_accept, prefixed with C_init) and passes it here after the run.
    force_hist : optional list of the SAME length as C_hist with a scalar to show alongside the
               shape (e.g. the drag force from loss_to_force(hist["loss"][i], scale, C_hist[i]));
               draws a second panel with the convergence curve and a marker tracking the current
               frame. None: shape panel only.
    out      : output basename (default SHAPE_GIF_NAME, written to OUT_DIR); ".gif" or ".mp4" is
               appended depending on whether ffmpeg is on the PATH (same rule as video_density.py).
    effective: plot the SIMULATED shape _effective(C) (rescaled to the fixed perimeter/area) --
               what the optimiser actually compares -- rather than the raw coefficients. True by
               default; the raw C is what's optimised but perimeter_rescale/area_rescale can move
               the visible outline by an overall scale factor, which is usually what you want to see.
    hold_frames : repeat the LAST frame this many times so the gif visibly pauses on the optimum
               instead of looping straight back to the start.

    Returns the path written."""
    C_hist = [np.asarray(C, float) for C in C_hist]
    shapes = [_effective(C) if effective else C for C in C_hist]
    outlines = [body_outline(C) for C in shapes]
    n = len(outlines)
    if n == 0:
        raise ValueError("shape_evolution_gif: C_hist is empty")
    rmax = max(float(np.max(np.hypot(x, y))) for x, y in outlines) * 1.15

    have_force = force_hist is not None and len(force_hist) == n
    fig, axes = plt.subplots(1, 2 if have_force else 1, figsize=(11, 5) if have_force else (6, 5.5))
    ax_shape = axes[0] if have_force else axes
    ax_shape.set_aspect("equal"); ax_shape.set_xlim(-rmax, rmax); ax_shape.set_ylim(-rmax, rmax)
    ax_shape.grid(True, alpha=0.3)
    ax_shape.plot(*outlines[0], color="firebrick", lw=1.5, ls="--", label="initial shape")
    (line,) = ax_shape.plot([], [], lw=2.5)
    title = ax_shape.set_title("")
    ax_shape.legend(loc="upper right", fontsize=8)
    if have_force:
        ax_force = axes[1]
        ax_force.plot(force_hist, "b-o", ms=3)
        ax_force.set_xlabel("iteration"); ax_force.set_ylabel("drag force" if OBJECTIVE == "drag" else "objective J")
        ax_force.set_title("Convergence"); ax_force.grid(True)
        (marker,) = ax_force.plot([0], [force_hist[0]], "ro", ms=8)
    n_frames = n + max(0, int(hold_frames))

    def update(k):
        i = min(k, n - 1)
        x, y = outlines[i]
        line.set_data(x, y)
        line.set_color(SHAPE_CMAP(0.15 + 0.85 * i / max(n - 1, 1)))
        t = f"iteration {i}/{n - 1}"
        if have_force:
            t += f"   {'F_x' if OBJECTIVE == 'drag' else 'J'} = {force_hist[i]:.1f}"
            marker.set_data([i], [force_hist[i]])
        title.set_text(t)
        return (line, title, marker) if have_force else (line, title)

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    base = os.path.join(OUT_DIR, out or SHAPE_GIF_NAME)
    base = base[:-4] if base.endswith((".gif", ".mp4")) else base
    if shutil.which("ffmpeg"):
        path = base + ".mp4"
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2000), dpi=dpi)
    else:
        path = base + ".gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved {path} ({n} iterations, {n_frames / fps:.1f}s incl. a hold on the final shape)")
    return path


def main():
    global _L0_VALUE
    selftest_geometry(); print("Geometry selftest passed.")
    set_reference_shape(C_init)
    print(f"Open-channel drag minimisation (inflow left / outflow right / {SIDE_BC} top-bottom) -- exact adjoint")
    print(f"  box {LX}x{LY}  U0={U0} T0={T0} Kn={KN}  dt={DT}  window ({T_START}, {n_steps()*DT}] = steps {_k_start()}..{n_steps()-1}"
          f"  N_REF={N_REF} (inflow {inflow_rate()*DT:.1f} particles/step, warm start T_WARM={T_WARM if WARM_START else 'off'})  collisions={COLLISIONS}")
    Cn = float(np.linalg.norm(C_init))
    P0 = curvature_penalty(_effective(C_init))
    print(f"  objective: {OBJECTIVE}" + (f" + {CURV_LAMBDA:g} x oint kappa^2 ds (= {CURV_LAMBDA*P0:.1f} at C_init, P0 = {P0:.3f})" if CURV_LAMBDA else " (no curvature penalty)")
          + f"   constraint: {CONSTRAINT} (L0 = {_L0_VALUE:.3f}, A0 = {_A0_VALUE:.3f})")
    units = {"both": "loss J/J(C_init) -> 1 and gradient/||g(C_init)|| -> 1 (independent factors; the optimiser uses the true slope kappa g.d)",
             "objective": "loss J/J(C_init) -> 1, gradient = its exact derivative",
             "gradient": f"one factor: ||g(C_init)|| = {GRAD_INIT_FRAC} x ||C_init||", None: "raw"}[NORMALIZE_MODE]
    print(f"  units: {units};  first trial moves C by {GRAD_INIT_FRAC} x ||C_init|| = {GRAD_INIT_FRAC*Cn:.4f}"
          f"   seeds {N_AVG}->{N_AVG_MAX}   edge term {'added' if ADD_EDGE_CORRECTION else 'reported only'}")
    init_disp = GRAD_INIT_FRAC * Cn
    seeds = list(range(N_AVG)); evaluate_core = make_evaluate_adjoint(seeds); t0 = time.time()

    class _Evaluate:
        """Wrapper: before every GRADIENT evaluation (i.e. at every accepted iterate) the cached
        warm states of all seeds in use are advanced by REWARM_T at that shape, so the counted
        window starts from the steady state of the shape being evaluated. Loss-only trials reuse
        the same states, so the line search compares like with like."""
        def __init__(self):
            # instance attributes, not class attributes: a plain function assigned in the CLASS
            # body becomes a descriptor, so evaluate.grow() would auto-bind self as an argument
            # grow() does not take -- "grow() takes 0 positional arguments but 1 was given" --
            # triggered only via shape_optimizer's evaluate.grow() (the noise-limited retry path),
            # not via the ADAPTIVE_SEEDS path inside evaluate() itself, which calls grow() as a
            # plain local name. Instance attributes are returned as-is, with no such binding.
            self.state = evaluate_core.state
            self.seeds = evaluate_core.seeds
            self.grow = evaluate_core.grow

        def __call__(self, C, want_grad=True):
            if want_grad and REWARM_T > 0:
                C_sim = _effective(C)
                for s_ in list(evaluate_core.seeds):
                    rewarm(s_, C_sim)
            return evaluate_core(C, want_grad)
    evaluate = _Evaluate()

    shape_hist = [C_init.copy()]      # one entry per on_accept call (line search + endgame), prefixed
                                       # with C_init; aligns with hist["loss"] (see shape_evolution_gif below)

    def on_accept(it, C, loss, grad):
        shape_hist.append(C.copy())
        st = evaluate.state; sc = st["scale"] or 1.0
        print(f"   [{time.time()-t0:.0f}s] perimeter={perimeter(_effective(C)):.3f}  seeds={st['n_seeds']}  ||g||/s.e.={st['grad_signif']:.1f}"
              f"  ||g||corr={st['gnorm_corr']:.4g} (={100*st['gnorm_corr']/st['gnorm0']:.1f}% of initial)  F_x={loss_to_force(loss, sc, C):.1f}"
              f"  edge-term={100*st['edge_rel']:.0f}%")
    last = {"loss": None}

    def converged(it, C, loss, grad):
        st = evaluate.state; dec = None if last["loss"] is None else last["loss"] - loss; last["loss"] = loss
        if GRAD_TOL_REL > 0 and np.isfinite(st["gnorm_corr"]) and st["gnorm_corr"] <= GRAD_TOL_REL * st["gnorm0"]:
            return True
        if not STOP_ON_SIGNIFICANCE:
            return False
        return (st["grad_signif"] < GRAD_SIGNIFICANCE and (not ADAPTIVE_SEEDS or st["n_seeds"] >= N_AVG_MAX)
                and dec is not None and np.isfinite(st["loss_se"]) and dec < 2.0 * st["loss_se"])

    C_opt, hist = backtracking_gd(
        C_init.copy(), evaluate, n_iter=N_ITER, init_step=1.0,
        init_displacement=(None if NORMALIZE_MODE == "gradient" else init_disp),   # "gradient" mode: alpha = 1 already moves C by init_disp
        step_increase=STEP_INC, step_decrease=STEP_DEC, armijo_c=ARMIJO_C,
        project=project_C, is_valid=is_valid, verbose=True, on_accept=on_accept,
        min_displacement=MIN_DISPLACEMENT, converged=converged, noise_aware=True, noise_snr=NOISE_SNR)
    print(f"\nLine-search phase done in {time.time()-t0:.0f}s.  status: {hist['status']}  seeds {hist['n_seeds'][0]} -> {hist['n_seeds'][-1]}")
    if ENDGAME and hist["status"] in ("noise-limited", "converged", "min_displacement", "max iterations"):
        hs = [h for h in hist.get("h_secant", []) if np.isfinite(h) and h > 0]
        C_opt, eg = stochastic_endgame(C_opt, evaluate, n_iter=ENDGAME_ITERS,
                                       signif_stop=(ENDGAME_SIGNIF if STOP_ON_SIGNIFICANCE else 0.0),
                                       max_disp=init_disp, min_disp=MIN_DISPLACEMENT, project=project_C, is_valid=is_valid,
                                       verbose=True, on_accept=on_accept, h_init=(np.median(hs[-3:]) if hs else None),
                                       grad_tol=(GRAD_TOL_REL * evaluate.state["gnorm0"] if GRAD_TOL_REL > 0 else None))
        hist["loss"] += eg["loss"][1:]; hist["gnorm"] += eg["gnorm"][1:]
        print(f"Endgame status: {eg['status']}   ||g||/s.e.: {eg['signif'][0]:.2f} -> {eg['signif'][-1]:.2f}")
    sc = evaluate.state["scale"] or 1.0
    L0_same, _ = evaluate(C_init, want_grad=False)      # NOTE: from the warm states last advanced at C_opt
    print(f"\nDone in {time.time()-t0:.0f}s.  Drag force on the final {evaluate.state['n_seeds']}-seed list: "
          f"F_x {loss_to_force(hist['loss'][0], sc, C_init):.1f} (first evaluation, {hist['n_seeds'][0]} seeds, C_init steady state) -> "
          f"{loss_to_force(hist['loss'][-1], sc, C_opt):.1f}   [C_init re-evaluated on the final list from the C_opt-adapted states: {loss_to_force(L0_same, sc, C_init):.1f}]"
          + (f"   (objective J incl. penalty: {L0_same*sc:.1f} -> {hist['loss'][-1]*sc:.1f})" if CURV_LAMBDA else "")
          + f"   normalised loss {L0_same:.4f} -> {hist['loss'][-1]:.4f}")
    print(f"C_opt = {np.round(C_opt, 4)}")
    plot_results(_effective(C_init), _effective(C_opt), hist, sc)
    nf = min(len(shape_hist), len(hist["loss"]))          # usually equal; truncate defensively otherwise
    force_hist = [loss_to_force(hist["loss"][i], sc, shape_hist[i]) for i in range(nf)]
    shape_evolution_gif(shape_hist[:nf], force_hist=force_hist)


if __name__ == "__main__":
    main()
