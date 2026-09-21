"""
Boundary geometry functions, with a vectorised batch-Newton intersection solver.
"""

import numpy as np
from scipy.optimize import brentq

# -------------------------------------------------------------------------
# Radius and derivatives (original, keep for backward compatibility)
# -------------------------------------------------------------------------
def _radius_r_vec(thetas, C):
    C = np.asarray(C, dtype=float)
    N = (len(C)-1)//2
    c0 = C[0]
    a = C[1:N+1]
    b = C[N+1:2*N+1]
    k = np.arange(1, N+1)
    phases = 2*np.pi*np.outer(thetas, k)
    return c0 + (a*np.cos(phases)).sum(axis=1) + (b*np.sin(phases)).sum(axis=1)

def _radius_r_theta_vec(thetas, C):
    """Vectorised r_theta(theta;C) -- batched counterpart of radius_r_theta."""
    C = np.asarray(C, dtype=float)
    N = (len(C)-1)//2
    a = C[1:N+1]
    b = C[N+1:2*N+1]
    k = np.arange(1, N+1)
    phases = 2*np.pi*np.outer(thetas, k)
    return ((-2*np.pi*k)*a*np.sin(phases)).sum(axis=1) + ((2*np.pi*k)*b*np.cos(phases)).sum(axis=1)

def radius_r(theta, C):
    C = np.asarray(C)
    N = (len(C)-1)//2
    c0 = C[0]
    a = C[1:N+1]
    b = C[N+1:2*N+1]
    k = np.arange(1, N+1)
    return c0 + np.dot(a, np.cos(2*np.pi*k*theta)) + np.dot(b, np.sin(2*np.pi*k*theta))

def radius_r_theta(theta, C):
    C = np.asarray(C)
    N = (len(C)-1)//2
    a = C[1:N+1]
    b = C[N+1:2*N+1]
    k = np.arange(1, N+1)
    return (np.dot(a, -2*np.pi*k*np.sin(2*np.pi*k*theta)) +
            np.dot(b,  2*np.pi*k*np.cos(2*np.pi*k*theta)))

def radius_r_theta_theta(theta, C):
    C = np.asarray(C)
    N = (len(C)-1)//2
    a = C[1:N+1]
    b = C[N+1:2*N+1]
    k = np.arange(1, N+1)
    return (np.dot(a, -(2*np.pi*k)**2*np.cos(2*np.pi*k*theta)) +
            np.dot(b, -(2*np.pi*k)**2*np.sin(2*np.pi*k*theta)))

def gamma(theta, C):
    r = radius_r(theta, C)
    return r * np.array([np.cos(2*np.pi*theta), np.sin(2*np.pi*theta)])

def gamma_prime(theta, C):
    r = radius_r(theta, C)
    r_th = radius_r_theta(theta, C)
    c = np.cos(2*np.pi*theta); s = np.sin(2*np.pi*theta)
    return np.array([r_th*c - 2*np.pi*r*s, r_th*s + 2*np.pi*r*c])

def f_unnormalized(theta, C):
    r = radius_r(theta, C); r_th = radius_r_theta(theta, C)
    c = np.cos(2*np.pi*theta); s = np.sin(2*np.pi*theta)
    return np.array([r_th*s + 2*np.pi*r*c, -r_th*c + 2*np.pi*r*s])

def f_unnormalized_prime(theta, C):
    r = radius_r(theta, C); r_th = radius_r_theta(theta, C); r_thth = radius_r_theta_theta(theta, C)
    c = np.cos(2*np.pi*theta); s = np.sin(2*np.pi*theta); two_pi = 2*np.pi
    df1 = r_thth*s + 4*np.pi*r_th*c - two_pi**2*r*s
    df2 = -r_thth*c + 4*np.pi*r_th*s + two_pi**2*r*c
    return np.array([df1, df2])

def normal_n(theta, C):
    f = f_unnormalized(theta, C)
    return f / np.linalg.norm(f)

def compute_c_inter(theta_inter, C):
    r = radius_r(theta_inter, C)
    n = normal_n(theta_inter, C)
    e_r = np.array([np.cos(2*np.pi*theta_inter), np.sin(2*np.pi*theta_inter)])
    return r * np.dot(n, e_r)

# -------------------------------------------------------------------------
# Original pure‑Python intersection solver (kept as fallback)
# -------------------------------------------------------------------------
def solve_theta_inter(x_k, v_k, C, n_grid=400):
    x_k = np.asarray(x_k); v_k = np.asarray(v_k)
    vx, vy = v_k; xx, xy = x_k
    rhs = vy*xx - vx*xy
    def F(theta):
        r = radius_r(theta, C)
        return r*(vy*np.cos(2*np.pi*theta) - vx*np.sin(2*np.pi*theta)) - rhs
    thetas = np.linspace(0, 1, n_grid, endpoint=True)
    dth = thetas[1]-thetas[0]
    F_vals = np.array([F(th) for th in thetas])
    intervals = []
    for i in range(len(thetas)-1):
        if F_vals[i]*F_vals[i+1] < 0:
            intervals.append((thetas[i], thetas[i+1]))
    if F_vals[-1]*F_vals[0] < 0:
        intervals.append((thetas[-1], thetas[-1]+dth))
    eps_zero = 1e-12
    if abs(F_vals[0]) < eps_zero:
        intervals.append((0.0, dth))
    if abs(F_vals[-1]) < eps_zero:
        intervals.append((1.0-dth, 1.0))
    if not intervals:
        return None
    v_sq = vx*vx + vy*vy
    best_t = np.inf
    best_theta = None
    for lo, hi in intervals:
        try:
            root = brentq(F, lo, hi, xtol=1e-12, rtol=1e-12)
        except ValueError:
            continue
        gx = radius_r(root, C)*np.cos(2*np.pi*root)
        gy = radius_r(root, C)*np.sin(2*np.pi*root)
        t = ((gx-xx)*vx + (gy-xy)*vy) / (v_sq+1e-30)
        if t > 1e-8 and t < best_t:
            best_t = t
            best_theta = root
    if best_theta is not None:
        return best_theta
    # Fallback: smallest |t|
    for lo, hi in intervals:
        try:
            root = brentq(F, lo, hi, xtol=1e-12, rtol=1e-12)
        except ValueError:
            continue
        gx = radius_r(root, C)*np.cos(2*np.pi*root)
        gy = radius_r(root, C)*np.sin(2*np.pi*root)
        t = ((gx-xx)*vx + (gy-xy)*vy) / (v_sq+1e-30)
        if abs(t) < best_t:
            best_t = abs(t)
            best_theta = root
    return best_theta

# -------------------------------------------------------------------------
# Public batch solver -- fully vectorised, bracket-safeguarded batch Newton
# iteration on the ray parameter t in [0,1] (not on theta), starting from
# t=0 for every ray.
# -------------------------------------------------------------------------
def solve_theta_inter_batch(x, v, C, dt, n_newton=25, tol=1e-10):
    """
    Find intersection parameters theta for a batch of rays via a fully
    vectorised, bracket-safeguarded batch Newton iteration on the ray
    parameter t in [0,1], starting every ray from t=0.

    Each ray is x(t) = x + t*dt*v for t in [0,1]: t=0 is the particle's own
    position x_k (inside Omega by construction, since this function is only
    ever called on particles that started the step inside), and t=1 is the
    free-flight endpoint x_prime = x_k + dt*v' (outside Omega by
    construction, since this function is only called on particles whose
    endpoint left Omega). Writing X(t),Y(t) for the components of x(t),
    R(t)=||x(t)||, and theta(t) = angle(x(t))/(2*pi) mod 1, the intersection
    condition is

        G(t) := R(t) - r(theta(t); C) = 0,   with G(0)<0, G(1)>0.

    Since r(.;C) is an exactly 1-periodic Fourier series, G is smooth in t
    right across the theta=0/1 wraparound -- no special-casing of brackets
    or grid intervals is needed, unlike a grid+bisection search over theta.

    Plain (unsafeguarded) Newton from t=0 is NOT reliable here: G can have a
    local extremum very close to t=0 (e.g. for near-tangential encounters),
    whose local slope points Newton toward a spurious root outside [0,1]
    even though the true root is nearby and forward. This was verified
    empirically: on realistic DSMC-scale trajectories, unsafeguarded Newton
    failed to find the (correct, scalar-solver-confirmed) intersection on
    roughly 1 in 300 boundary hits, and catastrophically more often for
    larger/faster trajectories. Instead we use the classic Newton-
    safeguarded-by-bisection scheme ("rtsafe"): maintain a bracket [lo,hi]
    that always contains a root (initially [0,1], since G(0)<0<G(1)); at
    each iteration take the Newton step if it lands strictly inside the
    current bracket, otherwise bisect the bracket. This is still Newton-
    dominated (bisection only triggers when Newton misbehaves) and inherits
    Newton's quadratic convergence once it settles near the root, while
    being guaranteed to converge to *a* root inside [0,1]. For the
    well-resolved-timestep regime this code targets (at most one crossing
    per step), that root is the (unique, hence smallest) forward
    intersection; see the module docstring note below for the multi-
    crossing caveat.

    Newton step (from G(t) = R(t) - r(theta(t))), with W := dt*v:

        dR/dt     = (X*Wx + Y*Wy) / R
        dtheta/dt = (X*Wy - Y*Wx) / (2*pi*R^2)
        G'(t)     = dR/dt - r_theta(theta(t)) * dtheta/dt
        t        <- t - G(t)/G'(t)

    Parameters
    ----------
    x : (K,2) array -- ray origins x_k (assumed inside Omega)
    v : (K,2) array -- velocities v' (NOT pre-multiplied by dt; this
        matches the v'_{k,i} convention used everywhere else in this
        codebase -- internally we use the displacement W = dt*v)
    C : (2N+1,) array -- Fourier coefficients of the boundary
    dt : float -- time step, so that x(t) = x + t*dt*v for t in [0,1]
    n_newton : int -- fixed number of (fully vectorised) safeguarded-Newton
        iterations. Each one is either a Newton step or a bisection of the
        current bracket, so n_newton also upper-bounds the achievable
        precision in the worst case (pure bisection: bracket width 2^-n);
        in the common case most iterations are accepted Newton steps and
        convergence is quadratic.
    tol : float -- a ray is only accepted if its final residual
        |R(t)-r(theta(t))| < tol; otherwise (non-convergence, or the two
        endpoints failing to bracket a root at all -- e.g. a mis-specified
        call where x is not actually inside Omega or x+dt*v not outside)
        it is marked invalid.

    Returns
    -------
    theta : (K,) array -- intersection parameters, NaN if no valid forward
        hit was found.

    Caveat (multiple crossings within one step): if a trajectory crosses
    the boundary more than once within [0,1] (only plausible for a coarse
    timestep relative to boundary wiggle), this bracket-safeguarded scheme
    converges to *some* root in [0,1], not necessarily the smallest -- it
    no longer exhaustively enumerates all crossings the way the previous
    grid+bisection-over-theta implementation did. This matches the
    "well-resolved timestep, single crossing" assumption implicit in the
    Delta-t terms of the adjoint formulas throughout this package.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    K = x.shape[0]
    xx, xy = x[:, 0], x[:, 1]
    Wx, Wy = dt * v[:, 0], dt * v[:, 1]
    two_pi = 2.0 * np.pi

    def eval_G(t):
        X = xx + t * Wx
        Y = xy + t * Wy
        R2 = X * X + Y * Y
        R = np.sqrt(R2)
        theta = (np.arctan2(Y, X) / two_pi) % 1.0
        r_val = _radius_r_vec(theta, C)
        r_th = _radius_r_theta_vec(theta, C)
        G = R - r_val
        dR_dt = (X * Wx + Y * Wy) / np.maximum(R, 1e-300)
        dtheta_dt = (X * Wy - Y * Wx) / (two_pi * np.maximum(R2, 1e-300))
        Gp = dR_dt - r_th * dtheta_dt
        return G, Gp, theta

    lo = np.zeros(K, dtype=np.float64)
    hi = np.ones(K, dtype=np.float64)
    f_lo, _, _ = eval_G(lo)
    f_hi, _, _ = eval_G(hi)
    bracketed = (f_lo < 0.0) & (f_hi > 0.0)   # guaranteed by the caller's contract

    t = np.zeros(K, dtype=np.float64)   # initial guess t=0 for every ray
    converged = np.zeros(K, dtype=bool)

    for _ in range(n_newton):
        G, Gp, _ = eval_G(t)
        # Freeze any ray that has already converged: once |G| is within
        # tolerance, further bracket bookkeeping is not just unnecessary but
        # actively harmful -- the very next "Newton step" computed from a
        # point right on top of a freshly-narrowed lo/hi can, from floating-
        # point noise alone, fail the strict newton_t>lo / newton_t<hi test
        # (since it lands on/at the bound it was just used to set), falling
        # back to bisection against a STALE far bracket endpoint (e.g. hi
        # still ~1.0 because it was never narrowed) and kicking an already-
        # converged ray far away, from which only slow (linear-rate)
        # bisection can claw it back -- often not fully within the
        # remaining iteration budget. Freezing avoids this entirely.
        converged = converged | (np.abs(G) < tol)

        safe_deriv = np.abs(Gp) > 1e-12
        newton_t = t - np.where(safe_deriv, G / np.where(safe_deriv, Gp, 1.0), 0.0)

        # Accept the Newton step only if it stays strictly inside the
        # current bracket; otherwise fall back to plain bisection of the
        # bracket, which always makes guaranteed progress.
        in_bounds = safe_deriv & (newton_t > lo) & (newton_t < hi)
        t_trial = np.where(in_bounds, newton_t, 0.5 * (lo + hi))
        t = np.where(converged, t, t_trial)

        f_new, _, _ = eval_G(t)
        same_sign_as_lo = (f_new > 0.0) == (f_lo > 0.0)
        update_lo = ~converged & same_sign_as_lo
        update_hi = ~converged & ~same_sign_as_lo
        lo = np.where(update_lo, t, lo)
        f_lo = np.where(update_lo, f_new, f_lo)
        hi = np.where(update_hi, t, hi)
        f_hi = np.where(update_hi, f_new, f_hi)

    G_final, _, theta_final = eval_G(t)
    # t >= 0 (not t > 1e-8): a root arbitrarily close to the start point is legitimate --
    # runDraft.py calls this on the REVERSED segment (start = x' just inside the body),
    # where t -> 0+ is exactly the common 'barely penetrated' hit. Rejecting it froze
    # those particles (a discontinuity in C) instead of reflecting them.
    valid = bracketed & np.isfinite(t) & (t >= 0.0) & (np.abs(G_final) < tol)
    return np.where(valid, theta_final, np.nan)