"""
shape_gradient.py – fully vectorised dL/dC for DSMC adjoint,
with per‑hit gradient clipping (restores area‑preserving behaviour).
"""

import numpy as np

from .boundary_geometry import (
    radius_r, radius_r_theta, radius_r_theta_theta,
    f_unnormalized, f_unnormalized_prime,
    normal_n, gamma, gamma_prime,
    solve_theta_inter, solve_theta_inter_batch, compute_c_inter,
)


# ----------------------------------------------------------------------
# Vectorised geometry helpers
# ----------------------------------------------------------------------

def _radius_r_vec(thetas: np.ndarray, C) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    c0 = C[0]
    a = C[1:N+1]
    b = C[N+1:2*N+1]
    k = np.arange(1, N+1)
    phases = 2 * np.pi * np.outer(thetas, k)
    return c0 + (a * np.cos(phases)).sum(axis=1) + (b * np.sin(phases)).sum(axis=1)


def _batch_normal_and_dn(thetas: np.ndarray, C) -> tuple[np.ndarray, np.ndarray]:
    K = len(thetas)
    C = np.asarray(C, dtype=float)
    Nf = (len(C) - 1) // 2
    k = np.arange(1, Nf+1)
    phases = 2 * np.pi * np.outer(thetas, k)
    a = C[1:Nf+1]
    b = C[Nf+1:]

    r = C[0] + (a * np.cos(phases)).sum(axis=1) + (b * np.sin(phases)).sum(axis=1)
    r_th = ((-2*np.pi*k) * a * np.sin(phases) + (2*np.pi*k) * b * np.cos(phases)).sum(axis=1)
    r_thth = ((-(2*np.pi*k)**2) * a * np.cos(phases) + (-(2*np.pi*k)**2) * b * np.sin(phases)).sum(axis=1)

    c2 = np.cos(2 * np.pi * thetas)
    s2 = np.sin(2 * np.pi * thetas)
    two_pi = 2 * np.pi

    f = np.stack([r_th * s2 + two_pi * r * c2,
                  -r_th * c2 + two_pi * r * s2], axis=1)
    fnorm = np.linalg.norm(f, axis=1, keepdims=True)
    n = f / fnorm

    fp = np.stack([r_thth * s2 + 4*np.pi * r_th * c2 - (two_pi)**2 * r * s2,
                   -r_thth * c2 + 4*np.pi * r_th * s2 + (two_pi)**2 * r * c2], axis=1)
    n_dot_fp = (n * fp).sum(axis=1, keepdims=True)
    dn = (fp - n_dot_fp * n) / fnorm
    return n, dn


def _batch_dc_c(thetas: np.ndarray, C, n_all, dn_all, r_all=None):
    K = len(thetas)
    if r_all is None:
        r_all = _radius_r_vec(thetas, C)
    c2 = np.cos(2 * np.pi * thetas)
    s2 = np.sin(2 * np.pi * thetas)
    e_r = np.stack([c2, s2], axis=1)
    e_perp = np.stack([-s2, c2], axis=1)

    n_dot_er = (n_all * e_r).sum(axis=1)
    c_vals = r_all * n_dot_er

    Nf = (len(C) - 1) // 2
    k = np.arange(1, Nf+1)
    phases = 2 * np.pi * np.outer(thetas, k)
    a = C[1:Nf+1]
    b = C[Nf+1:]
    r_th = ((-2*np.pi*k) * a * np.sin(phases) + (2*np.pi*k) * b * np.cos(phases)).sum(axis=1)

    e_dot_dn = (e_r * dn_all).sum(axis=1)
    n_dot_ep = (n_all * e_perp).sum(axis=1)
    F_vals = r_th * n_dot_er + r_all * (e_dot_dn + 2 * np.pi * n_dot_ep)
    return c_vals, F_vals


def _batch_dtheta_dv(thetas: np.ndarray, x_k_arr, v_arr, C):
    K = len(thetas)
    C = np.asarray(C)
    Nf = (len(C)-1)//2
    k = np.arange(1, Nf+1)
    phases = 2 * np.pi * np.outer(thetas, k)
    a = C[1:Nf+1]
    b = C[Nf+1:]
    r = C[0] + (a * np.cos(phases)).sum(axis=1) + (b * np.sin(phases)).sum(axis=1)
    r_th = ((-2*np.pi*k) * a * np.sin(phases) + (2*np.pi*k) * b * np.cos(phases)).sum(axis=1)

    tw = 2 * np.pi * thetas
    c2, s2 = np.cos(tw), np.sin(tw)
    vx, vy = v_arr[:,0], v_arr[:,1]
    xx, xy = x_k_arr[:,0], x_k_arr[:,1]

    num = np.stack([r * s2 - xy, -r * c2 + xx], axis=1)
    denom = r_th * (vy*c2 - vx*s2) - 2*np.pi * r * (vy*s2 + vx*c2)
    safe = np.abs(denom) >= 1e-2
    result = np.zeros_like(num)
    if safe.any():
        result[safe] = num[safe] / denom[safe, None]
    return result


def _batch_dtheta_dx(thetas: np.ndarray, v_arr, C):
    K = len(thetas)
    C = np.asarray(C)
    Nf = (len(C)-1)//2
    k = np.arange(1, Nf+1)
    phases = 2 * np.pi * np.outer(thetas, k)
    a = C[1:Nf+1]
    b = C[Nf+1:]
    r = C[0] + (a * np.cos(phases)).sum(axis=1) + (b * np.sin(phases)).sum(axis=1)
    r_th = ((-2*np.pi*k) * a * np.sin(phases) + (2*np.pi*k) * b * np.cos(phases)).sum(axis=1)

    tw = 2 * np.pi * thetas
    c2, s2 = np.cos(tw), np.sin(tw)
    vx, vy = v_arr[:,0], v_arr[:,1]

    num = np.stack([vy, -vx], axis=1)
    denom = r_th * (vy*c2 - vx*s2) - 2*np.pi * r * (vy*s2 + vx*c2)
    safe = np.abs(denom) >= 1e-2
    result = np.zeros_like(num)
    if safe.any():
        result[safe] = num[safe] / denom[safe, None]
    return result


def _batch_dtheta_inter_dC(thetas: np.ndarray, v_arr, C):
    K = len(thetas)
    C_arr = np.asarray(C)
    Nf = (len(C_arr)-1)//2
    k = np.arange(1, Nf+1)                     # (Nf,)
    tw = 2 * np.pi * thetas                    # (K,)
    c2, s2 = np.cos(tw), np.sin(tw)
    vx, vy = v_arr[:,0], v_arr[:,1]

    # Compute r(θ) and r_θ(θ) for all thetas (vectorised)
    phases = 2 * np.pi * np.outer(thetas, k)   # (K, Nf)
    cos_ph = np.cos(phases)
    sin_ph = np.sin(phases)
    a = C_arr[1:Nf+1]
    b = C_arr[Nf+1:]
    r = C_arr[0] + a @ cos_ph.T + b @ sin_ph.T   # (K,)
    r_th = (-2*np.pi*k) * a * sin_ph + (2*np.pi*k) * b * cos_ph
    r_th = r_th.sum(axis=1)                     # (K,)

    # Build dr_dC matrix (K, 2N+1)
    dr_dC = np.zeros((K, len(C_arr)))
    dr_dC[:, 0] = 1.0
    # For each mode, fill two columns
    # We can do this by creating a matrix of cos and sin terms for all modes
    cos_terms = np.cos(2 * np.pi * np.outer(thetas, k))   # (K, Nf)
    sin_terms = np.sin(2 * np.pi * np.outer(thetas, k))   # (K, Nf)
    dr_dC[:, 1:Nf+1] = cos_terms
    dr_dC[:, Nf+1:2*Nf+1] = sin_terms

    factor = vy * c2 - vx * s2
    denom = r_th * factor - 2 * np.pi * r * (vy * s2 + vx * c2)
    safe = np.abs(denom) >= 1e-2
    result = np.zeros((K, len(C_arr)))
    if safe.any():
        result[safe] = -dr_dC[safe] * factor[safe, None] / denom[safe, None]
    return result


def _batch_dn_dC_fixed(thetas: np.ndarray, C):
    K = len(thetas)
    C_arr = np.asarray(C)
    Nf = (len(C_arr)-1)//2
    tw = 2 * np.pi * thetas
    c2, s2 = np.cos(tw), np.sin(tw)

    k = np.arange(1, Nf+1)                     # (Nf,)
    cos_terms = np.cos(2 * np.pi * np.outer(thetas, k))   # (K, Nf)
    sin_terms = np.sin(2 * np.pi * np.outer(thetas, k))   # (K, Nf)
    dr_dC = np.zeros((K, len(C_arr)))
    dr_dC[:, 0] = 1.0
    dr_dC[:, 1:Nf+1] = cos_terms
    dr_dC[:, Nf+1:2*Nf+1] = sin_terms

    # drth_dC: derivatives of r_θ w.r.t coefficients
    drth_dC = np.zeros((K, len(C_arr)))
    # For a_k: ∂r_θ/∂a_k = -2πk sin(2πkθ)
    drth_dC[:, 1:Nf+1] = -2*np.pi*k * sin_terms
    # For b_k: ∂r_θ/∂b_k =  2πk cos(2πkθ)
    drth_dC[:, Nf+1:2*Nf+1] = 2*np.pi*k * cos_terms

    # Compute r(θ) and r_θ(θ) using Fourier series (vectorised)
    phases = 2 * np.pi * np.outer(thetas, k)   # (K, Nf)
    cos_ph = np.cos(phases)
    sin_ph = np.sin(phases)
    a = C_arr[1:Nf+1]
    b = C_arr[Nf+1:]
    r = C_arr[0] + a @ cos_ph.T + b @ sin_ph.T   # (K,)
    r_th = (-2*np.pi*k) * a * sin_ph + (2*np.pi*k) * b * cos_ph
    r_th = r_th.sum(axis=1)

    f1 = r_th * s2 + 2*np.pi * r * c2
    f2 = -r_th * c2 + 2*np.pi * r * s2
    f = np.stack([f1, f2], axis=1)             # (K,2)
    fnorm = np.linalg.norm(f, axis=1, keepdims=True)
    n = f / fnorm

    df1_dC = drth_dC * s2[:, None] + 2*np.pi * dr_dC * c2[:, None]
    df2_dC = -drth_dC * c2[:, None] + 2*np.pi * dr_dC * s2[:, None]
    df_dC = np.stack([df1_dC, df2_dC], axis=1) # (K,2,2N+1)

    I_min_nn = np.eye(2)[None, :, :] - n[:, :, None] * n[:, None, :]  # (K,2,2)
    dn_fixed = (I_min_nn @ df_dC) / fnorm[:, None, :]                 # (K,2,2N+1)
    return dn_fixed

def _batch_dc_dC_fixed(thetas: np.ndarray, C, n_all, dn_fixed, r_all=None):
    K = len(thetas)
    C_arr = np.asarray(C)
    if r_all is None:
        r_all = _radius_r_vec(thetas, C_arr)
    Nf = (len(C_arr)-1)//2

    c2 = np.cos(2 * np.pi * thetas)
    s2 = np.sin(2 * np.pi * thetas)
    e_r = np.stack([c2, s2], axis=1)
    n_dot_er = (n_all * e_r).sum(axis=1)

    # Construct dr_dC matrix (K, 2N+1) without loop
    dr_dC = np.zeros((K, len(C_arr)))
    dr_dC[:, 0] = 1.0
    if Nf > 0:
        k = np.arange(1, Nf+1)
        cos_terms = np.cos(2 * np.pi * np.outer(thetas, k))   # (K, Nf)
        sin_terms = np.sin(2 * np.pi * np.outer(thetas, k))   # (K, Nf)
        dr_dC[:, 1:Nf+1] = cos_terms
        dr_dC[:, Nf+1:2*Nf+1] = sin_terms

    e_r_dot_dn = np.einsum('ki,kij->kj', e_r, dn_fixed)
    return n_dot_er[:, None] * dr_dC + r_all[:, None] * e_r_dot_dn


# ----------------------------------------------------------------------
# Main vectorised shape gradient (with per‑hit clipping)
# ----------------------------------------------------------------------

def shape_gradient(history, betas: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Fully vectorised gradient dL/dC, with per‑hit gradient clipping
    to prevent near‑tangential events from dominating the direction.
    """
    C = history.C
    grad = np.zeros_like(C)

    for k, step in enumerate(history.steps):
        bd = step.boundary                  # BoundaryBatch (reflected particles)
        if bd.n == 0:
            continue

        # Exclude particles whose specular-reflection point was itself
        # outside Omega and got redrawn independently (BoundaryBatch.
        # was_resampled): their post-step (x, v) is a statistically
        # independent random draw, not a smooth/deterministic function of C,
        # so the closed-form d(v_tilde)/dC, d(x_tilde)/dC formulas below do
        # not apply and must not contribute to the shape gradient (mirrors
        # the same exclusion in SimulationHistory.backward_pass).
        reflect_mask = ~bd.was_resampled
        if not reflect_mask.any():
            continue

        idxs = bd.idx[reflect_mask]
        thetas = bd.theta_inter[reflect_mask]
        x_prime = bd.x_prime[reflect_mask]
        v_prime = bd.v_prime[reflect_mask]
        x_k = bd.x_k[reflect_mask]

        n_all, dn_dtheta = _batch_normal_and_dn(thetas, C)               # (K,2), (K,2)
        c_vals, F_vals = _batch_dc_c(thetas, C, n_all, dn_dtheta)       # (K,), (K,)
        dth_dC = _batch_dtheta_inter_dC(thetas, v_prime, C)              # (K, 2N+1)

        dn_fixed = _batch_dn_dC_fixed(thetas, C)                         # (K,2,2N+1)
        # ∂n/∂C_total = (∂n/∂θ) dθ/dC + dn_fixed
        dn_dC = np.einsum('ki,kj->kij', dn_dtheta, dth_dC) + dn_fixed   # (K,2,2N+1)

        dc_fixed = _batch_dc_dC_fixed(thetas, C, n_all, dn_fixed)        # (K, 2N+1)
        dc_dC = F_vals[:, None] * dth_dC + dc_fixed                      # (K, 2N+1)

        n_dot_vp = (n_all * v_prime).sum(axis=1)                         # (K,)
        n_dot_xp = (n_all * x_prime).sum(axis=1)                         # (K,)

        beta_hit = betas[k+1][idxs]      # (K,2)
        alpha_hit = alphas[k+1][idxs]    # (K,2)

        # dv_dC = -2 n_dot_vp * dn_dC - 2 n (v'^T dn_dC)
        vp_dot_dn = np.einsum('ki,kij->kj', v_prime, dn_dC)             # (K,2N+1)
        dv_dC = -2 * n_dot_vp[:, None, None] * dn_dC \
                - 2 * n_all[:, :, None] * vp_dot_dn[:, None, :]          # (K,2,2N+1)

        # dx_dC = -2 (n_dot_xp - c) * dn_dC -2 n (xp^T dn_dC) + 2 n (dc_dC)
        xp_dot_dn = np.einsum('ki,kij->kj', x_prime, dn_dC)             # (K,2N+1)
        termA = -2 * (n_dot_xp - c_vals)[:, None, None] * dn_dC
        termB = -2 * n_all[:, :, None] * xp_dot_dn[:, None, :]
        termC = 2 * n_all[:, :, None] * dc_dC[:, None, :]
        dx_dC = termA + termB + termC

        # Contribution for each hit
        contrib = -(beta_hit[:, None, :] @ dv_dC).squeeze(1) \
                  - (alpha_hit[:, None, :] @ dx_dC).squeeze(1)          # (K, 2N+1)

        # ------------------------------------------------------------------
        # Per‑hit clipping (same as original scalar version)
        # ------------------------------------------------------------------
        norms = np.linalg.norm(contrib, axis=1, keepdims=True)
        # Use np.where to avoid indexing errors
        contrib = np.where(norms > 1.0, contrib / norms, contrib)

        grad += contrib.sum(axis=0)

    return grad


# ----------------------------------------------------------------------
# Original single‑particle functions (kept for compatibility)
# ----------------------------------------------------------------------

def dr_dC(theta: float, C) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    k = np.arange(1, N + 1)
    result = np.empty(len(C))
    result[0] = 1.0
    result[1:N+1] = np.cos(2 * np.pi * k * theta)
    result[N+1:] = np.sin(2 * np.pi * k * theta)
    return result


def drtheta_dC(theta: float, C) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    k = np.arange(1, N + 1)
    result = np.zeros(len(C))
    result[1:N+1] = -2 * np.pi * k * np.sin(2 * np.pi * k * theta)
    result[N+1:] =  2 * np.pi * k * np.cos(2 * np.pi * k * theta)
    return result


def dtheta_inter_dC(theta_inter: float, v_prime, C) -> np.ndarray:
    v_prime = np.asarray(v_prime, dtype=float)
    vx, vy = v_prime
    c2 = np.cos(2 * np.pi * theta_inter)
    s2 = np.sin(2 * np.pi * theta_inter)
    r = radius_r(theta_inter, C)
    r_th = radius_r_theta(theta_inter, C)
    factor = vy * c2 - vx * s2
    denom = r_th * factor - 2 * np.pi * r * (vy * s2 + vx * c2)
    if abs(denom) < 1e-2:
        return np.zeros(len(C))
    return -dr_dC(theta_inter, C) * factor / denom


def dv_reflected_dC(theta_inter: float, v_prime, C) -> np.ndarray:
    v_prime = np.asarray(v_prime, dtype=float)
    n = normal_n(theta_inter, C)
    dndC = _dn_dC_total(theta_inter, v_prime, C)
    n_dot_v = np.dot(n, v_prime)
    return -2 * n_dot_v * dndC - 2 * np.outer(n, v_prime @ dndC)


def dx_reflected_dC(theta_inter: float, x_prime, v_prime, C) -> np.ndarray:
    x_prime = np.asarray(x_prime, dtype=float)
    n = normal_n(theta_inter, C)
    c_val = compute_c_inter(theta_inter, C)
    dndC = _dn_dC_total(theta_inter, v_prime, C)
    dcdC = _dc_dC_total(theta_inter, v_prime, C)
    n_dot_x = np.dot(n, x_prime)
    termA = -2 * (n_dot_x - c_val) * dndC
    termB = -2 * np.outer(n, x_prime @ dndC)
    termC = 2 * np.outer(n, dcdC)
    return termA + termB + termC


def area(C) -> float:
    C = np.asarray(C, dtype=float)
    return np.pi * C[0]**2 + (np.pi/2) * np.sum(C[1:]**2)


def area_gradient(C) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    grad = np.empty_like(C)
    grad[0] = 2 * np.pi * C[0]
    grad[1:] = np.pi * C[1:]
    return grad


def perimeter(C, n_quad=400) -> float:
    thetas = np.linspace(0.0, 1.0, n_quad, endpoint=False)
    norms = np.array([np.linalg.norm(gamma_prime(t, C)) for t in thetas])
    return float(norms.mean())


def perimeter_gradient(C, n_quad=400) -> np.ndarray:
    """
    Vectorized computation of perimeter gradient using Fourier series.
    """
    C = np.asarray(C, dtype=float)
    thetas = np.linspace(0.0, 1.0, n_quad, endpoint=False)
    Nf = (len(C) - 1) // 2
    k = np.arange(1, Nf + 1)

    # Precompute trigonometric terms for all thetas and all modes
    tw = 2 * np.pi * thetas[:, None] * k  # (n_quad, Nf)
    cos_tw = np.cos(tw)
    sin_tw = np.sin(tw)

    # Compute r(θ) for all thetas
    a = C[1:Nf + 1]
    b = C[Nf + 1:]
    r = C[0] + np.dot(cos_tw, a) + np.dot(sin_tw, b)  # (n_quad,)

    # Compute r_θ(θ) for all thetas
    r_th = np.dot(-2 * np.pi * k * sin_tw, a) + np.dot(2 * np.pi * k * cos_tw, b)  # (n_quad,)

    # Cosine and sine of 2πθ (not multiplied by k)
    two_pi_theta = 2 * np.pi * thetas
    c2 = np.cos(two_pi_theta)
    s2 = np.sin(two_pi_theta)

    # gamma_prime components
    gp1 = r_th * c2 - 2 * np.pi * r * s2
    gp2 = r_th * s2 + 2 * np.pi * r * c2
    gp = np.stack([gp1, gp2], axis=1)  # (n_quad,2)
    gp_norm = np.linalg.norm(gp, axis=1)  # (n_quad,)

    # Build dr_dC matrix (n_quad, len(C))
    dr_dC = np.zeros((n_quad, len(C)))
    dr_dC[:, 0] = 1.0
    if Nf > 0:
        cos_terms = np.cos(2 * np.pi * np.outer(thetas, k))  # (n_quad, Nf)
        sin_terms = np.sin(2 * np.pi * np.outer(thetas, k))  # (n_quad, Nf)
        dr_dC[:, 1:Nf + 1] = cos_terms
        dr_dC[:, Nf + 1:] = sin_terms

    # Build drtheta_dC matrix (n_quad, len(C))
    drtheta_dC = np.zeros((n_quad, len(C)))
    if Nf > 0:
        drtheta_dC[:, 1:Nf + 1] = -2 * np.pi * k * sin_terms
        drtheta_dC[:, Nf + 1:] = 2 * np.pi * k * cos_terms

    # Compute dgp1 and dgp2
    dgp1 = drtheta_dC * c2[:, None] - 2 * np.pi * dr_dC * s2[:, None]
    dgp2 = drtheta_dC * s2[:, None] + 2 * np.pi * dr_dC * c2[:, None]

    # Integrand
    integrand = (gp1[:, None] * dgp1 + gp2[:, None] * dgp2) / gp_norm[:, None]  # (n_quad, len(C))
    grad = np.sum(integrand, axis=0) / n_quad
    return grad


# ----------------------------------------------------------------------
# Perimeter-rescaling reparameterisation (Ideas, Section 7.2)
#
# Eliminates the fixed-perimeter constraint by always simulating the rescaled
# shape whose perimeter is exactly L0. We keep the SAME Fourier parameter vector
# C = [c0, a1..aN, b1..bN] used elsewhere; C plays the role of the unnormalised
# radius r̂(θ;C). The simulated ("normalised") shape has coefficients
#
#       C̃ = c(C) · C ,        c(C) = L0 / L̂(C),     L̂(C) = perimeter(C).
#
# Because the Fourier radius is linear in its coefficients, scaling the radius by
# c(C) is the same as scaling every coefficient by c(C). The perimeter is
# homogeneous of degree 1 in C, so L(C̃) = c(C) L̂(C) = L0 for every C: the
# constraint holds automatically and the optimisation over C is unconstrained.
# ----------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# Fixed-AREA constraint by rescaling (the PDF's alternative to the curvature penalty). The area
# of the star-shaped body is analytic and homogeneous of degree 2 in C:
#     A(C) = pi a0^2 + (pi/2) sum_k (a_k^2 + b_k^2),   A(sC) = s^2 A(C),   grad A . C = 2 A.
# ---------------------------------------------------------------------------------------------
def area_scale_factor(C, A0) -> float:
    """s(C) = sqrt(A0 / A(C)): the dilation that gives the shape the area A0."""
    return float(np.sqrt(float(A0) / area(C)))


def area_rescale(C, A0) -> np.ndarray:
    """C~ = s(C) C with area(C~) = A0 exactly. Feed C~ to the forward simulation."""
    C = np.asarray(C, dtype=float)
    return area_scale_factor(C, A0) * C


def rescaled_shape_gradient_area(C, A0, grad_tilde) -> np.ndarray:
    """Chain rule through the area rescaling. With J~(C) = J(s(C) C), s = (A0/A(C))^(1/2) and
    ds/dC_i = -(s / (2A)) dA/dC_i:
        dJ~/dC = s [ g - ((g . C) / (2 A)) grad A ],     g = dJ/dC~ on the rescaled shape.
    grad J~ . C = s [g.C - (g.C)/(2A) * 2A] = 0: the rescaled objective is invariant under C -> tC,
    so this gradient lives in the area-preserving subspace and vanishes at a constrained optimum."""
    C = np.asarray(C, dtype=float)
    g = np.asarray(grad_tilde, dtype=float)
    A = area(C)
    s = float(np.sqrt(float(A0) / A))
    return s * (g - (float(np.dot(g, C)) / (2.0 * A)) * area_gradient(C))


# ---------------------------------------------------------------------------------------------
# Curvature penalty  P(C) = oint_{dS_C} kappa^2 ds  (the optional regularisation of the PDF's
# objective J = I_x + lambda_R P). For the polar curve r(phi), phi = 2 pi theta in [0, 2 pi):
#     kappa = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^(3/2),     ds = (r^2 + r'^2)^(1/2) dphi,
#     P = int_0^{2pi} N^2 / D^(5/2) dphi,   N = r^2 + 2 r'^2 - r r'',   D = r^2 + r'^2
# (' = d/dphi). r, r', r'' are linear in C, so the gradient is analytic:
#     dP/dC = int [ 2 N N_C / D^(5/2) - (5/2) N^2 D_C / D^(7/2) ] dphi,
#     N_C = 2 r B + 4 r' B1 - r'' B - r B2,   D_C = 2 r B + 2 r' B1,
# with B, B1, B2 the basis (1, cos k phi, sin k phi) and its phi-derivatives. Circle of radius R:
# P = 2 pi / R. n_quad = 2000 uniform nodes (the integrand peaks with width ~1/kappa_max; shapes
# admitted by project_C/is_valid reach kappa_max ~ 600, where 400 nodes were off by up to 40%).
# ---------------------------------------------------------------------------------------------
def _polar_basis(C, n_quad=2000):
    C = np.asarray(C, dtype=float)
    Nf = (len(C) - 1) // 2
    k = np.arange(1, Nf + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False)
    kp = np.outer(phi, k)                                   # (n_quad, Nf)
    cs, sn = np.cos(kp), np.sin(kp)
    B = np.column_stack([np.ones(n_quad), cs, sn])                      # r      = B  @ C
    B1 = np.column_stack([np.zeros(n_quad), -k * sn, k * cs])           # dr/dphi = B1 @ C
    B2 = np.column_stack([np.zeros(n_quad), -k * k * cs, -k * k * sn])  # d2r/dphi2 = B2 @ C
    return phi, B, B1, B2, B @ C, B1 @ C, B2 @ C


def curvature_penalty(C, n_quad=2000) -> float:
    """P(C) = oint kappa^2 ds over the boundary of the star-shaped body (2 pi / R for a circle)."""
    _, _, _, _, r, r1, r2 = _polar_basis(C, n_quad)
    N = r * r + 2.0 * r1 * r1 - r * r2
    D = r * r + r1 * r1
    return float(2.0 * np.pi * np.mean(N * N / D ** 2.5))


def curvature_penalty_gradient(C, n_quad=2000) -> np.ndarray:
    """dP/dC (analytic, see above); checked against central differences in test_runInflow.py."""
    _, B, B1, B2, r, r1, r2 = _polar_basis(C, n_quad)
    N = r * r + 2.0 * r1 * r1 - r * r2
    D = r * r + r1 * r1
    N_C = (2.0 * r - r2)[:, None] * B + (4.0 * r1)[:, None] * B1 - r[:, None] * B2
    D_C = (2.0 * r)[:, None] * B + (2.0 * r1)[:, None] * B1
    integrand = (2.0 * N / D ** 2.5)[:, None] * N_C - (2.5 * N * N / D ** 3.5)[:, None] * D_C
    return 2.0 * np.pi * integrand.mean(axis=0)


def perimeter_scale_factor(C, L0, n_quad=400) -> float:
    """c(C) = L0 / L̂(C), where L̂(C)=perimeter(C) (Section 7.2)."""
    return float(L0) / perimeter(C, n_quad)


def perimeter_rescale(C, L0, n_quad=400) -> np.ndarray:
    """Return the normalised coefficients C̃ = c(C)·C whose shape has perimeter
    exactly L0. Feed C̃ to the forward simulation / shape_gradient."""
    C = np.asarray(C, dtype=float)
    return perimeter_scale_factor(C, L0, n_quad) * C


def rescaled_shape_gradient(C, L0, grad_tilde, n_quad=400) -> np.ndarray:
    """Chain-rule the shape gradient back through the perimeter rescaling.

    Let J̃(C) = J(C̃) with C̃ = c(C)·C and c(C)=L0/L̂(C). Then, since
    C̃_j = c(C) C_j and ∂c/∂C_i = -(c/L̂) ∂L̂/∂C_i,

        ∂J̃/∂C_i = Σ_j (∂J/∂C̃_j)(∂C̃_j/∂C_i)
                 = c (∂J/∂C̃_i) + (∂c/∂C_i) Σ_j (∂J/∂C̃_j) C_j
                 = c · g_i - (c/L̂) (g·C) (∂L̂/∂C_i),

    i.e.  ∂J̃/∂C = c [ g - ((g·C)/L̂) ∇L̂ ],

    where g = grad_tilde = ∂J/∂C̃ is the ordinary shape gradient evaluated on the
    rescaled simulation (e.g. the return value of `shape_gradient`), L̂=perimeter(C)
    and ∇L̂ = perimeter_gradient(C).

    Note ∇J̃·C = 0 (the rescaled objective is invariant under C→sC), so this
    gradient lives in the perimeter-preserving subspace and vanishes at a genuine
    constrained optimum.
    """
    C = np.asarray(C, dtype=float)
    g = np.asarray(grad_tilde, dtype=float)
    Lhat = perimeter(C, n_quad)
    c = float(L0) / Lhat
    gL = perimeter_gradient(C, n_quad)
    return c * g - (c / Lhat) * float(np.dot(g, C)) * gL


def project_step_perimeter_cap(C, direction, lr_max, project_fn, p_max, n_grid=48, tol=1e-9):
    if lr_max <= 0:
        return np.asarray(project_fn(np.asarray(C, dtype=float)), dtype=float)
    C = np.asarray(C, dtype=float)
    direction = np.asarray(direction, dtype=float)
    for j in range(n_grid + 1):
        alpha = lr_max * (1.0 - j / max(n_grid, 1))
        C_try = np.asarray(project_fn(C - alpha * direction), dtype=float)
        if perimeter(C_try) <= p_max + tol:
            return C_try
    return np.asarray(project_fn(C), dtype=float)


# ----------------------------------------------------------------------
# Internal scalar helpers for single‑particle functions
# ----------------------------------------------------------------------

def _dn_dC_total(theta_inter: float, v_prime, C) -> np.ndarray:
    dth_dC = dtheta_inter_dC(theta_inter, v_prime, C)
    dn_dth = dn_dtheta(theta_inter, C)
    dn_fix = _dn_dC_fixed_scalar(theta_inter, C)
    return np.outer(dn_dth, dth_dC) + dn_fix


def _dn_dC_fixed_scalar(theta: float, C) -> np.ndarray:
    f = f_unnormalized(theta, C)
    fnorm = np.linalg.norm(f)
    n = f / fnorm
    Nf = (len(C)-1)//2
    dr_dC_vec = dr_dC(theta, C)
    drth_dC_vec = drtheta_dC(theta, C)
    c2 = np.cos(2*np.pi*theta)
    s2 = np.sin(2*np.pi*theta)
    df1_dC = drth_dC_vec * s2 + 2*np.pi * dr_dC_vec * c2
    df2_dC = -drth_dC_vec * c2 + 2*np.pi * dr_dC_vec * s2
    df_dC = np.stack([df1_dC, df2_dC])
    I_min_nn = np.eye(2) - np.outer(n, n)
    return (I_min_nn @ df_dC) / fnorm


def _dc_dC_total(theta_inter: float, v_prime, C) -> np.ndarray:
    dth_dC = dtheta_inter_dC(theta_inter, v_prime, C)
    n = normal_n(theta_inter, C)
    dn = dn_dtheta(theta_inter, C)
    r = radius_r(theta_inter, C)
    c2 = np.cos(2*np.pi*theta_inter)
    s2 = np.sin(2*np.pi*theta_inter)
    e_r = np.array([c2, s2])
    F = (radius_r_theta(theta_inter, C) * np.dot(n, e_r)
         + r * (np.dot(e_r, dn) + 2*np.pi * np.dot(n, np.array([-s2, c2]))))
    dn_fixed = _dn_dC_fixed_scalar(theta_inter, C)
    dc_fixed = np.dot(n, e_r) * dr_dC(theta_inter, C) + r * (e_r @ dn_fixed)
    return F * dth_dC + dc_fixed


def dn_dtheta(theta: float, C) -> np.ndarray:
    f = f_unnormalized(theta, C)
    fnorm = np.linalg.norm(f)
    n = f / fnorm
    fp = f_unnormalized_prime(theta, C)
    return (fp - np.dot(n, fp) * n) / fnorm