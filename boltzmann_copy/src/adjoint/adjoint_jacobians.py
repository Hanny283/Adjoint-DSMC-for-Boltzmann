"""
Adjoint Jacobians for the space-dependent Boltzmann equation with
Fourier-parameterised reflective boundary.

Implements the following results from the paper:

    Lemma 2.6  --  ∂ñ/∂θ̃                          (dn_dtheta)
    Lemma 2.7  --  ∂θ̃/∂v'_{k,i}                   (dtheta_dv_prime)
    Lemma 2.10 --  ∂θ̃/∂x_{k,i}                    (dtheta_dx)
    Lemma 2.8  --  ∂c̃/∂v'_{k,i}                   (dc_dv_prime)
    Lemma 2.2  --  ∂ṽ_{k,i}/∂v'_{k,i}  (J=I case) (dv_reflected_dv)
    ---        --  M_{k,i} = (∂ṽ/∂v') J            (compute_M_ki)
    Lemma 2.3  --  N_{k,i}  from eq (24)            (compute_N_ki)
    Prop 2.1   --  4×4 collision Jacobian J_coll    (collision_jacobian)
               --  J_coll^T                         (collision_jacobian_transpose)
               --  one backward step for β           (apply_proposition_21)

Convention note
---------------
The matrices M_{k,i} and N_{k,i} implemented here correspond to the physical
Jacobians ∂ṽ/∂v' and ∂x̃/∂v' (Lemmas 2.2–2.3).  In the backward (adjoint)
pass the TRANSPOSED matrices M^T and N^T appear:

    β_{k,i} ← (M_{k,i} J)^T β_{k+1,i} + (N_{k,i} J)^T α_{k+1,i} + …

This is what apply_proposition_21 computes.  Equations (31)–(32) in the paper
state the same formula using M directly (without an explicit transpose), which
is consistent only when M is symmetric.  Because M is not symmetric in
general, apply_proposition_21 uses the provably correct transposed form.
"""

import numpy as np
from .boundary_geometry import (
    radius_r,
    radius_r_theta,
    f_unnormalized,
    f_unnormalized_prime,
    normal_n,
    compute_c_inter,
)


# -------------------------------------------------------------------------
# Lemma 2.6  --  ∂ñ/∂θ̃
# -------------------------------------------------------------------------

def dn_dtheta(theta: float, C) -> np.ndarray:
    """
    Derivative of the unit outward normal w.r.t. the angle parameter (Lemma 2.6).

        ∂ñ/∂θ̃ = (1/||f||) (I − ñ ñ^T) ∂f/∂θ̃

    This is a 2-vector: the derivative of the 2-D unit normal w.r.t. the
    scalar angle θ̃.

    Parameters
    ----------
    theta : float
        Angular parameter θ̃ (typically the intersection angle θ_inter).
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    f      = f_unnormalized(theta, C)        # shape (2,)
    f_norm = np.linalg.norm(f)
    n      = f / f_norm                      # unit normal ñ
    df     = f_unnormalized_prime(theta, C)  # ∂f/∂θ̃

    # (I − ñ ñ^T) df  =  df − (ñ · df) ñ
    return (df - np.dot(n, df) * n) / f_norm


# -------------------------------------------------------------------------
# Lemma 2.7  --  ∂θ̃/∂v'_{k,i}
# -------------------------------------------------------------------------

def dtheta_dv_prime(theta_inter: float, x_k, v_k, C) -> np.ndarray:
    """
    Gradient of the intersection angle θ̃ w.r.t. the post-free-flight velocity
    v'_{k,i} (Lemma 2.7).

        ∂θ̃/∂v'_{k,i} = [ r(θ̃)(sin 2πθ̃, −cos 2πθ̃) + (−x^y, x^x) ]
                         ─────────────────────────────────────────────────
                         rθ̃(v^y cos 2πθ̃ − v^x sin 2πθ̃)
                           − 2πr(θ̃)(v^y sin 2πθ̃ + v^x cos 2πθ̃)

    Returned as a shape-(2,) array (row-vector convention: ∂θ̃/∂[v^x, v^y]).

    Parameters
    ----------
    theta_inter : float
    x_k : array-like, shape (2,)  -- position (x^x, x^y)
    v_k : array-like, shape (2,)  -- velocity (v^x, v^y)
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    x_k = np.asarray(x_k, dtype=float)
    v_k = np.asarray(v_k, dtype=float)
    vx, vy = v_k[0], v_k[1]
    xx, xy = x_k[0], x_k[1]

    th   = theta_inter
    r    = radius_r(th, C)
    r_th = radius_r_theta(th, C)
    c    = np.cos(2 * np.pi * th)
    s    = np.sin(2 * np.pi * th)

    numerator   = r * np.array([s, -c]) + np.array([-xy, xx])
    denominator = r_th * (vy * c - vx * s) - 2 * np.pi * r * (vy * s + vx * c)

    if abs(denominator) < 1e-2:   # near-tangential: gradient is singular → zero out
        return np.zeros(2)
    return numerator / denominator


# -------------------------------------------------------------------------
# Lemma 2.10  --  ∂θ̃/∂x_{k,i}
# -------------------------------------------------------------------------

def dtheta_dx(theta_inter: float, v_k, C) -> np.ndarray:
    """
    Gradient of the intersection angle θ̃ w.r.t. the pre-step position x_{k,i}
    (Lemma 2.10).

        ∂θ̃/∂x_{k,i} = (v^y, −v^x)
                        ─────────────────────────────────────────────────
                        rθ̃(v^y cos 2πθ̃ − v^x sin 2πθ̃)
                          − 2πr(θ̃)(v^y sin 2πθ̃ + v^x cos 2πθ̃)

    Returned as a shape-(2,) array (column-vector convention:
    [∂θ̃/∂x^x, ∂θ̃/∂x^y]).

    Parameters
    ----------
    theta_inter : float
    v_k : array-like, shape (2,)  -- velocity (v^x, v^y)
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    v_k = np.asarray(v_k, dtype=float)
    vx, vy = v_k[0], v_k[1]

    th   = theta_inter
    r    = radius_r(th, C)
    r_th = radius_r_theta(th, C)
    c    = np.cos(2 * np.pi * th)
    s    = np.sin(2 * np.pi * th)

    numerator   = np.array([vy, -vx])
    denominator = r_th * (vy * c - vx * s) - 2 * np.pi * r * (vy * s + vx * c)

    if abs(denominator) < 1e-2:   # near-tangential: gradient is singular → zero out
        return np.zeros(2)
    return numerator / denominator


# -------------------------------------------------------------------------
# Lemma 2.8  --  ∂c̃/∂v'_{k,i}
# -------------------------------------------------------------------------

def dc_dv_prime(theta_inter: float, x_k, v_k, C) -> np.ndarray:
    """
    Gradient of the reflection constant c̃ w.r.t. v'_{k,i} (Lemma 2.8, eq 20).

        ∂c̃/∂v' = scalar_coeff · ∂θ̃/∂v'

    where

        scalar_coeff = ⟨ñ, e_θ⟩ ∂r̃/∂θ̃
                      + r̃ (e_θ · ∂ñ/∂θ̃ + 2π ñ^T e_θ⊥)

    with e_θ = (cos 2πθ̃, sin 2πθ̃)  and  e_θ⊥ = (−sin 2πθ̃, cos 2πθ̃).

    Returned as a shape-(2,) row vector.

    Parameters
    ----------
    theta_inter : float
    x_k : array-like, shape (2,)
    v_k : array-like, shape (2,)
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    th   = theta_inter
    r    = radius_r(th, C)
    r_th = radius_r_theta(th, C)
    n    = normal_n(th, C)
    dn   = dn_dtheta(th, C)

    c  = np.cos(2 * np.pi * th)
    s  = np.sin(2 * np.pi * th)
    e_theta     = np.array([c,  s])   # radial direction at θ̃
    e_theta_perp = np.array([-s, c])  # tangential direction at θ̃

    scalar = (np.dot(n, e_theta) * r_th
              + r * (np.dot(e_theta, dn) + 2 * np.pi * np.dot(n, e_theta_perp)))

    dth_dv = dtheta_dv_prime(th, x_k, v_k, C)   # shape (2,)
    return scalar * dth_dv


# -------------------------------------------------------------------------
# Lemma 2.2  --  ∂ṽ_{k,i}/∂v'_{k,i}  (with J = I)
# -------------------------------------------------------------------------

def dv_reflected_dv(theta_inter: float, x_k, v_prime, C) -> np.ndarray:
    """
    Jacobian of the reflected velocity ṽ w.r.t. the pre-reflection velocity v'
    (Lemma 2.2 with J = I, i.e. ∂ṽ/∂v').

    For a particle that hits the boundary:

        ∂ṽ/∂v' = I − 2⟨ñ, v'⟩A − 2ñ((v')^T A + ñ^T)

    where  A = outer(∂ñ/∂θ̃,  ∂θ̃/∂v').

    To obtain the full ∂ṽ_{k,i}/∂v_{k,i} (including the collision Jacobian
    J = ∂v'/∂v_{k,i}), call compute_M_ki() instead.

    Parameters
    ----------
    theta_inter : float
    x_k : array-like, shape (2,)
        Position x_{k,i} before free flight.
    v_prime : array-like, shape (2,)
        Post-free-flight velocity v'_{k,i}.
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2, 2)
        The 2×2 matrix  ∂ṽ/∂v'.
    """
    v_prime = np.asarray(v_prime, dtype=float)

    n      = normal_n(theta_inter, C)
    dn_dth = dn_dtheta(theta_inter, C)
    dth_dv = dtheta_dv_prime(theta_inter, x_k, v_prime, C)

    # A = outer(∂ñ/∂θ̃,  ∂θ̃/∂v')  shape (2,2)
    A = np.outer(dn_dth, dth_dv)

    n_dot_v   = np.dot(n, v_prime)          # scalar  ⟨ñ, v'⟩
    vT_A      = v_prime @ A                 # shape (2,) = (v')^T A
    outer_nv  = np.outer(n, vT_A + n)       # shape (2,2)

    return np.eye(2) - 2 * n_dot_v * A - 2 * outer_nv


# -------------------------------------------------------------------------
# Full M_{k,i}  (Lemma 2.2 with general J)
# -------------------------------------------------------------------------

def compute_M_ki(theta_inter: float, x_k, v_prime, C, J, in_domain: bool = False) -> np.ndarray:
    """
    Full M_{k,i} = (∂ṽ/∂v') J from Lemma 2.2 / eq (24) -- the whole
    piecewise Lemma, not just the reflected branch:

        M_{k,i} = J                                            if x'_{k,i} ∈ Ω
        M_{k,i} = [I − 2⟨ñ, v'⟩A − 2ñ((v')^T A + ñ^T)] J        otherwise

    Parameters
    ----------
    theta_inter : float or None
        Intersection angle; unused (and may be None) when in_domain=True.
    x_k : array-like, shape (2,)
    v_prime : array-like, shape (2,)
    C : array-like, shape (2N+1,)
    J : ndarray, shape (2, 2)
        ∂v'_{k,i}/∂v_{k,i}  (the appropriate block of the collision Jacobian).
        Pass np.eye(2) if no collision precedes the free flight.
    in_domain : bool, default False
        True if x'_{k,i} ∈ Ω (no reflection occurs). When True, the
        reflection geometry (theta_inter, x_k, v_prime, C) is not touched
        and J itself is returned, matching the in-domain branch of Lemma 2.2.

    Returns
    -------
    ndarray, shape (2, 2)
    """
    J = np.asarray(J, dtype=float)
    if in_domain:
        return J
    return dv_reflected_dv(theta_inter, x_k, v_prime, C) @ J


# -------------------------------------------------------------------------
# N_{k,i}  (Lemma 2.3 / eq 24)
# -------------------------------------------------------------------------

def compute_N_ki(theta_inter: float, x_prime, x_k, v_prime, C, dt: float) -> np.ndarray:
    """
    Matrix N_{k,i} from Lemma 2.3 / eq (24):

        N_{k,i} = ΔtI − 2ñ((x')^T A + Δt ñ^T − ∂c̃/∂v') − 2(⟨ñ, x'⟩ − c̃) A

    This gives  ∂x̃_{k,i}/∂v_{k,i} = N_{k,i} · J  where J = ∂v'/∂v_{k,i}.

    Parameters
    ----------
    theta_inter : float
    x_prime : array-like, shape (2,)
        Post-free-flight position x'_{k,i} (before boundary reflection).
    x_k : array-like, shape (2,)
        Pre-step position x_{k,i}.
    v_prime : array-like, shape (2,)
        Post-free-flight velocity v'_{k,i}.
    C : array-like, shape (2N+1,)
    dt : float
        Time step Δt.

    Returns
    -------
    ndarray, shape (2, 2)
    """
    x_prime = np.asarray(x_prime, dtype=float)

    n      = normal_n(theta_inter, C)
    dn_dth = dn_dtheta(theta_inter, C)
    dth_dv = dtheta_dv_prime(theta_inter, x_k, v_prime, C)
    dc_dv  = dc_dv_prime(theta_inter, x_k, v_prime, C)
    c_val  = compute_c_inter(theta_inter, C)

    A = np.outer(dn_dth, dth_dv)       # shape (2,2)

    n_dot_x = np.dot(n, x_prime)       # scalar  ⟨ñ, x'⟩

    # row vector:  (x')^T A + Δt n − ∂c̃/∂v'
    row = x_prime @ A + dt * n - dc_dv  # shape (2,)

    return dt * np.eye(2) - 2 * np.outer(n, row) - 2 * (n_dot_x - c_val) * A


# -------------------------------------------------------------------------
# Collision Jacobian for Proposition 2.1
# -------------------------------------------------------------------------

def collision_jacobian(v_i, v_j, omega) -> np.ndarray:
    """
    4×4 Jacobian of the Nanbu-Babovsky collision operator.

    The collision rule is:
        v'_i = 0.5(v_i + v_j) + 0.5|v_i − v_j| ω
        v'_j = 0.5(v_i + v_j) − 0.5|v_i − v_j| ω

    where ω ∈ S¹ is the sampled random unit vector.

    Defining ê = (v_i − v_j)/|v_i − v_j|, the 2×2 blocks are:

        ∂v'_i/∂v_i = ∂v'_j/∂v_j = 0.5(I + ω ê^T)
        ∂v'_i/∂v_j = ∂v'_j/∂v_i = 0.5(I − ω ê^T)

    Assembled with row ordering (v'_i, v'_j) and column ordering (v_i, v_j).

    Parameters
    ----------
    v_i : array-like, shape (2,)
    v_j : array-like, shape (2,)
    omega : array-like, shape (2,)
        Unit vector chosen at collision time.

    Returns
    -------
    ndarray, shape (4, 4)
        ∂(v'_i, v'_j) / ∂(v_i, v_j).
    """
    v_i   = np.asarray(v_i,   dtype=float)
    v_j   = np.asarray(v_j,   dtype=float)
    omega = np.asarray(omega, dtype=float)

    u      = v_i - v_j
    u_norm = np.linalg.norm(u)

    if u_norm < 1e-14:
        # Zero relative velocity: gradient reduces to 0.5 I blocks.
        block_plus  = 0.5 * np.eye(2)
        block_minus = 0.5 * np.eye(2)
    else:
        e_hat      = u / u_norm
        outer_oe   = np.outer(omega, e_hat)         # ω ê^T  (2×2)
        block_plus  = 0.5 * (np.eye(2) + outer_oe)  # ∂v'_i/∂v_i = ∂v'_j/∂v_j
        block_minus = 0.5 * (np.eye(2) - outer_oe)  # ∂v'_i/∂v_j = ∂v'_j/∂v_i

    J = np.zeros((4, 4))
    J[0:2, 0:2] = block_plus    # ∂v'_i/∂v_i
    J[0:2, 2:4] = block_minus   # ∂v'_i/∂v_j
    J[2:4, 0:2] = block_minus   # ∂v'_j/∂v_i
    J[2:4, 2:4] = block_plus    # ∂v'_j/∂v_j
    return J


def collision_jacobian_transpose(v_i, v_j, omega) -> np.ndarray:
    """
    Transpose of the 4×4 collision Jacobian (needed for the adjoint backward pass).

    Parameters
    ----------
    v_i, v_j : array-like, shape (2,)
    omega : array-like, shape (2,)

    Returns
    -------
    ndarray, shape (4, 4)
    """
    return collision_jacobian(v_i, v_j, omega).T


# -------------------------------------------------------------------------
# Proposition 2.1  --  one backward step for β
# -------------------------------------------------------------------------

def apply_proposition_21(
    beta_k1_i, beta_k1_i1,
    alpha_k1_i, alpha_k1_i1,
    v_i, v_j, omega,
    dt: float,
    x_prime_i, x_prime_i1,
    x_k_i, x_k_i1,
    v_prime_i, v_prime_i1,
    theta_inter_i, theta_inter_i1,
    C,
    in_domain_i: bool,
    in_domain_i1: bool,
):
    """
    One backward step of the adjoint recurrence for β (Proposition 2.1).

    Computes (β_{k,i}, β_{k,i1}) given (β_{k+1}, α_{k+1}) using all four
    boundary-hit cases:

        Case 1  both inside     :  rhs_j = β_{k+1,j} + Δt α_{k+1,j}
        Case 2  both outside    :  rhs_j = M_{k,j}^T β_{k+1,j} + N_{k,j}^T α_{k+1,j}
        Case 3  i inside, i1 out:  mix of cases 1 and 2
        Case 4  i out, i1 inside:  mix of cases 2 and 1

    Then  (β_{k,i}, β_{k,i1}) = J_coll^T · (rhs_i, rhs_i1).

    Note: M^T and N^T are used (not M, N directly) to ensure consistency
    with the expanded adjoint derivation from first principles.

    Parameters
    ----------
    beta_k1_i, beta_k1_i1 : ndarray, shape (2,)
        Velocity adjoint variables at step k+1.
    alpha_k1_i, alpha_k1_i1 : ndarray, shape (2,)
        Position adjoint variables at step k+1.
    v_i, v_j : ndarray, shape (2,)
        Pre-collision velocities v_{k,i} and v_{k,i1}.
    omega : ndarray, shape (2,)
        Random unit vector used in the DSMC collision at step k.
    dt : float
        Time step Δt.
    x_prime_i, x_prime_i1 : ndarray, shape (2,)
        Post-free-flight positions x'_{k,i}, x'_{k,i1}.
    x_k_i, x_k_i1 : ndarray, shape (2,)
        Pre-step positions x_{k,i}, x_{k,i1}.
    v_prime_i, v_prime_i1 : ndarray, shape (2,)
        Post-collision, pre-reflection velocities v'_{k,i}, v'_{k,i1}.
    theta_inter_i, theta_inter_i1 : float or None
        Intersection angles (None if particle stays inside Ω).
    C : array-like, shape (2N+1,)
        Fourier boundary coefficients.
    in_domain_i, in_domain_i1 : bool
        True if x'_{k,j} ∈ Ω (particle does NOT hit the boundary).

    Returns
    -------
    beta_k_i : ndarray, shape (2,)
    beta_k_i1 : ndarray, shape (2,)
    """
    beta_k1_i   = np.asarray(beta_k1_i,  dtype=float)
    beta_k1_i1  = np.asarray(beta_k1_i1, dtype=float)
    alpha_k1_i  = np.asarray(alpha_k1_i, dtype=float)
    alpha_k1_i1 = np.asarray(alpha_k1_i1, dtype=float)

    # Retrieve J_coll^T (4×4)
    J_coll_T = collision_jacobian_transpose(v_i, v_j, omega)

    def _rhs(in_domain, theta_inter, x_k, x_prime, v_prime, beta_k1, alpha_k1):
        """Build the right-hand-side vector for one particle."""
        if in_domain:
            # No boundary hit: identity maps
            rhs = beta_k1 + dt * alpha_k1
        else:
            # Boundary hit: use Lemma 2.2 / 2.3 matrices (transposed for adjoint)
            Mmat = dv_reflected_dv(theta_inter, x_k, v_prime, C)   # ∂ṽ/∂v'
            Nmat = compute_N_ki(theta_inter, x_prime, x_k, v_prime, C, dt)
            # Adjoint uses transposes: (Mmat J)^T β = J^T Mmat^T β
            # Here J is handled separately via J_coll_T; we only accumulate Mmat^T β
            rhs = Mmat.T @ beta_k1 + Nmat.T @ alpha_k1
        return rhs

    rhs_i  = _rhs(in_domain_i,  theta_inter_i,  x_k_i,  x_prime_i,  v_prime_i,
                  beta_k1_i,  alpha_k1_i)
    rhs_i1 = _rhs(in_domain_i1, theta_inter_i1, x_k_i1, x_prime_i1, v_prime_i1,
                  beta_k1_i1, alpha_k1_i1)

    rhs    = np.concatenate([rhs_i, rhs_i1])   # shape (4,)
    result = J_coll_T @ rhs                    # shape (4,)

    return result[:2], result[2:]


# -------------------------------------------------------------------------
# Scalar  F = ∂c̃/∂θ̃   (used in Lemma 2.9 / compute_G_ki)
# -------------------------------------------------------------------------

def dc_dtheta_scalar(theta: float, C) -> float:
    """
    Total scalar derivative of c̃ = r̃ ⟨ñ, e_θ⟩ w.r.t. θ̃ (Lemma 2.9, quantity F).

        F = ∂c̃/∂θ̃
          = (∂r̃/∂θ̃) ⟨ñ, e_θ⟩
            + r̃ (e_θ · ∂ñ/∂θ̃  +  2π ñ^T e_θ⊥)

    where  e_θ = (cos 2πθ̃, sin 2πθ̃)  and  e_θ⊥ = (−sin 2πθ̃, cos 2πθ̃).

    Parameters
    ----------
    theta : float
        Intersection angle θ̃.
    C : array-like, shape (2N+1,)

    Returns
    -------
    float
    """
    r    = radius_r(theta, C)
    r_th = radius_r_theta(theta, C)
    n    = normal_n(theta, C)
    dn   = dn_dtheta(theta, C)

    c  = np.cos(2 * np.pi * theta)
    s  = np.sin(2 * np.pi * theta)
    e_theta      = np.array([ c,  s])
    e_theta_perp = np.array([-s,  c])

    return (r_th * np.dot(n, e_theta)
            + r * (np.dot(e_theta, dn) + 2 * np.pi * np.dot(n, e_theta_perp)))


# -------------------------------------------------------------------------
# Lemma 2.9  --  G_{k,i} = ∂x̃_{k,i}/∂x_{k,i}
# -------------------------------------------------------------------------

def compute_G_ki(theta_inter: float, x_prime, v_prime, C) -> np.ndarray:
    """
    Jacobian of the reflected position x̃_{k,i} w.r.t. the pre-step position
    x_{k,i} (Lemma 2.9, eq 35).

    For a particle that hits the boundary:

        G_{k,i} = I − 2ñ ñ^T
                  − 2 outer(
                        (⟨ñ, x'⟩ − c̃) ∂ñ/∂θ̃  +  ñ ((x')^T ∂ñ/∂θ̃ − F),
                        ∂θ̃/∂x_{k,i}
                    )

    where F = ∂c̃/∂θ̃ is the scalar from dc_dtheta_scalar().

    Used in Proposition 2.2 to back-propagate the position adjoint α.

    Parameters
    ----------
    theta_inter : float
    x_prime : array-like, shape (2,)
        Post-free-flight position x'_{k,i} (before reflection).
    v_prime : array-like, shape (2,)
        Post-free-flight velocity v'_{k,i}  (trajectory direction needed for
        ∂θ̃/∂x_{k,i} via Lemma 2.10).
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2, 2)
        G_{k,i} = ∂x̃_{k,i}/∂x_{k,i}.
    """
    x_prime = np.asarray(x_prime, dtype=float)

    n        = normal_n(theta_inter, C)
    dn_dth   = dn_dtheta(theta_inter, C)
    dth_dx   = dtheta_dx(theta_inter, v_prime, C)   # shape (2,)
    c_val    = compute_c_inter(theta_inter, C)
    F        = dc_dtheta_scalar(theta_inter, C)

    n_dot_x  = np.dot(n, x_prime)              # scalar ⟨ñ, x'⟩
    x_dot_dn = np.dot(x_prime, dn_dth)         # scalar (x')^T ∂ñ/∂θ̃

    # 2-vector: (n_dot_x − c̃) ∂ñ/∂θ̃ + ñ ((x')^T ∂ñ/∂θ̃ − F)
    col = (n_dot_x - c_val) * dn_dth + n * (x_dot_dn - F)

    return np.eye(2) - 2 * np.outer(n, n) - 2 * np.outer(col, dth_dx)


# -------------------------------------------------------------------------
# H_{k,i} = ∂ṽ_{k,i}/∂x_{k,i}   (Lemma dv'dx)
# -------------------------------------------------------------------------

def compute_H_ki(theta_inter: float, v_prime, C) -> np.ndarray:
    """
    Jacobian of the reflected velocity ṽ_{k,i} w.r.t. the pre-step position
    x_{k,i} (Lemma dv'dx, paper lines 764-778).

    J2 depends on x_{k,i} whenever a reflection occurs, because the
    reflected velocity ṽ_{k,i} = v'_{k,i} - 2⟨ñ,v'_{k,i}⟩ñ involves the
    boundary normal ñ = ñ(θ̃) at the intersection point, and θ̃ = θ̃(x_{k,i},
    v'_{k,i}) depends on the starting position x_{k,i}. Proposition 2.2
    therefore needs H_{k,i} IN ADDITION to G_{k,i} = ∂x̃/∂x whenever the
    particle reflects.

        H_{k,i} = ∂ṽ/∂x_{k,i}
                = −2[⟨ñ, v'⟩ I + ñ (v')^T] outer(∂ñ/∂θ̃, ∂θ̃/∂x_{k,i})

    Parameters
    ----------
    theta_inter : float
    v_prime : array-like, shape (2,)
        Post-free-flight velocity v'_{k,i}.
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2, 2)
        H_{k,i} = ∂ṽ_{k,i}/∂x_{k,i}.
    """
    v_prime = np.asarray(v_prime, dtype=float)

    n      = normal_n(theta_inter, C)
    dn_dth = dn_dtheta(theta_inter, C)
    dth_dx = dtheta_dx(theta_inter, v_prime, C)

    n_dot_v = np.dot(n, v_prime)
    bracket = n_dot_v * np.eye(2) + np.outer(n, v_prime)   # ⟨ñ,v'⟩I + ñ(v')^T
    return -2.0 * bracket @ np.outer(dn_dth, dth_dx)


# -------------------------------------------------------------------------
# Proposition 2.2  --  one backward step for α
# -------------------------------------------------------------------------

def apply_proposition_22(
    alpha_k1: np.ndarray,
    beta_k1: np.ndarray,
    theta_inter,
    x_prime,
    v_k,
    C,
    in_domain: bool,
) -> np.ndarray:
    """
    One backward step of the adjoint recurrence for α (Proposition 2.2).

        α_{k,i} = G_{k,i}^T α_{k+1,i} + H_{k,i}^T β_{k+1,i}    if x'_{k,i} ∉ Ω
        α_{k,i} = α_{k+1,i}                                     if x'_{k,i} ∈ Ω

    The H_{k,i}^T β_{k+1,i} coupling term is required whenever the particle
    reflects: J2 (hence β) feeds back into the position adjoint through the
    boundary normal at the intersection point (see compute_H_ki). Dropping
    it (as an earlier version of this function did) silently produces the
    wrong α whenever β_{k+1,i} is nonzero. This mirrors what
    forward_pass.vectorized_boundary_backward computes inline
    (GT_alpha + HT_beta).

    Parameters
    ----------
    alpha_k1 : ndarray, shape (2,)
        Position adjoint at step k+1.
    beta_k1 : ndarray, shape (2,)
        Velocity adjoint at step k+1 (needed for the H^T β coupling term
        whenever the particle reflects).
    theta_inter : float or None
        Intersection angle from the forward pass; None if in_domain.
    x_prime : array-like, shape (2,)
        Post-free-flight position x'_{k,i}.
    v_k : array-like, shape (2,)
        Velocity used for free flight v'_{k,i}
        (needed to compute ∂θ̃/∂x_{k,i} inside compute_G_ki / compute_H_ki).
    C : array-like, shape (2N+1,)
    in_domain : bool
        True if x'_{k,i} ∈ Ω.

    Returns
    -------
    ndarray, shape (2,)
        α_{k,i}.
    """
    if in_domain:
        return np.asarray(alpha_k1, dtype=float).copy()

    G = compute_G_ki(theta_inter, x_prime, v_k, C)
    H = compute_H_ki(theta_inter, v_k, C)
    return G.T @ np.asarray(alpha_k1, dtype=float) + H.T @ np.asarray(beta_k1, dtype=float)
