"""
forward_pass.py – DSMC forward simulation with batch resampling for particles
that remain outside after reflection (instead of projection).
Optimised: vectorised boundary step + Numba cell assignment / rebinning with
bounding‑box pre‑check.
"""

from __future__ import annotations

import sys, os
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Path setup (unchanged)
# ---------------------------------------------------------------------------
_here    = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_here)                            # …/src/
_arb_dir = os.path.join(_src_dir, "2d", "Arbitrary Shape")   # …/src/2d/Arbitrary Shape/
for _p in (_src_dir, _arb_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygmsh
from scipy.spatial import cKDTree
import cell_class as _ct
import universal_sim_helpers as _uh
import arbitrary_helpers as _ah

from .boundary_geometry import (
    radius_r,
    normal_n,
    solve_theta_inter_batch,
    compute_c_inter,
    _radius_r_vec,
)
from .adjoint_jacobians import (
    dv_reflected_dv,
    compute_N_ki,
    collision_jacobian_transpose,
    apply_proposition_22,
    dn_dtheta,
    dtheta_dv_prime,
    dtheta_dx,
    dc_dv_prime,
    dc_dtheta_scalar,
)

# ---------------------------------------------------------------------------
# Small scalar helpers (unchanged)
# ---------------------------------------------------------------------------
def _iround(x: float, rng: np.random.Generator) -> int:
    lo   = int(np.floor(x))
    frac = x - lo
    return lo + int(rng.random() < frac)

def _is_inside(x: np.ndarray, C) -> bool:
    r_x = np.linalg.norm(x)
    if r_x == 0.0:
        return True
    theta = np.arctan2(x[1], x[0]) / (2.0 * np.pi) % 1.0
    return r_x < radius_r(theta, C)          # strict

def _is_inside_batch(x: np.ndarray, C) -> np.ndarray:
    r_x = np.linalg.norm(x, axis=1)
    thetas = (np.arctan2(x[:, 1], x[:, 0]) / (2.0 * np.pi)) % 1.0
    r_boundary = _radius_r_vec(thetas, C)
    return r_x < r_boundary                  # strict, no tolerance


def _reflect(x_prime: np.ndarray, v_prime: np.ndarray, theta_inter: float, C):
    """(Kept for reference; not used in the optimised boundary step.)"""
    n = normal_n(theta_inter, C)
    c = compute_c_inter(theta_inter, C)
    v_tilde = v_prime - 2.0 * np.dot(n, v_prime) * n
    x_tilde = x_prime - 2.0 * (np.dot(n, x_prime) - c) * n
    return x_tilde, v_tilde


# ---------------------------------------------------------------------------
# Numba‑compiled triangle search for fast cell assignment / rebin
# ---------------------------------------------------------------------------
from numba import njit

@njit
def _point_in_triangle(pt, v0, v1, v2):
    """Return True if point pt is inside triangle (including edges)."""
    d00 = (v1[0] - v0[0])*(v1[0] - v0[0]) + (v1[1] - v0[1])*(v1[1] - v0[1])
    d01 = (v1[0] - v0[0])*(v2[0] - v0[0]) + (v1[1] - v0[1])*(v2[1] - v0[1])
    d11 = (v2[0] - v0[0])*(v2[0] - v0[0]) + (v2[1] - v0[1])*(v2[1] - v0[1])
    denom = d00*d11 - d01*d01
    if abs(denom) < 1e-14:
        return False
    d20 = (pt[0] - v0[0])*(v1[0] - v0[0]) + (pt[1] - v0[1])*(v1[1] - v0[1])
    d21 = (pt[0] - v0[0])*(v2[0] - v0[0]) + (pt[1] - v0[1])*(v2[1] - v0[1])
    v = (d11*d20 - d01*d21) / denom
    w = (d00*d21 - d01*d20) / denom
    return (v >= 0.0) and (w >= 0.0) and (v + w <= 1.0)


@njit
def _walk_to_containing_cell(pos, start_cell, tri_vertices, tri_adjacency):
    """
    Walk the mesh from start_cell to the triangle that actually contains pos.
    Uses barycentric coordinates to decide which edge to cross.
    """
    current = start_cell
    for _ in range(200):   # safety limit
        v0 = tri_vertices[current, 0]
        v1 = tri_vertices[current, 1]
        v2 = tri_vertices[current, 2]
        if _point_in_triangle(pos, v0, v1, v2):
            return current

        # Barycentric coordinates w.r.t. triangle (v0,v1,v2)
        denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
        if abs(denom) < 1e-14:
            return current   # degenerate triangle, stay put
        a = ((v1[1] - v2[1]) * (pos[0] - v2[0]) + (v2[0] - v1[0]) * (pos[1] - v2[1])) / denom
        b = ((v2[1] - v0[1]) * (pos[0] - v2[0]) + (v0[0] - v2[0]) * (pos[1] - v2[1])) / denom
        c = 1.0 - a - b

        # Choose neighbour based on the negative barycentric coordinate
        if a < 0.0:
            next_cell = tri_adjacency[current, 0]   # opposite v0
        elif b < 0.0:
            next_cell = tri_adjacency[current, 1]   # opposite v1
        else:
            next_cell = tri_adjacency[current, 2]   # opposite v2 (c < 0)

        if next_cell < 0:
            return current   # no neighbour, stay (should not happen)
        current = next_cell
    return current


@njit
def _assign_cells_numba(positions, tri_centroids, tri_vertices, tri_adjacency):
    """
    For each particle, find the closest centroid cell, then walk to the true containing cell.
    """
    N = positions.shape[0]
    cell_asgn = np.zeros(N, dtype=np.int32)
    for i in range(N):
        pos = positions[i]
        # find closest centroid by simple linear scan (fine for <1000 cells)
        best_dist = 1e30
        best_cell = 0
        for c in range(tri_centroids.shape[0]):
            dx = pos[0] - tri_centroids[c, 0]
            dy = pos[1] - tri_centroids[c, 1]
            dist = dx*dx + dy*dy
            if dist < best_dist:
                best_dist = dist
                best_cell = c
        # walk from that cell to the containing one
        cell_asgn[i] = _walk_to_containing_cell(pos, best_cell, tri_vertices, tri_adjacency)
    return cell_asgn


@njit
def _rebin_numba_fast(positions, current_asgn, tri_vertices, tri_adjacency,
                      tri_centroids, tri_bbox_min, tri_bbox_max):
    """
    Rebin particles using a bounding‑box filter: only walk if the particle
    has left the bounding box of its current cell or is outside the triangle.
    """
    N = positions.shape[0]
    new_asgn = current_asgn.copy()
    for i in range(N):
        pos = positions[i]
        c = new_asgn[i]
        # Quick bounding‑box check
        if (pos[0] >= tri_bbox_min[c, 0] and pos[0] <= tri_bbox_max[c, 0] and
            pos[1] >= tri_bbox_min[c, 1] and pos[1] <= tri_bbox_max[c, 1]):
            # Inside the bbox; do a full triangle test
            v0 = tri_vertices[c, 0]
            v1 = tri_vertices[c, 1]
            v2 = tri_vertices[c, 2]
            if _point_in_triangle(pos, v0, v1, v2):
                continue   # still inside the same cell
        # Either out of bbox or not inside triangle – find the nearest centroid
        best_dist = 1e30
        best_cell = c   # start from current cell
        for nc in range(tri_centroids.shape[0]):
            dx = pos[0] - tri_centroids[nc, 0]
            dy = pos[1] - tri_centroids[nc, 1]
            dist = dx*dx + dy*dy
            if dist < best_dist:
                best_dist = dist
                best_cell = nc
        new_asgn[i] = _walk_to_containing_cell(pos, best_cell, tri_vertices, tri_adjacency)
    return new_asgn


# ---------------------------------------------------------------------------
# Batch resampling of particles inside domain (fully vectorised)
# ---------------------------------------------------------------------------
def _resample_inside_batch(C, speeds, rng):
    """
    Resample positions uniformly inside the star‑shaped domain using inversion sampling.
    For each particle, sample θ ~ Uniform(0,2π), then r = R(θ) * sqrt(u) with u ~ Uniform(0,1).
    Velocity direction is randomised uniformly (same speed).
    Returns (new_x, new_v) with shape (K,2).
    """
    K = len(speeds)
    # 1. Sample angles for position
    thetas = rng.uniform(0, 2*np.pi, K)
    theta_norm = thetas / (2*np.pi)                     # map to [0,1)
    r_max = _radius_r_vec(theta_norm, C)                # vectorised
    u = rng.uniform(0, 1, K)
    r = r_max * np.sqrt(u)
    new_x = np.column_stack([r * np.cos(thetas), r * np.sin(thetas)])
    # 2. Sample random direction for velocity (same speed)
    angles_dir = rng.uniform(0, 2*np.pi, K)
    new_v = speeds[:, None] * np.column_stack([np.cos(angles_dir), np.sin(angles_dir)])
    return new_x, new_v


# ---------------------------------------------------------------------------
# Record dataclasses (FIXED – no default factories that can swallow data)
# ---------------------------------------------------------------------------
# --- Legacy per-event record classes (kept for API back-compat; no longer used
#     internally — the forward pass now stores events columnar in the *Batch
#     classes below, which avoids creating one Python object per event). --------
@dataclass
class CollisionRecord:
    idx_i:      int
    idx_i1:     int
    v_i:        np.ndarray
    v_i1:       np.ndarray
    v_prime_i:  np.ndarray
    v_prime_i1: np.ndarray
    omega:      np.ndarray

@dataclass
class BoundaryRecord:
    idx:         int
    x_k:         np.ndarray
    v_prime:     np.ndarray
    x_prime:     np.ndarray
    in_domain:   bool
    theta_inter: Optional[float]
    x_tilde:     np.ndarray
    v_tilde:     np.ndarray
    was_resampled: bool = False


# --- Columnar event batches (one struct-of-arrays per step) ------------------
@dataclass
class CollisionBatch:
    """All accepted collisions in a step, stored columnar.

    Index pairs (idx_i, idx_i1) are disjoint within a step, so the arrays can be
    consumed with vectorised gather/scatter. v_i/v_i1 are the PRE-collision
    velocities (needed for the backward Jacobian)."""
    idx_i: np.ndarray   # (C,)  intp
    idx_i1: np.ndarray  # (C,)  intp
    v_i:   np.ndarray   # (C,2)
    v_i1:  np.ndarray   # (C,2)
    omega: np.ndarray   # (C,2)

    @property
    def n(self) -> int:
        return int(self.idx_i.shape[0])

    @classmethod
    def empty(cls) -> "CollisionBatch":
        z2 = np.empty((0, 2))
        zi = np.empty(0, dtype=np.intp)
        return cls(zi, zi, z2, z2, z2)


@dataclass
class BoundaryBatch:
    """All boundary reflections in a step, stored columnar. Every entry is an
    out-of-domain (reflected) particle; in-domain particles are not recorded.

    was_resampled[i] is True when the specular-reflection point x_tilde was
    itself still outside Omega, so _boundary_step discarded the deterministic
    reflection and redrew (x, v) independently via _resample_inside_batch.
    Those entries are NOT a deterministic function of (x_k, v_prime, x_prime)
    the way a genuine specular reflection is, so the backward pass must not
    apply the M/N/G/H reflection Jacobians to them (see backward_pass).

    invalid_idx holds the (separate) set of particles that were outside Omega
    after free flight but for which solve_theta_inter_batch found NO valid
    intersection angle at all. _boundary_step freezes those at their
    start-of-step position while leaving their (post-collision) velocity
    untouched, i.e. x_tilde=x_k, v_tilde=v' -- so d(x_tilde)/dv=0 and
    d(x_tilde)/dx=I, d(v_tilde)/dv=I, unlike the plain in-domain map
    (x_tilde=x_k+dt*v') the backward pass otherwise assumes for particles not
    in idx (see backward_pass)."""
    idx:           np.ndarray  # (R,)  intp
    x_k:           np.ndarray  # (R,2)
    v_prime:       np.ndarray  # (R,2)
    x_prime:       np.ndarray  # (R,2)
    theta_inter:   np.ndarray  # (R,)  float
    was_resampled: np.ndarray  # (R,)  bool
    invalid_idx:   np.ndarray  # (I,)  intp

    @property
    def n(self) -> int:
        return int(self.idx.shape[0])

    @classmethod
    def empty(cls) -> "BoundaryBatch":
        z2 = np.empty((0, 2))
        return cls(np.empty(0, dtype=np.intp), z2, z2, z2, np.empty(0),
                   np.empty(0, dtype=bool), np.empty(0, dtype=np.intp))


@dataclass
class StepRecord:
    """All fields are mandatory; no default factory that can discard passed values."""
    k:                int
    positions_start:  np.ndarray
    velocities_start: np.ndarray
    collisions:       CollisionBatch
    boundary:         BoundaryBatch
    positions_end:    np.ndarray
    velocities_end:   np.ndarray


# ===========================================================================
# BATCH GEOMETRY HELPERS (unchanged – used by the backward pass)
# ===========================================================================
def _batch_normal_and_dn(thetas: np.ndarray, C) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(C, dtype=float)
    K = len(thetas)
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
    C_arr = np.asarray(C)
    Nf = (len(C_arr)-1)//2
    k = np.arange(1, Nf+1)
    phases = 2 * np.pi * np.outer(thetas, k)
    a = C_arr[1:Nf+1]
    b = C_arr[Nf+1:]
    r = C_arr[0] + (a * np.cos(phases)).sum(axis=1) + (b * np.sin(phases)).sum(axis=1)
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
    C_arr = np.asarray(C)
    Nf = (len(C_arr)-1)//2
    k = np.arange(1, Nf+1)
    phases = 2 * np.pi * np.outer(thetas, k)
    a = C_arr[1:Nf+1]
    b = C_arr[Nf+1:]
    r = C_arr[0] + (a * np.cos(phases)).sum(axis=1) + (b * np.sin(phases)).sum(axis=1)
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

def vectorized_boundary_backward(
    thetas:    np.ndarray,
    x_k_arr:   np.ndarray,
    x_prime:   np.ndarray,
    v_prime:   np.ndarray,
    beta_k1:   np.ndarray,
    alpha_k1:  np.ndarray,
    C,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batched backward pass for reflected particles.
    Returns:
        rhs   = M^T β + N^T α      (to be multiplied by J^T later)
        GT_α  = G^T α
    """
    K = len(thetas)
    C = np.asarray(C, dtype=float)

    # Common geometry
    n_all, dn_all = _batch_normal_and_dn(thetas, C)
    r_all = _radius_r_vec(thetas, C)
    c_vals, F_vals = _batch_dc_c(thetas, C, n_all, dn_all, r_all)
    dth_dv = _batch_dtheta_dv(thetas, x_k_arr, v_prime, C)   # ∂θ/∂v'  (row vectors)
    dth_dx = _batch_dtheta_dx(thetas, v_prime, C)             # ∂θ/∂x   (row vectors)

    # Scalars
    n_dot_v    = (n_all * v_prime).sum(axis=1)                # ⟨n, v'⟩
    n_dot_xp   = (n_all * x_prime).sum(axis=1)                # ⟨n, x'⟩
    n_dot_beta = (n_all * beta_k1).sum(axis=1)                # n·β
    n_dot_alpha = (n_all * alpha_k1).sum(axis=1)              # n·α
    dn_dot_beta = (dn_all * beta_k1).sum(axis=1)              # dn·β
    dn_dot_alpha = (dn_all * alpha_k1).sum(axis=1)            # dn·α
    v_dot_dn   = (v_prime * dn_all).sum(axis=1)               # v'·dn
    xp_dot_dn  = (x_prime * dn_all).sum(axis=1)               # x'·dn

    # A^T β = (dn·β) ∂θ/∂v' ,   A^T α = (dn·α) ∂θ/∂v'   (row vectors)
    AT_beta  = dth_dv * dn_dot_beta[:, None]
    AT_alpha = dth_dv * dn_dot_alpha[:, None]

    # ----- M^T β (FIXED) -----
    # M^T β = β - 2⟨n,v'⟩ A^T β - 2 (n·β)[ (v'·dn) ∂θ/∂v' + n ]
    MT_beta = (beta_k1
               - 2.0 * n_dot_v[:, None] * AT_beta
               - 2.0 * n_dot_beta[:, None] * (v_dot_dn[:, None] * dth_dv + n_all))

    # ----- N^T α (FIXED) -----
    # N^T α = Δt α - 2 (n·α)[ (x'·dn) ∂θ/∂v' + Δt n - F ∂θ/∂v' ] - 2 (⟨n,x'⟩ - c) A^T α
    NT_alpha = (dt * alpha_k1
                - 2.0 * n_dot_alpha[:, None] * (xp_dot_dn[:, None] * dth_dv
                                                 + dt * n_all
                                                 - F_vals[:, None] * dth_dv)
                - 2.0 * (n_dot_xp - c_vals)[:, None] * AT_alpha)

    # ----- G^T α  (∂x̃/∂x contribution) -----
    # S = (⟨n,x'⟩-c) dn + n( x'^T dn - F )
    S = ((n_dot_xp - c_vals)[:, None] * dn_all +
         n_all * (xp_dot_dn - F_vals)[:, None])
    S_dot_alpha = (S * alpha_k1).sum(axis=1)
    GT_alpha = (alpha_k1
                - 2.0 * n_dot_alpha[:, None] * n_all        # -2 (n·α) n
                - 2.0 * S_dot_alpha[:, None] * dth_dx)      # -2 (S^T α) ∂θ/∂x

    # ----- H^T β  (∂ṽ/∂x contribution — couples β into the position adjoint) -----
    # H = ∂ṽ/∂x = -2[⟨n,v'⟩ I + n v'^T] (∂n/∂θ)(∂θ/∂x).
    # H^T β = -2 ( ⟨n,v'⟩ (dn·β) + (v'·dn) (n·β) ) ∂θ/∂x.
    HT_beta = (-2.0 * (n_dot_v * dn_dot_beta
                       + v_dot_dn * n_dot_beta)[:, None] * dth_dx)
    GT_alpha = GT_alpha + HT_beta

    # Combined right‑hand side for β
    rhs = MT_beta + NT_alpha
    return rhs, GT_alpha


# ===========================================================================
# SimulationHistory (unchanged – full backward pass)
# ===========================================================================
@dataclass
class SimulationHistory:
    C:     np.ndarray
    dt:    float
    steps: List[StepRecord]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_particles(self) -> int:
        if not self.steps:
            return 0
        return self.steps[0].positions_start.shape[0]

    @property
    def final_positions(self) -> np.ndarray:
        if not self.steps:
            raise ValueError("No steps recorded.")
        return self.steps[-1].positions_end

    @property
    def final_velocities(self) -> np.ndarray:
        if not self.steps:
            raise ValueError("No steps recorded.")
        return self.steps[-1].velocities_end

    def backward_pass(
        self,
        terminal_beta_fn:  Callable[[np.ndarray, np.ndarray], np.ndarray],
        terminal_alpha_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        M = self.n_steps
        N = self.n_particles
        C = self.C
        dt = self.dt

        betas = np.zeros((M+1, N, 2), dtype=float)
        alphas = np.zeros((M+1, N, 2), dtype=float)

        xM = self.final_positions
        vM = self.final_velocities
        betas[M] = np.asarray(terminal_beta_fn(vM, xM), dtype=float)
        alphas[M] = np.asarray(terminal_alpha_fn(vM, xM), dtype=float)

        for k in reversed(range(M)):
            step = self.steps[k]
            beta_k1 = betas[k+1]
            alpha_k1 = alphas[k+1]

            bd = step.boundary               # BoundaryBatch (reflected particles)
            out_idx = bd.idx                 # (R,) int — all out-of-domain
            # Restrict to particles that were genuinely, deterministically
            # specular-reflected. Particles whose reflection point x_tilde
            # was still outside Omega were instead redrawn independently by
            # _resample_inside_batch (see BoundaryBatch.was_resampled) -- for
            # those, (x, v) is a statistically independent random draw, not a
            # deterministic function of (x_k, v_prime, x_prime), so the
            # deterministic M/N/G/H reflection Jacobians below do not apply
            # to them and must be skipped.
            reflect_mask = ~bd.was_resampled
            refl_idx = out_idx[reflect_mask]
            resampled_idx = out_idx[bd.was_resampled]

            # In-domain particles: β_k = β_{k+1} + Δt α_{k+1}, α_k = α_{k+1}.
            alpha_k = alpha_k1.copy()
            rhs = beta_k1 + dt * alpha_k1    # correct for in-domain; overwritten below for reflected/resampled/invalid particles

            # Resampled particles (see reflect_mask comment above): the same
            # reasoning that excludes them from M/N/G/H also rules out the
            # in-domain passthrough above -- that formula assumes the
            # equally-deterministic map x_tilde=x_k+dt*v' (with Jacobians
            # d(x_tilde)/dx_k=I, d(x_tilde)/dv'=dt*I, d(v_tilde)/dv'=I), but
            # for a resampled particle x_tilde/v_tilde is an independent
            # random draw that does not causally depend on x_k or v_prime at
            # all (only |v_tilde|=|v_prime| is preserved). The correct
            # d(x_tilde)/dx_k, d(x_tilde)/dv', d(v_tilde)/dv' are all
            # (approximately) zero, so both adjoint contributions are zeroed
            # here -- mirroring shape_gradient()'s complete exclusion of
            # these same hits from the C-gradient.
            if resampled_idx.size > 0:
                rhs[resampled_idx] = 0.0
                alpha_k[resampled_idx] = 0.0

            # Particles for which no valid boundary intersection was found at
            # all (bd.invalid_idx): _boundary_step freezes them in place with
            # their post-collision velocity untouched, i.e. x_tilde=x_k (not
            # x_k+dt*v') and v_tilde=v'. That means d(x_tilde)/dv=0 (no Δt
            # term) while d(v_tilde)/dv=I and d(x_tilde)/dx=I still hold, so
            # β_k = β_{k+1} exactly (α_k is unaffected -- it already equals
            # α_{k+1}, as set above).
            inv_idx = bd.invalid_idx
            if inv_idx.size > 0:
                rhs[inv_idx] = beta_k1[inv_idx]

            if refl_idx.size > 0:
                # Single call returns BOTH the β right-hand side and G^T α + H^T β.
                batch_rhs, GT_alpha = vectorized_boundary_backward(
                    thetas=bd.theta_inter[reflect_mask],
                    x_k_arr=bd.x_k[reflect_mask],
                    x_prime=bd.x_prime[reflect_mask],
                    v_prime=bd.v_prime[reflect_mask],
                    beta_k1=beta_k1[refl_idx],
                    alpha_k1=alpha_k1[refl_idx],
                    C=C, dt=dt,
                )
                alpha_k[refl_idx] = GT_alpha
                rhs[refl_idx] = batch_rhs
            alphas[k] = alpha_k

            beta_k = rhs.copy()
            coll = step.collisions
            if coll.n > 0:
                # Vectorized application of the (transposed) collision Jacobian.
                # For each disjoint pair (i,j) with ê=(v_i-v_j)/|v_i-v_j|, ω:
                #   β_i = ½(r_i+r_j) + ½ ê (ω·(r_i-r_j))
                #   β_j = ½(r_i+r_j) - ½ ê (ω·(r_i-r_j))
                ii, jj = coll.idx_i, coll.idx_i1
                u  = coll.v_i - coll.v_i1
                un = np.linalg.norm(u, axis=1)
                ehat = np.zeros_like(u)
                safe = un > 1e-14
                ehat[safe] = u[safe] / un[safe, None]

                ri, rj = rhs[ii], rhs[jj]
                mean  = 0.5 * (ri + rj)
                coeff = 0.5 * (coll.omega * (ri - rj)).sum(axis=1)   # ½ ω·(r_i-r_j)
                s = ehat * coeff[:, None]
                beta_k[ii] = mean + s
                beta_k[jj] = mean - s
            betas[k] = beta_k

        return betas, alphas


# ===========================================================================
# ForwardSimulation (optimised)
# ===========================================================================
class ForwardSimulation:
    def __init__(
        self,
        C,
        dt: float,
        n_coll_pairs: Optional[int] = None,
        seed: Optional[int] = None,
        e: float = 1.0,
        num_boundary_points: int = 80,
        mesh_size: float = 0.3,
        collisions: bool = True,    # <-- NEW FLAG
    ):
        self.C = np.asarray(C, dtype=float)
        self.dt = float(dt)
        self._e = float(e)
        self._rng = np.random.default_rng(seed)
        self.resampled_count = 0
        self._collisions_on = collisions   # <-- STORE FLAG

        # Build triangle mesh (unchanged)
        boundary_pts = _ah.sample_star_shape(self.C, num_boundary_points)
        mesh = _ah.create_arbitrary_shape_mesh_2d(0, boundary_pts, mesh_size=mesh_size)
        self._cell_list, self._edge_to_cells = \
            _ah.create_cell_list_and_adjacency_lists(mesh)

        centroids = np.array([c.center for c in self._cell_list])
        self._centroid_kdtree = cKDTree(centroids)
        self._cell_index = {id(c): i for i, c in enumerate(self._cell_list)}

        # ---- Numba‑friendly triangle data ----
        num_cells = len(self._cell_list)
        max_adj = 3
        tri_vertices = np.zeros((num_cells, 3, 2), dtype=np.float64)
        tri_centroids = np.zeros((num_cells, 2), dtype=np.float64)
        tri_adjacency = np.full((num_cells, max_adj), -1, dtype=np.int32)
        # Bounding boxes for rebinning optimisation
        tri_bbox_min = np.zeros((num_cells, 2), dtype=np.float64)
        tri_bbox_max = np.zeros((num_cells, 2), dtype=np.float64)

        for i, cell in enumerate(self._cell_list):
            tri_vertices[i] = cell.vertices.astype(np.float64)
            tri_centroids[i] = cell.center
            minx, maxx, miny, maxy = cell.bounding_box
            tri_bbox_min[i] = (minx, miny)
            tri_bbox_max[i] = (maxx, maxy)

        # Build adjacency from shared edges (vertex‑based)
        from collections import defaultdict
        edge_to_cells_map = defaultdict(list)
        for i, verts in enumerate(tri_vertices):
            for j in range(3):
                v1 = verts[j]
                v2 = verts[(j+1)%3]
                key = ( (round(v1[0], 10), round(v1[1], 10)),
                        (round(v2[0], 10), round(v2[1], 10)) )
                if key[0] > key[1]:
                    key = (key[1], key[0])
                edge_to_cells_map[key].append(i)

        neighbours = [[] for _ in range(num_cells)]
        for cell_list in edge_to_cells_map.values():
            if len(cell_list) == 2:
                i1, i2 = cell_list
                neighbours[i1].append(i2)
                neighbours[i2].append(i1)

        for i, adj in enumerate(neighbours):
            for j, nb in enumerate(adj):
                if j >= max_adj:
                    break
                tri_adjacency[i, j] = nb

        self._tri_vertices = tri_vertices
        self._tri_centroids = tri_centroids
        self._tri_adjacency = tri_adjacency
        self._tri_bbox_min = tri_bbox_min
        self._tri_bbox_max = tri_bbox_max

    def get_resampled_count(self):
        return self.resampled_count

    # ------------------------------------------------------------------
    # Cell assignment / rebinning (now using Numba)
    # ------------------------------------------------------------------
    def _assign_cells(self, x: np.ndarray) -> np.ndarray:
        return _assign_cells_numba(
            x, self._tri_centroids, self._tri_vertices, self._tri_adjacency
        )

    def _rebin(self, x: np.ndarray, cell_asgn: np.ndarray) -> np.ndarray:
        # Use the fast rebinning with bounding‑box filter
        return _rebin_numba_fast(
            x, cell_asgn,
            self._tri_vertices, self._tri_adjacency,
            self._tri_centroids,
            self._tri_bbox_min, self._tri_bbox_max
        )

    # ------------------------------------------------------------------
    # Collision step (now with on/off switch)
    # ------------------------------------------------------------------
    def _collision_step(self, v: np.ndarray, cell_asgn: np.ndarray) -> CollisionBatch:
        if not self._collisions_on:    # <-- collisions disabled
            return CollisionBatch.empty()

        N = v.shape[0]
        if N < 2:
            return CollisionBatch.empty()
        # Per-cell accepted-collision chunks, concatenated once at the end.
        ci_chunks, cj_chunks, vi_chunks, vj_chunks, om_chunks = [], [], [], [], []
        for c_idx, cell in enumerate(self._cell_list):
            mask = np.nonzero(cell_asgn == c_idx)[0]
            n_cell = len(mask)
            if n_cell < 2:
                continue
            v_cell = v[mask]
            v_mean = v_cell.mean(axis=0)
            delta_v = np.linalg.norm(v_cell - v_mean, axis=1).max()
            ub_sigma = 2.0 * delta_v
            if ub_sigma == 0.0:
                continue
            rho_cell = n_cell / cell.area()
            expected = (n_cell * rho_cell * ub_sigma * self.dt) / (2.0 * self._e)   # original formula (no *0)
            lo = int(np.floor(expected))
            frac = expected - lo
            n_select = min(lo + int(self._rng.random() < frac), n_cell)
            if n_select < 2:
                continue
            local_perm = self._rng.permutation(n_cell)[:n_select]
            half = n_select // 2
            i_local = local_perm[:half]
            j_local = local_perm[half:2*half]
            gi = mask[i_local]
            gj = mask[j_local]
            v_i_cand = v[gi]
            v_j_cand = v[gj]
            v_rel = v_i_cand - v_j_cand
            v_rel_mag = np.linalg.norm(v_rel, axis=1)
            u_rand = self._rng.random(half) * ub_sigma
            accept = u_rand < v_rel_mag
            if not accept.any():
                continue
            gi_acc = gi[accept]
            gj_acc = gj[accept]
            n_acc = accept.sum()
            vi_arr = v[gi_acc].copy()
            vj_arr = v[gj_acc].copy()
            v_cm   = 0.5 * (vi_arr + vj_arr)
            spd    = np.linalg.norm(vi_arr - vj_arr, axis=1)
            angles = self._rng.uniform(0.0, 2*np.pi, n_acc)
            omegas = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            vp_i = v_cm + 0.5 * spd[:, None] * omegas
            vp_j = v_cm - 0.5 * spd[:, None] * omegas
            # Indices within a cell are disjoint -> vectorised in-place update.
            v[gi_acc] = vp_i
            v[gj_acc] = vp_j
            ci_chunks.append(gi_acc.astype(np.intp, copy=False))
            cj_chunks.append(gj_acc.astype(np.intp, copy=False))
            vi_chunks.append(vi_arr)   # pre-collision velocities (already copies)
            vj_chunks.append(vj_arr)
            om_chunks.append(omegas)

        if not ci_chunks:
            return CollisionBatch.empty()
        return CollisionBatch(
            idx_i=np.concatenate(ci_chunks),
            idx_i1=np.concatenate(cj_chunks),
            v_i=np.concatenate(vi_chunks),
            v_i1=np.concatenate(vj_chunks),
            omega=np.concatenate(om_chunks),
        )

    # ------------------------------------------------------------------
    # Optimised boundary step – fully vectorised reflection & resampling
    # ------------------------------------------------------------------
    def _boundary_step(self, x: np.ndarray, v: np.ndarray) -> BoundaryBatch:
        C = self.C
        dt = self.dt
        N = x.shape[0]

        x_start = x.copy()
        v_pre = v.copy()                       # velocities after collision, before update
        x_prime_all = x_start + dt * v_pre

        inside_all = _is_inside_batch(x_prime_all, C)
        outside_mask = ~inside_all
        outside_idx = np.nonzero(outside_mask)[0]
        n_out = len(outside_idx)

        # particles that stay inside: just update position
        x[inside_all] = x_prime_all[inside_all]

        if n_out == 0:
            return BoundaryBatch.empty()

        # ------------------------------------------------------------
        # 1. Intersection angles for all outside particles
        # ------------------------------------------------------------
        theta_arr = solve_theta_inter_batch(x_start[outside_idx], v_pre[outside_idx], C, dt)
        valid_mask = ~np.isnan(theta_arr)
        invalid_mask = ~valid_mask
        # Particles with no valid intersection: keep at start position (x
        # stays x_start, v stays v_pre, i.e. x_tilde=x_k, v_tilde=v'). Record
        # them so backward_pass can propagate the adjoint through the
        # correct (frozen-position) map instead of assuming the plain
        # in-domain map x_tilde=x_k+dt*v'.
        invalid_idx = outside_idx[invalid_mask]

        # Particles with no valid intersection: keep at start position
        if invalid_mask.any():
            x[outside_idx[invalid_mask]] = x_start[outside_idx[invalid_mask]]

        if valid_mask.sum() == 0:
            bb = BoundaryBatch.empty()
            bb.invalid_idx = invalid_idx.astype(np.intp, copy=True)
            return bb

        # Indices of particles that have a valid intersection
        valid_global = outside_idx[valid_mask]
        theta_valid = theta_arr[valid_mask]

        # ------------------------------------------------------------
        # 2. Batch normals and c‑values
        # ------------------------------------------------------------
        n_valid, _ = _batch_normal_and_dn(theta_valid, C)    # shape (n_valid,2)
        r_valid = _radius_r_vec(theta_valid, C)              # (n_valid,)
        tw = 2.0 * np.pi * theta_valid
        e_r = np.stack([np.cos(tw), np.sin(tw)], axis=1)    # (n_valid,2)
        c_valid = r_valid * (n_valid * e_r).sum(axis=1)     # (n_valid,)

        # ------------------------------------------------------------
        # 3. Reflect all valid particles at once
        # ------------------------------------------------------------
        x_k_valid = x_start[valid_global]
        v_prime_valid = v_pre[valid_global]
        x_prime_valid = x_prime_all[valid_global]

        n_dot_v = (n_valid * v_prime_valid).sum(axis=1, keepdims=True)
        v_tilde = v_prime_valid - 2.0 * n_dot_v * n_valid

        n_dot_xp = (n_valid * x_prime_valid).sum(axis=1, keepdims=True)
        x_tilde = x_prime_valid - 2.0 * (n_dot_xp - c_valid[:, None]) * n_valid

        # ------------------------------------------------------------
        # 4. Check which reflected points are inside
        # ------------------------------------------------------------
        inside_tilde = _is_inside_batch(x_tilde, C)
        resample_mask = ~inside_tilde
        n_resample = resample_mask.sum()

        x[valid_global[~resample_mask]] = x_tilde[~resample_mask]
        v[valid_global[~resample_mask]] = v_tilde[~resample_mask]

        if n_resample > 0:
            self.resampled_count += n_resample
            resample_global = valid_global[resample_mask]
            speeds = np.linalg.norm(v_tilde[resample_mask], axis=1)
            new_x, new_v = _resample_inside_batch(C, speeds, self._rng)
            x[resample_global] = new_x
            v[resample_global] = new_v

        # ------------------------------------------------------------
        # 5. Build the columnar boundary batch (one entry per reflected
        #    particle with a valid intersection). The reflection columns
        #    x_k_valid / v_prime_valid / x_prime_valid were already gathered.
        # ------------------------------------------------------------
        return BoundaryBatch(
            idx=valid_global.astype(np.intp, copy=True),
            x_k=x_k_valid.copy(),
            v_prime=v_prime_valid.copy(),
            x_prime=x_prime_valid.copy(),
            theta_inter=theta_valid.copy(),
            was_resampled=resample_mask.copy(),
            invalid_idx=invalid_idx.astype(np.intp, copy=True),
        )

    # ------------------------------------------------------------------
    # Run method (unchanged)
    # ------------------------------------------------------------------
    def run(
        self,
        positions:  np.ndarray,
        velocities: np.ndarray,
        n_steps:    int,
    ) -> SimulationHistory:
        x = np.asarray(positions,  dtype=float).copy()
        v = np.asarray(velocities, dtype=float).copy()
        cell_asgn = self._assign_cells(x)
        steps = []

        for k in range(n_steps):
            x_start = x.copy()
            v_start = v.copy()
            collision_records = self._collision_step(v, cell_asgn)
            boundary_records = self._boundary_step(x, v)
            cell_asgn = self._rebin(x, cell_asgn)
            steps.append(StepRecord(
                k=k,
                positions_start=x_start,
                velocities_start=v_start,
                collisions=collision_records,
                boundary=boundary_records,
                positions_end=x.copy(),
                velocities_end=v.copy(),
            ))
        return SimulationHistory(C=self.C.copy(), dt=self.dt, steps=steps)