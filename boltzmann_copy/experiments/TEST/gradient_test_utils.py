"""
gradient_test_utils.py
======================
Shared utilities for validating the DSMC shape gradient against finite
differences (FD). Imported by TEST.py, test2.py and compare_adjoint_fd.py.

Design choices (these matter for a *valid* comparison):

* FD is computed ONE COMPONENT OF C AT A TIME (central difference on C[i] only).
* Within a single seed the initial particles (x0, v0) are sampled ONCE and reused
  for the base, +H and -H evaluations. This isolates the *shape* gradient — the
  same quantity the adjoint computes — instead of also picking up the dependence
  of the initial sampling on the (shape-dependent) domain.
* Averaging is over several seeds: each seed draws an independent particle cloud,
  so the mean over seeds is a Monte-Carlo estimate of the gradient and the spread
  over seeds is its sampling error.
* Default H = 1e-6. Finite differences on this system are step-sensitive because
  particles can flip between reflecting / not-reflecting as C changes (this makes
  L(C) only piecewise smooth); H ~ 1e-6 sits below that and above round-off.
* Default collisions = OFF. FD requires a deterministic, smooth L(C); with
  collisions ON, perturbing C reshuffles the random collision pairing and L(C)
  becomes noisy, so FD is unreliable (use collisions ON only for illustration).
  The collision Jacobian itself is verified separately to machine precision.
"""
import os
import sys
import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from adjoint import ForwardSimulation, shape_gradient                       # noqa: E402
from adjoint import perimeter, perimeter_rescale, rescaled_shape_gradient   # noqa: E402
from adjoint.boundary_geometry import _radius_r_vec                          # noqa: E402


# ---------------------------------------------------------------------------
# Loss functions.  Each loss provides:
#   L(history)          -> scalar objective
#   term_beta(v, x)     -> terminal velocity adjoint  beta_M = -d g / d v
#   term_alpha(v, x)    -> terminal position adjoint   alpha_M = -d g / d x
# Sign conventions match the adjoint derivation (verified against FD).
# ---------------------------------------------------------------------------
class PositionLoss:
    """L = (1/N) sum_i |x_i(T)|^2  (position only; exercises the alpha path)."""
    name = "position  L=mean|x|^2"

    def L(self, h):
        return float(np.mean(np.sum(h.final_positions ** 2, axis=1)))

    def term_beta(self, v, x):
        return np.zeros_like(v)

    def term_alpha(self, v, x):
        return -2.0 * x / len(x)


class VelocityComponentLoss:
    """L = (1/N) sum_i v_{x,i}(T)^2  (velocity only; term_alpha = 0).

    Depends on the final velocity but NOT on position, so the terminal adjoint is
    purely beta. Because specular reflection conserves speed, a single velocity
    *component* (not |v|) is used so the objective genuinely depends on C. This is
    the strongest test of the dv~/dx coupling (the H term): alpha is produced only
    via H^T beta during back-propagation."""
    name = "velocity-component  L=mean(v_x^2)"

    def L(self, h):
        return float(np.mean(h.final_velocities[:, 0] ** 2))

    def term_beta(self, v, x):
        g = np.zeros_like(v)
        g[:, 0] = -(2.0 / len(v)) * v[:, 0]
        return g

    def term_alpha(self, v, x):
        return np.zeros_like(x)


class VelocityNearOriginLoss:
    """L = lambda * (1/N) sum_i |v_i(T)|^2 * phi(|x_i(T)|), with a smooth
    indicator phi concentrated near the origin (exercises both beta and alpha)."""
    name = "velocity-near-origin  L=mean(|v|^2 phi(|x|))"

    def __init__(self, lam=10.0, R=0.1, k=50.0, a=1.0):
        self.lam, self.R, self.k, self.a = lam, R, k, a

    def _phi(self, d):
        return 1.0 / (1.0 + np.exp(self.k * self.a * (d ** 2 - self.R ** 2)))

    def _dphi_dd(self, d):
        phi = self._phi(d)
        return -2.0 * self.k * self.a * d * phi * (1.0 - phi)

    def L(self, h):
        x, v = h.final_positions, h.final_velocities
        phi = self._phi(np.linalg.norm(x, axis=1))
        return self.lam * float(np.mean(np.sum(v ** 2, axis=1) * phi))

    def term_beta(self, v, x):
        d = np.linalg.norm(x, axis=1)
        phi = self._phi(d)
        return -(2.0 * self.lam / len(v)) * v * phi[:, None]

    def term_alpha(self, v, x):
        d = np.linalg.norm(x, axis=1)
        dphi = self._dphi_dd(d)
        d_safe = np.where(d < 1e-12, 1.0, d)
        direction = x / d_safe[:, None]
        v2 = np.sum(v ** 2, axis=1)
        return -(self.lam / len(x)) * v2[:, None] * dphi[:, None] * direction


# ---------------------------------------------------------------------------
# Particle sampling
# ---------------------------------------------------------------------------
def sample_inside(C, N, rng, fill=0.8):
    """Sample N particles uniformly inside the star-shaped domain defined by C.
    `fill` (<=1) keeps them strictly inside so small C perturbations stay valid."""
    C = np.asarray(C, dtype=float)
    thetas = rng.uniform(0.0, 2.0 * np.pi, N)
    r_max = _radius_r_vec(thetas / (2.0 * np.pi), C)
    r = r_max * np.sqrt(rng.uniform(0.0, 1.0, N)) * fill
    x = np.column_stack([r * np.cos(thetas), r * np.sin(thetas)])
    v = rng.standard_normal((N, 2))
    return x, v


# ---------------------------------------------------------------------------
# Gradients for a SINGLE seed / fixed particle cloud
# ---------------------------------------------------------------------------
def _effective(C, rescale, L0):
    return perimeter_rescale(C, L0) if rescale else np.asarray(C, dtype=float)


def adjoint_gradient(C, x0, v0, loss, dt, n_steps, *, collisions=False,
                     seed=0, rescale=False, L0=None):
    """Adjoint shape gradient at C for the fixed particles (x0, v0)."""
    C_sim = _effective(C, rescale, L0)
    sim = ForwardSimulation(C_sim, dt, seed=seed, collisions=collisions)
    hist = sim.run(x0.copy(), v0.copy(), n_steps)
    betas, alphas = hist.backward_pass(loss.term_beta, loss.term_alpha)
    g = shape_gradient(hist, betas, alphas)
    if rescale:
        g = rescaled_shape_gradient(C, L0, g)
    return loss.L(hist), g


def fd_gradient(C, x0, v0, loss, dt, n_steps, *, collisions=False, seed=0,
                H=1e-6, rescale=False, L0=None):
    """Central-difference gradient, computed ONE COMPONENT OF C AT A TIME,
    reusing the fixed particles (x0, v0) for every evaluation."""
    C = np.asarray(C, dtype=float)
    g = np.zeros_like(C)

    def Lval(Cx):
        C_sim = _effective(Cx, rescale, L0)
        sim = ForwardSimulation(C_sim, dt, seed=seed, collisions=collisions)
        return loss.L(sim.run(x0.copy(), v0.copy(), n_steps))

    for i in range(len(C)):                 # <-- each component separately
        Cp = C.copy(); Cp[i] += H
        Cm = C.copy(); Cm[i] -= H
        g[i] = (Lval(Cp) - Lval(Cm)) / (2.0 * H)
    return g


# ---------------------------------------------------------------------------
# Seed-averaged comparison
# ---------------------------------------------------------------------------
def seed_averaged_comparison(C, loss, *, seeds, N, dt, n_steps,
                             collisions=False, H=1e-6, fill=0.8,
                             rescale=False, L0=None):
    """For each seed: draw an independent particle cloud, compute the adjoint and
    the per-component FD gradient on that SAME cloud. Returns stacked arrays.

    Returns dict with keys:
        adj  : (n_seeds, nC) adjoint gradients
        fd   : (n_seeds, nC) finite-difference gradients
        L    : (n_seeds,)    loss values
        per_seed_cos : (n_seeds,) cosine(adj, fd) per seed
    """
    C = np.asarray(C, dtype=float)
    if rescale and L0 is None:
        L0 = perimeter(C)
    adj, fd, Ls, coss = [], [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        C_cloud = _effective(C, rescale, L0)      # sample inside the simulated shape
        x0, v0 = sample_inside(C_cloud, N, rng, fill=fill)
        L_val, ga = adjoint_gradient(C, x0, v0, loss, dt, n_steps,
                                     collisions=collisions, seed=s,
                                     rescale=rescale, L0=L0)
        gf = fd_gradient(C, x0, v0, loss, dt, n_steps,
                         collisions=collisions, seed=s, H=H,
                         rescale=rescale, L0=L0)
        adj.append(ga); fd.append(gf); Ls.append(L_val)
        denom = np.linalg.norm(ga) * np.linalg.norm(gf) + 1e-300
        coss.append(float(np.dot(ga, gf) / denom))
    return {
        "adj": np.array(adj),
        "fd": np.array(fd),
        "L": np.array(Ls),
        "per_seed_cos": np.array(coss),
    }


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def robust_mean_gradients(res, fd_outlier_factor=5.0):
    """Average the per-seed gradients, dropping seeds whose FINITE-DIFFERENCE
    norm is a gross outlier. Such spikes occur when a particle crosses the
    reflect/no-reflect threshold inside [C-H, C+H]: for velocity-type losses this
    jumps the loss by O(1) and the FD derivative by ~O(1/H), while the (smooth)
    adjoint stays well-behaved. We therefore trim on the FD norm only.

    Returns (mean_adj_clean, mean_fd_clean, kept_mask).
    """
    adj, fd = res["adj"], res["fd"]
    fdn = np.linalg.norm(fd, axis=1)
    med = np.median(fdn)
    keep = fdn <= fd_outlier_factor * med if med > 0 else np.ones(len(fdn), bool)
    if not keep.any():
        keep = np.ones(len(fdn), bool)
    return adj[keep].mean(0), fd[keep].mean(0), keep


def coeff_names(C):
    Nf = (len(C) - 1) // 2
    return ["c0"] + [f"a{k}" for k in range(1, Nf + 1)] + [f"b{k}" for k in range(1, Nf + 1)]
