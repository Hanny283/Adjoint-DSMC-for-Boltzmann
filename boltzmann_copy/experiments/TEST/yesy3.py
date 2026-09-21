"""
TEST_beta_multiseed_components.py – Per‑coefficient mean/variance across seeds
for the velocity‑only adjoint gradient (beta back‑propagation through boundary).
"""
import sys, os
import numpy as np
from joblib import Parallel, delayed

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from adjoint import ForwardSimulation
from adjoint.shape_gradient import shape_gradient
from adjoint.boundary_geometry import _radius_r_vec

# =============================================================================
# Loss L = (1/N) Σ v_x^2   (depends only on velocity, no position)
# =============================================================================
def compute_L(history):
    vx = history.final_velocities[:, 0]
    return np.mean(vx ** 2)

def term_beta(v, x):
    N = len(v)
    # ∂L/∂v = (2/N)*(v_x, 0) → beta = -∂L/∂v
    return -(2.0 / N) * np.column_stack([v[:, 0], np.zeros(N)])

def term_alpha(v, x):
    return np.zeros_like(x)            # no position contribution

# =============================================================================
# Uniform sampling inside domain
# =============================================================================
def sample_particles_inside(C, N, rng):
    thetas = rng.uniform(0, 2*np.pi, N)
    angles_norm = thetas / (2.0 * np.pi)
    r_max = _radius_r_vec(angles_norm, C)
    u = rng.uniform(0, 1, N)
    r = r_max * np.sqrt(u) * 0.8      # stay away from boundary
    positions = np.column_stack([r * np.cos(thetas), r * np.sin(thetas)])
    velocities = rng.standard_normal((N, 2))
    return positions, velocities

# =============================================================================
# Workers (collisions OFF, beta only)
# =============================================================================
def adj_worker(seed, C, x0, v0, DT, N_STEPS, BIRD_E):
    sim = ForwardSimulation(C, DT, seed=seed, e=BIRD_E, collisions=False)
    hist = sim.run(x0.copy(), v0.copy(), N_STEPS)
    betas, alphas = hist.backward_pass(term_beta, term_alpha)
    grad = shape_gradient(hist, betas, alphas)
    return grad, sim.resampled_count

def fd_worker(seed, k, C, H, x0, v0, DT, N_STEPS, BIRD_E):
    C_plus  = C.copy();  C_plus[k] += H
    C_minus = C.copy();  C_minus[k] -= H

    sim_plus  = ForwardSimulation(C_plus,  DT, seed=seed, e=BIRD_E, collisions=False)
    hist_plus = sim_plus.run(x0.copy(), v0.copy(), N_STEPS)
    L_plus = compute_L(hist_plus)

    sim_minus  = ForwardSimulation(C_minus, DT, seed=seed, e=BIRD_E, collisions=False)
    hist_minus = sim_minus.run(x0.copy(), v0.copy(), N_STEPS)
    L_minus = compute_L(hist_minus)

    return (L_plus - L_minus) / (2 * H)

# =============================================================================
# Parameters
# =============================================================================
C = np.array([1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])   # circle radius ~1, small a1
DT = 0.1
N_STEPS = 3
BIRD_E = 10.0
N_PARTICLES = 1_000_000
H = 1e-3                     # finite difference step

# Fixed initial particles
rng_init = np.random.default_rng(1234)
x0_fixed, v0_fixed = sample_particles_inside(C, N_PARTICLES, rng_init)

# Seeds
seeds = [1000 + i for i in range(5)]

# =============================================================================
# 1. Adjoint gradients
# =============================================================================
print("Computing adjoint gradients (beta only, collisions off) ...")
adj_results = Parallel(n_jobs=-1, backend='loky')(
    delayed(adj_worker)(seed, C, x0_fixed, v0_fixed, DT, N_STEPS, BIRD_E)
    for seed in seeds
)
adj_grads = np.array([r[0] for r in adj_results])
resampled_counts = [r[1] for r in adj_results]
print(f"Resampled counts: {resampled_counts}")

# =============================================================================
# 2. FD gradients
# =============================================================================
print("Computing FD gradients ...")
tasks = []
for seed in seeds:
    for k in range(len(C)):
        tasks.append(delayed(fd_worker)(seed, k, C, H, x0_fixed, v0_fixed, DT, N_STEPS, BIRD_E))

fd_estimates = Parallel(n_jobs=-1, backend='loky')(tasks)

fd_grads = np.zeros((len(seeds), len(C)))
idx = 0
for i in range(len(seeds)):
    for k in range(len(C)):
        fd_grads[i, k] = fd_estimates[idx]
        idx += 1

# =============================================================================
# 3. Cosine similarity
# =============================================================================
print("\nPer‑seed cosine similarity:")
for i, seed in enumerate(seeds):
    a = adj_grads[i]
    b = fd_grads[i]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    print(f"  Seed {seed}: {cos:.8f}")

mean_adj = np.mean(adj_grads, axis=0)
mean_fd  = np.mean(fd_grads, axis=0)
cos_mean = np.dot(mean_adj, mean_fd) / (np.linalg.norm(mean_adj) * np.linalg.norm(mean_fd) + 1e-30)
print(f"\nCosine similarity between mean gradients: {cos_mean:.8f}")

# =============================================================================
# 4. Per‑coefficient comparison
# =============================================================================
coeff_names = ['c0','a1','a2','a3','b1','b2','b3']
print("\n--- Per‑coefficient mean ± std (across seeds) ---")
for k, name in enumerate(coeff_names):
    adj_col = adj_grads[:, k]
    fd_col  = fd_grads[:, k]
    mean_adj = np.mean(adj_col)
    mean_fd  = np.mean(fd_col)
    std_adj  = np.std(adj_col)
    std_fd   = np.std(fd_col)
    rel_diff = np.abs(mean_adj - mean_fd) / (np.abs(mean_fd) + 1e-30)
    print(f"{name:3s}: adj={mean_adj:12.6f} ± {std_adj:.6f}, "
          f"FD={mean_fd:12.6f} ± {std_fd:.6f}, "
          f"rel_diff={rel_diff:.3e}")