"""
compare_adjoint_fd.py — fast deterministic sanity check across shapes.
=====================================================================
A quick smoke test (single seed each, collisions OFF, FD per component) over a
handful of shape/particle configurations: a circle, a wavy boundary, and a
multi-particle wavy case. Prints adjoint vs FD with cosine + relative error and
PASS/FAIL for each. Use this for a fast "is the gradient still correct?" check;
use TEST.py for the seed-averaged statistics.

Run:
    python compare_adjoint_fd.py
"""
import numpy as np
from gradient_test_utils import (
    PositionLoss, VelocityComponentLoss, VelocityNearOriginLoss,
    adjoint_gradient, fd_gradient, sample_inside, cosine,
)

DT = 0.10
H = 1e-6
PASS_COS, PASS_REL = 0.999, 1e-2


def check(name, C, loss, *, n_steps, n_particles, seed=0, fill=0.8):
    C = np.asarray(C, float)
    rng = np.random.default_rng(seed)
    x0, v0 = sample_inside(C, n_particles, rng, fill=fill)
    L, ga = adjoint_gradient(C, x0, v0, loss, DT, n_steps, collisions=False, seed=seed)
    gf = fd_gradient(C, x0, v0, loss, DT, n_steps, collisions=False, seed=seed, H=H)
    cos = cosine(ga, gf)
    rel = np.linalg.norm(ga - gf) / (np.linalg.norm(gf) + 1e-300)
    ok = (cos > PASS_COS) and (rel < PASS_REL)
    print(f"\n=== {name}   (L={L:.6g}) ===")
    print("  adjoint:", np.array2string(ga, precision=5))
    print("  FD     :", np.array2string(gf, precision=5))
    print(f"  cosine={cos:.6f}  rel={rel:.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("Deterministic adjoint-vs-FD sanity check (collisions OFF, FD per component)")
    results = []

    # A: circle, velocity-near-origin loss
    results.append(check(
        "A: circle, velocity loss",
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        VelocityNearOriginLoss(), n_steps=3, n_particles=20_000))

    # B: wavy boundary, position loss (alpha-only)
    results.append(check(
        "B: wavy boundary, alpha-only (position)",
        [1.0, 0.15, -0.10, 0.05, 0.08, 0.03, -0.04],
        PositionLoss(), n_steps=3, n_particles=20_000))

    # C: wavy boundary, beta-only (velocity component) -- stresses the H term
    results.append(check(
        "C: wavy boundary, beta-only (v_x^2)",
        [1.0, 0.20, -0.12, 0.06, 0.10, 0.04, -0.05],
        VelocityComponentLoss(), n_steps=4, n_particles=30_000))

    # D: wavy boundary, mixed (velocity-near-origin), more steps
    results.append(check(
        "D: wavy boundary, mixed, 5 steps",
        [1.0, 0.20, -0.12, 0.06, 0.10, 0.04, -0.05],
        VelocityNearOriginLoss(), n_steps=5, n_particles=30_000))

    print("\n" + ("ALL PASS" if all(results) else "SOME FAILED"))


if __name__ == "__main__":
    main()
