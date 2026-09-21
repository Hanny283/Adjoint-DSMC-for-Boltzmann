"""
TEST.py — Adjoint vs finite-difference shape gradient, averaged over seeds.
=========================================================================
Primary validation test. It runs the comparison for EACH class of objective:

    * alpha-only : L depends only on final position   (PositionLoss)
    * beta-only  : L depends only on final velocity    (VelocityComponentLoss)
    * mixed      : L depends on both position & velocity (VelocityNearOriginLoss)

so all three adjoint paths are exercised. For each loss and each of several seeds
it draws an independent particle cloud, computes the ADJOINT gradient and the
FINITE-DIFFERENCE gradient on the SAME cloud (FD perturbs each component of C
separately, particles fixed), averages over seeds, and reports per-component
mean +/- std, the cosine similarity / relative error of the mean gradients, and
PASS/FAIL.

Run:
    python TEST.py
"""
import numpy as np
from gradient_test_utils import (
    PositionLoss, VelocityComponentLoss, VelocityNearOriginLoss,
    seed_averaged_comparison, robust_mean_gradients, cosine, coeff_names,
)

# ============================ CONFIG ===================================
C          = np.array([1.0, 0.15, -0.10, 0.05, 0.08, 0.03, -0.04])  # N_FOURIER = 3
SEEDS      = list(range(8))
N          = 50_000
DT         = 0.10
N_STEPS    = 4
COLLISIONS = False          # FD is only valid on a deterministic forward
H          = 1e-6
RESCALE    = False          # Section 7.2 fixed-perimeter option
PASS_COS   = 0.999
PASS_REL   = 1e-2

LOSSES = [
    ("alpha-only", PositionLoss()),
    ("beta-only",  VelocityComponentLoss()),
    ("mixed",      VelocityNearOriginLoss()),
]
# ======================================================================


def run_one(kind, loss):
    print("\n" + "=" * 70)
    print(f"[{kind}]  {loss.name}")
    print("-" * 70)
    res = seed_averaged_comparison(
        C, loss, seeds=SEEDS, N=N, dt=DT, n_steps=N_STEPS,
        collisions=COLLISIONS, H=H, rescale=RESCALE,
    )
    adj, fd = res["adj"], res["fd"]
    # Robust aggregation: drop seeds where the FD norm explodes (a particle
    # crossed the reflect/no-reflect threshold within +/-H -> FD spike; the
    # adjoint is unaffected). Needed mainly for velocity-type (beta) losses.
    mean_adj, mean_fd, keep = robust_mean_gradients(res)
    n_trim = int((~keep).sum())
    std_adj, std_fd = adj[keep].std(0), fd[keep].std(0)

    print(f"mean loss = {res['L'].mean():.6f} (+/- {res['L'].std():.2e})    "
          f"per-seed cos: " + " ".join(f"{c:.4f}" for c in res["per_seed_cos"]))
    if n_trim:
        print(f"  (trimmed {n_trim} FD-outlier seed(s); adjoint norms there are normal)")
    print(f"  {'coef':4s} {'adjoint':>22s} {'finite-diff':>22s} {'rel.diff':>10s}")
    for name, ma, sa, mf, sf in zip(coeff_names(C), mean_adj, std_adj, mean_fd, std_fd):
        rel = abs(ma - mf) / (abs(mf) + 1e-12)
        print(f"  {name:4s} {ma:11.5f}+-{sa:8.5f} {mf:11.5f}+-{sf:8.5f} {rel:10.2e}")

    cos = cosine(mean_adj, mean_fd)
    rel = np.linalg.norm(mean_adj - mean_fd) / (np.linalg.norm(mean_fd) + 1e-300)
    ok = (cos > PASS_COS) and (rel < PASS_REL)
    print(f"  --> cosine={cos:.6f}  rel={rel:.3e}  (trimmed {n_trim})  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("Adjoint vs finite-difference shape gradient (seed-averaged)")
    print(f"shape C = {np.array2string(C, precision=3)}")
    print(f"seeds={len(SEEDS)} N={N} dt={DT} steps={N_STEPS} "
          f"collisions={COLLISIONS} H={H:g} rescale={RESCALE}")
    results = {kind: run_one(kind, loss) for kind, loss in LOSSES}
    print("\n" + "=" * 70)
    print("SUMMARY: " + "   ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    print("ALL PASS" if all(results.values()) else "SOME FAILED")


if __name__ == "__main__":
    main()
