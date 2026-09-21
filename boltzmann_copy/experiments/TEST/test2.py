"""
test2.py — FD step-size robustness and per-component breakdown.
==============================================================
Two things, both seed-averaged:

  (A) Sweep the finite-difference step H and report cosine(adjoint, FD).
      Finite differences on this system are step-sensitive: particles can flip
      between reflecting / not-reflecting as C changes, so L(C) is only
      piecewise smooth. Too-large H straddles those jumps; too-small H hits
      round-off. This sweep shows the usable window (typically H ~ 1e-6).

  (B) At the best H, print the per-component adjoint vs FD gradient with the
      mean and standard deviation across seeds.

Run:
    python test2.py
"""
import numpy as np
from gradient_test_utils import (
    PositionLoss, VelocityNearOriginLoss,
    seed_averaged_comparison, cosine, coeff_names,
)

# ============================ CONFIG ===================================
C          = np.array([0.9, 0.20, -0.12, 0.06, 0.10, 0.04, -0.05])
LOSS       = PositionLoss()                 # position-only -> stresses the alpha path
SEEDS      = list(range(6))
N          = 60_000
DT         = 0.10
N_STEPS    = 3
COLLISIONS = False
H_LIST     = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# ======================================================================


def main():
    print("=" * 70)
    print("FD step-size robustness  +  per-component breakdown (seed-averaged)")
    print(f"  loss={LOSS.name}   seeds={len(SEEDS)}   N={N}   "
          f"dt={DT} steps={N_STEPS} collisions={COLLISIONS}")
    print("=" * 70)

    # (A) sweep H ------------------------------------------------------
    print("\n(A) cosine(mean adjoint, mean FD) vs FD step H:")
    best = (None, -2.0)
    cache = {}
    for H in H_LIST:
        res = seed_averaged_comparison(C, LOSS, seeds=SEEDS, N=N, dt=DT,
                                       n_steps=N_STEPS, collisions=COLLISIONS, H=H)
        cache[H] = res
        cos = cosine(res["adj"].mean(0), res["fd"].mean(0))
        rel = np.linalg.norm(res["adj"].mean(0) - res["fd"].mean(0)) / \
            (np.linalg.norm(res["fd"].mean(0)) + 1e-300)
        flag = ""
        if cos > best[1]:
            best = (H, cos)
        print(f"    H={H:.0e}:  cosine={cos:.6f}   rel={rel:.2e}")
    H_best = best[0]
    print(f"  --> best H = {H_best:.0e}  (cosine={best[1]:.6f})")

    # (B) per-component breakdown at best H ----------------------------
    res = cache[H_best]
    adj, fd = res["adj"], res["fd"]
    print(f"\n(B) per-component adjoint vs FD at H={H_best:.0e} (mean +/- std over seeds):")
    print(f"  {'coef':4s} {'adjoint':>22s} {'finite-diff':>22s} {'rel.diff':>10s}")
    for name, ma, sa, mf, sf in zip(coeff_names(C),
                                    adj.mean(0), adj.std(0), fd.mean(0), fd.std(0)):
        rel = abs(ma - mf) / (abs(mf) + 1e-12)
        print(f"  {name:4s} {ma:11.5f}+-{sa:8.5f} {mf:11.5f}+-{sf:8.5f} {rel:10.2e}")

    cos = cosine(adj.mean(0), fd.mean(0))
    print(f"\n  cosine(mean adjoint, mean FD) at best H = {cos:.6f}")
    print("  RESULT:", "PASS" if cos > 0.999 else "FAIL")


if __name__ == "__main__":
    main()
