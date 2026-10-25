"""
Forward pass of the open-channel DSMC drag problem with an FFD wall.

Specular reflection only. No adjoint and no shape step. The body is the
polyline of samples embedded in a tensor-product B-spline lattice; at the
Greville control net Q0 that polyline is the sampled baseline (a circle, or a
Fourier star if you pass coefficients).

Run from any working directory. Defaults below are the quick fibonacci test;
CLI flags override them.

    python experiments/ffd/run_ffd_forward.py --preset identity --n-ref 4000 --t-obs 1
    python experiments/ffd/run_ffd_forward.py --compare --n-ref 20000 --t-obs 2 --seed 0
    python experiments/ffd/run_ffd_forward.py --preset dent --n-ref 20000 --t-obs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "simple"))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import runInflow as ri
from ffd_forward import simulate_ffd
from ffd_geometry import (
    DENT_PERTURBATION,
    BoundaryQuery,
    EmbeddedBody,
    FFDLattice,
    apply_perturbation,
    sample_circle,
    sample_polar_wall,
    shape_report,
)

# =============================================================================
# Config — defaults for a terminal run. Every field is overridable from the CLI.
# =============================================================================
N_REF = 20_000          # simulator particles representing the reference gas mass
T_OBS = 2.0             # counted window length (T_START stays 0). DT is runInflow's 0.1
T_WARM = 0.0            # uncounted steps before the window. 0 = cold start (the comparison)
SEED = 0
COLLISIONS = True
PRESET = "identity"     # "identity" (Q = Q0) or "dent" (non-star perturbation)
N_BOUNDARY = 256        # polyline samples of the initial wall
NU, NV, DEGREE = 6, 6, 3
LATTICE_HALF = 1.6      # initial rectangle [-half, half]^2 (must enclose the wall)
WALL_RADIUS = 1.0       # used when no Fourier coefficient vector is given
# Fourier wall, mode order [c0, a1..aN, b1..bN]. None -> circle of WALL_RADIUS.
FOURIER_C = None
# Extra (i, j, dx, dy) shifts applied after the preset. Empty keeps the preset.
PERTURB = []
N_BINS = 16             # uniform grid over the body bbox for the segment broad-phase
OUT_DIR = os.path.join(HERE, "output")


def _parse_perturb(text):
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("perturbation must be i,j,dx,dy")
    return parts


def _parse_coeffs(text):
    return [float(x) for x in text.split(",")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FFD polyline forward pass (specular drag).")
    p.add_argument("--preset", choices=("identity", "dent"), default=PRESET)
    p.add_argument("--n-ref", type=int, default=N_REF)
    p.add_argument("--t-obs", type=float, default=T_OBS)
    p.add_argument("--t-warm", type=float, default=T_WARM)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--collisions", dest="collisions", action="store_true", default=COLLISIONS)
    p.add_argument("--no-collisions", dest="collisions", action="store_false")
    p.add_argument("--n-boundary", type=int, default=N_BOUNDARY)
    p.add_argument("--nu", type=int, default=NU)
    p.add_argument("--nv", type=int, default=NV)
    p.add_argument("--degree", type=int, default=DEGREE)
    p.add_argument("--lattice-half", type=float, default=LATTICE_HALF)
    p.add_argument("--radius", type=float, default=WALL_RADIUS)
    p.add_argument("--fourier", type=_parse_coeffs, default=None,
                   help="comma-separated Fourier coefficients [c0, a1..aN, b1..bN] for the initial wall")
    p.add_argument("--perturb", action="append", type=_parse_perturb, default=None,
                   help="i,j,dx,dy added to a control point. Repeatable. Applied after the preset.")
    p.add_argument("--control-points", default=None,
                   help="npy file of absolute Q with shape (nu, nv, 2); replaces Q0 and the preset")
    p.add_argument("--n-bins", type=int, default=N_BINS)
    p.add_argument("--compare", action="store_true",
                   help="also run the stock star-shaped forward pass on the same seeds and window")
    p.add_argument("--noise-seed", type=int, default=None,
                   help="second seed used to estimate run-to-run scatter (default with --compare: seed+1)")
    p.add_argument("--no-noise-seed", action="store_true")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-particles", action="store_true")
    return p.parse_args(argv)


def configure_solver(n_ref, t_obs, collisions):
    """Point the shared runInflow globals at this run. Restored by the caller."""
    ri.WARM_START = False
    ri.COLLISIONS = bool(collisions)
    ri.PREFILL = True
    ri.T_START = 0.0
    ri.TOBS = float(t_obs)
    ri.N_STEPS = int(round(float(t_obs) / ri.DT))
    if ri.N_STEPS < 1:
        raise SystemExit("t-obs is shorter than one time step")
    ri.set_particle_count(int(n_ref), cap=max(int(round(1.65 * n_ref)), n_ref + 1000))


def solver_snapshot():
    keys = ("N_REF", "CAP", "M_P", "N0", "N_STEPS", "TOBS", "T_START",
            "WARM_START", "COLLISIONS", "PREFILL")
    return {k: getattr(ri, k) for k in keys}


def restore_solver(snap):
    for key, value in snap.items():
        setattr(ri, key, value)
    ri.set_particle_count(snap["N_REF"], cap=snap["CAP"])


def build_wall(args):
    coeffs = args.fourier if args.fourier is not None else FOURIER_C
    if coeffs is not None:
        return sample_polar_wall(np.asarray(coeffs, float), args.n_boundary), np.asarray(coeffs, float)
    return sample_circle(args.radius, args.n_boundary), None


def build_controls(args, lattice: FFDLattice):
    if args.control_points:
        Q = np.load(args.control_points)
        if Q.shape != lattice.Q0.shape:
            raise SystemExit(f"--control-points has shape {Q.shape}, lattice is {lattice.Q0.shape}")
        Q = np.array(Q, dtype=float, copy=True)
    else:
        Q = lattice.Q0.copy()
        if args.preset == "dent":
            if (lattice.nu, lattice.nv, lattice.degree) != (6, 6, 3) or abs(args.lattice_half - 1.6) > 1e-9:
                raise SystemExit(
                    "preset 'dent' is the 6x6 cubic perturbation on [-1.6, 1.6]^2. "
                    "Pass --perturb i,j,dx,dy (or --control-points) for a different lattice."
                )
            Q = apply_perturbation(Q, DENT_PERTURBATION)
    extra = args.perturb if args.perturb is not None else PERTURB
    if extra:
        Q = apply_perturbation(Q, extra)
    return Q


def make_body(args):
    half = float(args.lattice_half)
    lattice = FFDLattice(origin=(-half, -half), extent=(2 * half, 2 * half),
                         nu=args.nu, nv=args.nv, degree=args.degree)
    wall, coeffs = build_wall(args)
    body = EmbeddedBody(lattice, wall)
    Q = build_controls(args, lattice)
    pts = body.boundary(Q)
    report = shape_report(pts)
    query = BoundaryQuery(pts, n_bins=args.n_bins)
    return lattice, body, Q, query, report, coeffs


def _window_mean(arr):
    """Mean over the counted steps. The trailing entry is the extra unweighted step."""
    arr = np.asarray(arr)
    sl = arr[:-1] if len(arr) > 1 else arr
    return float(np.mean(sl)) if len(sl) else float("nan")


def run_ffd(query, seed, t_warm):
    t0 = time.perf_counter()
    out = simulate_ffd(query, seed, t_warm=t_warm)
    out["seconds"] = time.perf_counter() - t0
    return out


def run_fourier(seed, coeffs):
    """Stock star-shaped forward pass. ``coeffs is None`` uses the unit-circle C_init."""
    C = ri.C_init if coeffs is None else np.asarray(coeffs, float)
    t0 = time.perf_counter()
    # record=True is what fills LAST_DIAG's hit counts; the impulse is the same
    # estimator simulate_impulse returns.
    ix, _steps, _edge = ri.simulate_and_record(C, seed)
    n_alive, n_hits = ri.LAST_DIAG
    return {
        "Ix": float(ix),
        "drag": float(ri.drag_force(ix)),
        "n_alive": np.asarray(n_alive),
        "n_hits": np.asarray(n_hits),
        "seconds": time.perf_counter() - t0,
        "seed": int(seed),
    }


def plot_run(path, Q, boundary, particles, title):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    closed = np.vstack([boundary, boundary[:1]])
    for ax, zoom in zip(axes, (True, False)):
        if not zoom:
            box_x = [-ri.LX / 2, ri.LX / 2, ri.LX / 2, -ri.LX / 2, -ri.LX / 2]
            box_y = [-ri.LY / 2, -ri.LY / 2, ri.LY / 2, ri.LY / 2, -ri.LY / 2]
            ax.plot(box_x, box_y, color="0.75", lw=1.0)
        nu, nv, _ = Q.shape
        for i in range(nu):
            ax.plot(Q[i, :, 0], Q[i, :, 1], color="tab:orange", lw=0.8, alpha=0.9)
        for j in range(nv):
            ax.plot(Q[:, j, 0], Q[:, j, 1], color="tab:orange", lw=0.8, alpha=0.9)
        ax.scatter(Q[..., 0].ravel(), Q[..., 1].ravel(), s=14, c="tab:orange", zorder=3, label="control points")
        ax.plot(closed[:, 0], closed[:, 1], color="k", lw=1.6, label="boundary", zorder=4)
        if particles is not None and len(particles):
            step = max(1, len(particles) // 4000)
            ax.scatter(particles[::step, 0], particles[::step, 1], s=2, c="tab:blue",
                       alpha=0.35, linewidths=0, label="particles", zorder=2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        if zoom:
            pad = 0.35
            lo = np.minimum(Q.min(axis=(0, 1)), boundary.min(axis=0)) - pad
            hi = np.maximum(Q.max(axis=(0, 1)), boundary.max(axis=0)) + pad
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_title("body and control lattice")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_xlim(-ri.LX / 2, ri.LX / 2)
            ax.set_ylim(-ri.LY / 2, ri.LY / 2)
            ax.set_title("channel")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _jsonable_report(report):
    out = {}
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _result_row(name, seed, result):
    row = {
        "method": name,
        "seed": int(seed),
        "drag": float(result["drag"]),
        "Ix": float(result["Ix"]),
        "mean_alive": _window_mean(result["n_alive"]),
        "mean_hits_per_step": _window_mean(result["n_hits"]),
        "seconds": float(result["seconds"]),
    }
    for key in ("n_fallback", "n_frozen", "n_resample", "n_bounce_events"):
        if key in result:
            row[key + "_sum"] = int(np.sum(result[key]))
    return row


def main(argv=None):
    args = parse_args(argv)
    if args.compare and args.preset != "identity":
        raise SystemExit("--compare runs the identity lattice against the Fourier solver; use --preset dent separately.")
    if args.compare and (args.perturb or args.control_points):
        raise SystemExit("--compare uses Q0 so the embedded wall is the baseline curve. Drop --perturb / --control-points.")
    if args.compare and args.t_warm != 0.0:
        raise SystemExit("--compare is a cold start on both solvers (same prefill, no warm cache). Leave --t-warm at 0.")
    noise_seed = None
    if args.compare and not args.no_noise_seed:
        noise_seed = args.seed + 1 if args.noise_seed is None else args.noise_seed

    os.makedirs(args.out_dir, exist_ok=True)
    lattice, body, Q, query, report, coeffs = make_body(args)
    print(
        f"FFD lattice {args.nu}x{args.nv} degree {args.degree} on "
        f"[{-args.lattice_half:g}, {args.lattice_half:g}]^2, "
        f"{args.n_boundary} boundary samples, preset {args.preset}"
    )
    print(
        f"  area={report['area']:.6f}  perimeter={report['perimeter']:.6f}  "
        f"self_intersections={report['self_intersections']}  "
        f"radial_graph={report['radial_graph']}  "
        f"ray_crossings={report['radial_crossings_min']}..{report['radial_crossings_max']}"
    )
    if report["self_intersections"]:
        print("  warning: the polyline crosses itself; inside tests and drag are not meaningful")
    if not report["radial_graph"]:
        print("  the wall is not a radial graph about the origin (the Fourier solver cannot represent it)")

    snap = solver_snapshot()
    rows = []
    ffd_out = None
    try:
        configure_solver(args.n_ref, args.t_obs, args.collisions)
        window = ri.N_STEPS * ri.DT
        print(
            f"  N_REF={ri.N_REF}  steps={ri.N_STEPS}  dt={ri.DT:g}  window={window:g}  "
            f"collisions={ri.COLLISIONS}  t_warm={args.t_warm:g}  seed={args.seed}"
        )
        ffd_out = run_ffd(query, args.seed, args.t_warm)
        rows.append(_result_row("ffd", args.seed, ffd_out))
        print(
            f"FFD     seed {args.seed}:  drag F_x = {ffd_out['drag']:.6g}   "
            f"hits/step = {_window_mean(ffd_out['n_hits']):.2f}   "
            f"bounces/step = {_window_mean(ffd_out['n_bounce_events']):.2f}   "
            f"alive = {_window_mean(ffd_out['n_alive']):.1f}   "
            f"fallback = {int(ffd_out['n_fallback'].sum())}   "
            f"frozen = {int(ffd_out['n_frozen'].sum())}   "
            f"resampled = {int(ffd_out['n_resample'].sum())}   "
            f"[{ffd_out['seconds']:.2f}s]"
        )
        if args.compare:
            fourier = run_fourier(args.seed, coeffs)
            rows.append(_result_row("fourier", args.seed, fourier))
            print(
                f"Fourier seed {args.seed}:  drag F_x = {fourier['drag']:.6g}   "
                f"hits/step = {_window_mean(fourier['n_hits']):.2f}   "
                f"alive = {_window_mean(fourier['n_alive']):.1f}   "
                f"[{fourier['seconds']:.2f}s]"
            )
            rel = abs(ffd_out["drag"] - fourier["drag"]) / (abs(fourier["drag"]) + 1e-300)
            print(f"  |F_ffd - F_fourier| / |F_fourier| = {rel:.4g}   (same seed)")
            if noise_seed is not None:
                fourier_b = run_fourier(noise_seed, coeffs)
                rows.append(_result_row("fourier", noise_seed, fourier_b))
                ffd_b = run_ffd(query, noise_seed, args.t_warm)
                rows.append(_result_row("ffd", noise_seed, ffd_b))
                noise = abs(fourier["drag"] - fourier_b["drag"]) / (abs(fourier["drag"]) + 1e-300)
                noise_ffd = abs(ffd_out["drag"] - ffd_b["drag"]) / (abs(ffd_out["drag"]) + 1e-300)
                print(
                    f"Fourier seed {noise_seed}:  drag F_x = {fourier_b['drag']:.6g}   "
                    f"hits/step = {_window_mean(fourier_b['n_hits']):.2f}   [{fourier_b['seconds']:.2f}s]"
                )
                print(
                    f"FFD     seed {noise_seed}:  drag F_x = {ffd_b['drag']:.6g}   "
                    f"hits/step = {_window_mean(ffd_b['n_hits']):.2f}   [{ffd_b['seconds']:.2f}s]"
                )
                print(f"  Fourier seed scatter |F({args.seed})-F({noise_seed})| / |F({args.seed})| = {noise:.4g}")
                print(f"  FFD     seed scatter |F({args.seed})-F({noise_seed})| / |F({args.seed})| = {noise_ffd:.4g}")
    finally:
        restore_solver(snap)

    tag = f"{args.preset}_n{args.n_ref}_t{args.t_obs:g}_s{args.seed}"
    summary = {
        "preset": args.preset,
        "n_ref": args.n_ref,
        "t_obs": args.t_obs,
        "t_warm": args.t_warm,
        "seed": args.seed,
        "collisions": args.collisions,
        "n_boundary": args.n_boundary,
        "nu": args.nu,
        "nv": args.nv,
        "degree": args.degree,
        "lattice_half": args.lattice_half,
        "shape": _jsonable_report(report),
        "runs": rows,
    }
    summary_path = os.path.join(args.out_dir, f"summary_{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {summary_path}")

    if not args.no_plot and ffd_out is not None:
        particles = None if args.no_particles else ffd_out["x"]
        drag = ffd_out["drag"]
        title = f"{args.preset}  F_x = {drag:.4g}   N_REF={args.n_ref}  t={args.t_obs:g}  seed={args.seed}"
        plot_path = os.path.join(args.out_dir, f"ffd_{tag}.png")
        plot_run(plot_path, Q, query.pts, particles, title)
        np.savez(
            os.path.join(args.out_dir, f"ffd_{tag}.npz"),
            Q=Q, boundary=query.pts, drag=np.array(drag),
        )
        print(f"Wrote {plot_path}")
    return summary


if __name__ == "__main__":
    main()
