"""
video_density.py -- run the OPEN-CHANNEL forward simulation (runInflow.py: inflow left, outflow
right, top/bottom per runInflow.SIDE_BC) from the uniform pre-fill until the flow is statistically steady,
and render a video of the gas density n/n0 in every collision cell as a function of time.

    python3 video_density.py                       # defaults: seed 0, C_init, collisions on, grid 32x16
    python3 video_density.py --tmax 100 --tol 0.005 --every 0.5 --grid 64 32 --out my_density

Outputs (in this directory): <out>.gif (or .mp4 when ffmpeg is installed), <out>_final.png (the
steady density field), <out>_timeseries.png (alive count and drag force per block vs time).

Steady-state criterion: the density PROFILE along the channel (column means of the field,
averaged over consecutive blocks of `--block` time units; per-cell values are too noisy, ~6%
block to block at ~50 particles per cell) must change by less than `--tol` (relative L2)
between consecutive blocks AND the particle count by less than `--alive-tol` (per block; the
count is the cleanest slow variable: the profile's own block-to-block noise is ~2%), after at
least 10 time units; otherwise the run ends at --tmax. Frames show a running mean over
`--smooth` time units. Expect ~50-70 time units from the pre-fill: the body blocks a
quarter of the channel, so the upstream gas compresses to ~1.8 n0 before the inflow, the flux
through the throat and the ~20% of particles returning through the inlet balance.
"""
import argparse, os, shutil, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors
warnings.filterwarnings("ignore")
import runInflow as ri

# ---- palette (dataviz reference: one sequential hue light->dark, neutral gray for masked cells) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP.set_bad("#d9d8d3")          # cells inside the body


def gas_fraction(nx, ny, C, n_sub=8):
    """Fraction of each display cell that is gas (outside the body); index [ix, iy]."""
    u = (np.arange(n_sub) + 0.5) / n_sub
    cx = -ri.LX / 2 + (np.arange(nx)[:, None] + u[None, :]) * (ri.LX / nx)
    cy = -ri.LY / 2 + (np.arange(ny)[:, None] + u[None, :]) * (ri.LY / ny)
    X, Y = np.broadcast_arrays(cx[:, None, :, None], cy[None, :, None, :])
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = ri.inside_body(pts, C).reshape(nx, ny, n_sub * n_sub)
    return 1.0 - inside.mean(axis=2)


def density_field(x, alive, nx, ny, gfrac):
    """n/n0 per display cell (gas area of the cell), NaN where the cell is (almost) all body."""
    xa = x[alive]
    ix = np.clip(((xa[:, 0] + ri.LX / 2) / ri.LX * nx).astype(int), 0, nx - 1)
    iy = np.clip(((xa[:, 1] + ri.LY / 2) / ri.LY * ny).astype(int), 0, ny - 1)
    counts = np.bincount(ix * ny + iy, minlength=nx * ny).reshape(nx, ny).astype(float)
    area = (ri.LX / nx) * (ri.LY / ny) * gfrac
    with np.errstate(divide="ignore", invalid="ignore"):
        n = counts / (area * ri.N0)
    n[gfrac < 0.05] = np.nan
    return n


def run(args):
    C = ri.C_init.copy() if args.C is None else np.asarray(args.C, float)
    ri.COLLISIONS = not args.no_collisions
    nx, ny = args.grid
    gfrac = gas_fraction(nx, ny, C)
    seed = args.seed
    every = max(1, int(round(args.every / ri.DT)))
    block = max(1, int(round(args.block / ri.DT)))
    kmax = int(round(args.tmax / ri.DT))

    x = np.zeros((ri.CAP, 2)); v = np.zeros((ri.CAP, ri.VEL_DIM)); alive = np.zeros(ri.CAP, bool); pid = np.zeros(ri.CAP, np.int64)
    counter = ri.prefill(x, v, alive, pid, seed, C)
    frames, times, alive_ts, imp_ts = [], [], [], []
    block_means, block_alive = [], []
    acc = np.zeros((nx, ny)); acc_n = 0; imp_block = 0.0; steady_at = None
    n_smooth = max(1, int(round(args.smooth / ri.DT))); recent = []
    t0 = time.time()
    print(f"open channel {ri.LX}x{ri.LY}, seed {seed}, collisions={ri.COLLISIONS}, grid {nx}x{ny}, frame every {every*ri.DT:g}, "
          f"block {block*ri.DT:g}, tol {args.tol}, tmax {args.tmax}")
    for k in range(kmax):
        if ri.COLLISIONS:
            ri.collide_open(x, v, alive, pid, seed, k, C)
        Ix_step, _ = ri.flight_step(x, v, alive, C, ri._reflection_rng(seed, k), None, 1.0)
        _, counter = ri.inject_all(x, v, alive, pid, seed, k, counter)
        imp_block += Ix_step
        t = (k + 1) * ri.DT
        n = density_field(x, alive, nx, ny, gfrac)
        acc += np.nan_to_num(n); acc_n += 1
        recent.append(n); recent = recent[-n_smooth:]
        if (k + 1) % every == 0:
            frames.append(np.mean(recent, axis=0)); times.append(t); alive_ts.append(int(alive.sum()))
        if (k + 1) % block == 0:
            mean_field = acc / acc_n; acc[:] = 0; acc_n = 0
            block_means.append(mean_field); block_alive.append(int(alive.sum()))
            imp_ts.append((t, imp_block / (block * ri.DT))); imp_block = 0.0
            msg = f"  t={t:6.1f}  alive={block_alive[-1]:6d}  F_x(block)={imp_ts[-1][1]:7.1f}"
            if len(block_means) >= 2:
                prof_new = np.nanmean(np.where(gfrac > 0.05, block_means[-1], np.nan), axis=1)
                prof_old = np.nanmean(np.where(gfrac > 0.05, block_means[-2], np.nan), axis=1)
                d = np.linalg.norm(prof_new - prof_old) / np.linalg.norm(prof_new)
                da = abs(block_alive[-1] - block_alive[-2]) / block_alive[-1]
                msg += f"  profile change={100*d:5.2f}%  alive change={100*da:5.2f}%"
                if t >= 10.0 and d < args.tol and da < args.alive_tol and steady_at is None:
                    steady_at = t
            print(msg + f"   [{time.time()-t0:.0f}s]")
            if steady_at is not None:
                break
    if steady_at is None:
        print(f"not steady by tmax={args.tmax} (profile or count still changing above tolerance); rendering what was simulated")
    else:
        print(f"steady state reached at t={steady_at:g} (density-profile change < {100*args.tol:g}% per {block*ri.DT:g} time units)")
    up = np.nanmean(frames[-1][int(nx * 0.2):int(nx * 0.4), :]); down = np.nanmean(frames[-1][int(nx * 0.6):int(nx * 0.8), :])
    print(f"final field: upstream mean n/n0={up:.2f}, downstream mean n/n0={down:.2f}, alive={alive_ts[-1]}")
    render(frames, times, alive_ts, imp_ts, C, args, steady_at)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2); ax.xaxis.label.set_color(INK2)


def render(frames, times, alive_ts, imp_ts, C, args, steady_at):
    nx, ny = args.grid
    vmax = args.vmax if args.vmax else float(np.ceil(np.nanpercentile(np.stack(frames), 99.5) * 4) / 4)
    fig = plt.figure(figsize=(12.5, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.0], height_ratios=[1, 1], wspace=0.28, hspace=0.45,
                          left=0.05, right=0.97, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[:, 0]); ax_a = fig.add_subplot(gs[0, 1]); ax_f = fig.add_subplot(gs[1, 1])
    for a in (ax, ax_a, ax_f):
        _style(a)
    extent = [-ri.LX / 2, ri.LX / 2, -ri.LY / 2, ri.LY / 2]
    im = ax.imshow(np.ma.masked_invalid(frames[0]).T, origin="lower", extent=extent, cmap=CMAP, vmin=0.0, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    bx, by = ri.body_outline(C); ax.plot(bx, by, color=INK, lw=1.2)
    ax.set_xlabel(f"x  (inflow at left, outflow at right; top/bottom: {ri.SIDE_BC})"); ax.set_ylabel("y")
    ax.tick_params(colors=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("gas density  n / n0", color=INK2); cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    title = ax.set_title("", color=INK, loc="left", fontsize=12)
    # side panels: alive count and drag force per block (one series each -> no legend)
    ta = np.array(times); al = np.array(alive_ts)
    ax_a.plot(ta, al, color="#2a78d6", lw=2); ax_a.set_title("particles in the domain", color=INK, fontsize=10, loc="left")
    ax_a.set_xlim(0, ta[-1]); ax_a.grid(True, color=GRID, lw=0.6); ax_a.set_xlabel("t")
    tb = np.array([p[0] for p in imp_ts]); fb = np.array([p[1] for p in imp_ts])
    ax_f.plot(tb, fb, color="#2a78d6", lw=2, marker="o", ms=4, markerfacecolor=SURFACE)
    ax_f.set_title(f"drag force F_x per {args.block:g}-unit block", color=INK, fontsize=10, loc="left")
    ax_f.set_xlim(0, ta[-1]); ax_f.grid(True, color=GRID, lw=0.6); ax_f.set_xlabel("t")
    if steady_at is not None:
        for a in (ax_a, ax_f):
            a.axvline(steady_at, color=INK2, lw=1, ls=":")
    cur_a = ax_a.axvline(ta[0], color=INK, lw=1); cur_f = ax_f.axvline(ta[0], color=INK, lw=1)
    fig.text(0.05, 0.94, "Open-channel DSMC: gas density per cell while the flow settles" +
             (f"  (steady from t = {steady_at:g})" if steady_at is not None else ""), color=INK, fontsize=13, weight="bold")

    def update(i):
        im.set_data(np.ma.masked_invalid(frames[i]).T)
        title.set_text(f"t = {times[i]:5.2f}    alive = {alive_ts[i]}")
        cur_a.set_xdata([times[i], times[i]]); cur_f.set_xdata([times[i], times[i]])
        return im, title, cur_a, cur_f

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    update(len(frames) - 1); fig.savefig(out + "_final.png", dpi=130, facecolor=SURFACE)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    if shutil.which("ffmpeg"):
        anim.save(out + ".mp4", writer=animation.FFMpegWriter(fps=args.fps, bitrate=2500), dpi=100, savefig_kwargs={"facecolor": SURFACE})
        print(f"saved {out}.mp4 ({len(frames)} frames)")
    else:
        anim.save(out + ".gif", writer=animation.PillowWriter(fps=args.fps), dpi=80, savefig_kwargs={"facecolor": SURFACE})
        print(f"saved {out}.gif ({len(frames)} frames; install ffmpeg for an mp4)")
    print(f"saved {out}_final.png")
    # standalone time series
    fig2, (b1, b2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, facecolor=SURFACE)
    for a in (b1, b2): _style(a); a.grid(True, color=GRID, lw=0.6)
    b1.plot(ta, al, color="#2a78d6", lw=2); b1.set_ylabel("particles in domain")
    b2.plot(tb, fb, color="#2a78d6", lw=2, marker="o", ms=4, markerfacecolor=SURFACE); b2.set_ylabel(f"F_x per {args.block:g} units"); b2.set_xlabel("t")
    if steady_at is not None:
        for a in (b1, b2): a.axvline(steady_at, color=INK2, lw=1, ls=":")
    fig2.suptitle("Approach to the steady state", color=INK, x=0.12, ha="left")
    fig2.tight_layout(); fig2.savefig(out + "_timeseries.png", dpi=130, facecolor=SURFACE); print(f"saved {out}_timeseries.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=80.0, help="maximum simulated time")
    ap.add_argument("--tol", type=float, default=0.025, help="max relative change of the block-mean density profile between blocks")
    ap.add_argument("--alive-tol", type=float, default=0.005, help="max relative change of the particle count between blocks")
    ap.add_argument("--block", type=float, default=5.0, help="averaging block for the steadiness test (time units)")
    ap.add_argument("--every", type=float, default=0.25, help="time between video frames")
    ap.add_argument("--smooth", type=float, default=1.0, help="running-mean window of the displayed field (time units)")
    ap.add_argument("--grid", type=int, nargs=2, default=list(ri.N_COLL_CELLS), metavar=("NX", "NY"), help="display grid (default: the collision grid)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--vmax", type=float, default=None, help="colour scale maximum in units of n0 (default: 99.5th percentile)")
    ap.add_argument("--no-collisions", action="store_true")
    ap.add_argument("--C", type=float, nargs=7, default=None, help="Fourier coefficients of the body (default C_init)")
    ap.add_argument("--out", default="density_evolution")
    run(ap.parse_args())
