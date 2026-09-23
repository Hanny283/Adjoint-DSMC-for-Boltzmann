"""Load results, aggregate them, and draw the figures. Every figure saves <name>.png, <name>.pdf and one CSV
per plotted table (<name>__<table>.csv), so the numbers behind each panel travel with the picture.

Only numpy, pandas and matplotlib are needed: the analysis never imports the simulation code, so it runs on
any machine that has the results folder.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---- palette: the reference data-viz instance; SEQ_BLUE is the house ramp of video_density.py -------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SEQ_BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7", 400: "#3987e5",
            450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281", 700: "#0d366b"}
ORDINAL_STEPS = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700]        # ordinal ramp: no lighter than 250 on light
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]                                    # categorical slots 1-3 (all-pairs safe)
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b", none="#c3c2b7")
KNOBS = {"nref": "N_REF", "seeds": "N_AVG", "nfourier": "N_FOURIER", "frac": "GRAD_INIT_FRAC"}
KNOB_SHORT = {"N_REF": "N_REF", "N_AVG": "seeds", "N_FOURIER": "N", "GRAD_INIT_FRAC": "frac"}
KNOB_LABEL = {"N_REF": "particles N_REF", "N_AVG": "CRN seeds per gradient", "N_FOURIER": "Fourier modes N",
              "GRAD_INIT_FRAC": "GRAD_INIT_FRAC (first-step size)"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
                     "legend.frameon": False})


def ordinal_colors(n):
    if n <= 1:
        return [SEQ_BLUE[450]]
    idx = np.linspace(0, len(ORDINAL_STEPS) - 1, n).round().astype(int)
    return [SEQ_BLUE[ORDINAL_STEPS[i]] for i in idx]


def _fmt(v):
    if isinstance(v, (int, np.integer)) and v >= 1000:
        return f"{v // 1000}k" if v < 1_000_000 else f"{v / 1e6:g}M"
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


def save_fig(fig, out_base, tables=None, dpi=160):
    """Write <out_base>.png/.pdf and <out_base>__<table>.csv for every DataFrame/array in `tables`."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    for name, tab in (tables or {}).items():
        if isinstance(tab, pd.DataFrame):
            tab.to_csv(f"{out_base}__{name}.csv", index=False)
        else:
            np.savez(f"{out_base}__{name}.npz", **tab)
    plt.close(fig)
    return out_base.with_suffix(".png")


# ---- loading -------------------------------------------------------------------------------------------------
def load_history(run_dir):
    with np.load(Path(run_dir) / "history.npz", allow_pickle=False) as z:
        h = {k: z[k] for k in z.files}
    for k, v in h.items():
        if v.ndim == 0:
            h[k] = v.item() if v.dtype.kind in "fiub" else str(v)
    return h


def load_summary(run_dir):
    return json.loads((Path(run_dir) / "summary.json").read_text())


def _run_dirs(root, kind):
    for cfgp in sorted(Path(root).rglob("config.json")):
        rd = cfgp.parent
        if (rd / "DONE").exists() and (rd / "summary.json").exists():
            s = load_summary(rd)
            if s.get("kind") == kind:
                yield rd, s


def load_runs(root):
    """One row per finished optimisation run: params, classification, final shape, timing, path."""
    rows = []
    for rd, s in _run_dirs(root, "optimise"):
        p = s["params"]
        cl = s.get("classification", {})
        row = dict(run_dir=str(rd), group=s["group"], name=s["name"], run_id=s["run_id"], status=s.get("status"),
                   wall_min=s["wall_total_s"] / 60, n_workers=s.get("n_workers"))
        row.update({k: p[k] for k in ("N_REF", "N_FOURIER", "N_AVG", "SEED0", "GRAD_INIT_FRAC", "N_ITER", "TOBS", "CURV_LAMBDA")})
        row.update({k: cl.get(k) for k in ("verdict", "decayed", "plateau", "at_noise_floor", "decay_rel", "decay_z", "gnorm_tail_cv",
                                           "gnorm_tail_mean", "signif_tail_median", "iter_reach_floor", "n_iter_done", "loss_first", "loss_last",
                                           "force_first", "force_last", "first_step_disp", "backtracks_it0", "wall_per_iter_s")})
        row["C_opt_eff"] = np.asarray(s["C_opt_eff"], float)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["group", "name"]).reset_index(drop=True) if len(df) else df


def load_gradstats(root):
    """List of dicts (one per gradstat run) with the arrays, plus a summary DataFrame."""
    out = []
    for rd, s in _run_dirs(root, "gradstat"):
        with np.load(rd / "gradstat.npz", allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["N_REF"] = int(d["N_REF"]); d["N_FOURIER"] = int(d["N_FOURIER"])
        d["shape"] = s.get("shape", "C_init"); d["run_dir"] = str(rd); d["t_grad"] = float(d["t_grad"]); d["t_warm"] = float(d["t_warm"])
        out.append(d)
    out.sort(key=lambda d: (d["shape"], d["N_REF"]))
    tab = pd.DataFrame([dict(shape=d["shape"], N_REF=d["N_REF"], N_FOURIER=d["N_FOURIER"], K=len(d["seeds"]),
                             gnorm_mean=float(np.linalg.norm(d["G"].mean(0))),
                             sd_norm=float(np.sqrt(d["G"].var(0, ddof=1).sum())) if len(d["seeds"]) > 1 else np.nan,
                             loss_mean=float(d["losses"].mean()), loss_sd=float(d["losses"].std(ddof=1)) if len(d["seeds"]) > 1 else np.nan,
                             t_grad_s=d["t_grad"], t_warm_s=d["t_warm"], run_dir=d["run_dir"]) for d in out])
    return out, tab


def mode_labels(P):
    N = (P - 1) // 2
    return ["c0"] + [f"a{k}" for k in range(1, N + 1)] + [f"b{k}" for k in range(1, N + 1)]


def _style_modes(ax, P):
    ax.set_xticks(range(P))
    ax.set_xticklabels(mode_labels(P), rotation=90 if P > 11 else 0)
    ax.axhline(0, color=INK2, lw=0.6)


# ---- E1: gradient statistics -----------------------------------------------------------------------------
def fig_grad_mean_vs_nref(gradstats, out_dir):
    """Per-mode mean +- s.e. of the per-seed gradient, one line per N_REF, one panel per shape."""
    shapes = sorted({d["shape"] for d in gradstats}, key=lambda s: 0 if s == "C_init" else 1)
    fig, axes = plt.subplots(1, len(shapes), figsize=(6.2 * len(shapes), 3.6), squeeze=False)
    rows = []
    for ax, shape in zip(axes[0], shapes):
        ds = [d for d in gradstats if d["shape"] == shape]
        cols = ordinal_colors(len(ds))
        for d, c in zip(ds, cols):
            G = d["G"]; K = len(d["seeds"]); P = G.shape[1]
            m = G.mean(0); se = G.std(0, ddof=1) / np.sqrt(K) if K > 1 else np.zeros(P)
            x = np.arange(P)
            ax.errorbar(x, m, yerr=se, color=c, marker="o", ms=4, lw=1.2, capsize=2, label=f"N_REF = {_fmt(d['N_REF'])}")
            for k in range(P):
                rows.append(dict(shape=shape, N_REF=d["N_REF"], mode=mode_labels(P)[k], mean=m[k], se=se[k],
                                 sd=(G[:, k].std(ddof=1) if K > 1 else np.nan), K=K, g_pen=float(d["g_pen"][k])))
        _style_modes(ax, P)
        ax.set_title(f"gradient components at {shape}  (mean ± s.e. over seeds)")
        ax.set_ylabel("dJ/dC  (raw units)")
        ax.legend(loc="best")
    return save_fig(fig, Path(out_dir) / "grad_mean_vs_nref", dict(values=pd.DataFrame(rows)))


def fig_grad_sd_vs_nref(gradstats, out_dir):
    """Per-seed spread of the gradient vs particle count: norm of the sd vector, with an N^-1/2 guide."""
    shapes = sorted({d["shape"] for d in gradstats}, key=lambda s: 0 if s == "C_init" else 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rows = []
    for shape, c in zip(shapes, CAT):
        ds = sorted([d for d in gradstats if d["shape"] == shape and len(d["seeds"]) > 1], key=lambda d: d["N_REF"])
        if not ds:
            continue
        N = np.array([d["N_REF"] for d in ds], float)
        sd = np.array([np.sqrt(d["G"].var(0, ddof=1).sum()) for d in ds])
        gm = np.array([np.linalg.norm(d["G"].mean(0)) for d in ds])
        ax.plot(N, sd, "o-", color=c, label=f"per-seed sd, {shape}")
        ax.plot(N, gm, "s--", color=c, lw=1.0, ms=4, alpha=0.8, label=f"‖mean g‖, {shape}")
        for d, s_, g_ in zip(ds, sd, gm):
            rows.append(dict(shape=shape, N_REF=d["N_REF"], sd_norm=s_, gnorm_mean=g_, K=len(d["seeds"]),
                             **{f"sd_{lab}": v for lab, v in zip(mode_labels(d["G"].shape[1]), d["G"].std(0, ddof=1))}))
        guide = sd[0] * np.sqrt(N[0] / N)
        ax.plot(N, guide, ":", color=INK2, lw=1.0, label="N_REF^-1/2 guide" if shape == shapes[0] else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("particles N_REF"); ax.set_ylabel("gradient (raw units)")
    ax.set_title("gradient noise per seed vs particle count")
    ax.legend(loc="best")
    return save_fig(fig, Path(out_dir) / "grad_sd_vs_nref", dict(values=pd.DataFrame(rows)))


def subset_stats(G, ns=(1, 2, 4, 8, 16), R=500, rng=None):
    """For n-seed subsets of the K per-seed gradients: rms deviation of the subset mean from the K-seed mean,
    mean norm of the subset mean, mean cosine with the K-seed mean (random subsets without replacement)."""
    rng = np.random.default_rng(0) if rng is None else rng
    K = len(G); gK = G.mean(0); rows = []
    for n in [n for n in ns if n <= K]:
        if n == K:
            subs = [np.arange(K)]
        else:
            subs = [rng.choice(K, n, replace=False) for _ in range(R)]
        M = np.array([G[s].mean(0) for s in subs])
        dev = np.sqrt(np.mean(np.sum((M - gK) ** 2, axis=1)))
        norm = np.mean(np.linalg.norm(M, axis=1))
        cos = np.mean(M @ gK / (np.linalg.norm(M, axis=1) * np.linalg.norm(gK) + 1e-300))
        rows.append(dict(n=n, rms_dev=dev, mean_norm=norm, mean_cos=cos, analytic_se=np.sqrt(G.var(0, ddof=1).sum() / n) if K > 1 else np.nan))
    return pd.DataFrame(rows)


def fig_grad_se_vs_nseeds(gradstats, out_dir, shape="C_init"):
    """Noise of the n-seed mean gradient vs n: rms deviation (with the sd/sqrt(n) line), ‖ĝ_n‖ (noise floor),
    and cosine with the 16-seed mean; one line per N_REF."""
    ds = sorted([d for d in gradstats if d["shape"] == shape and len(d["seeds"]) > 2], key=lambda d: d["N_REF"])
    if not ds:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    cols = ordinal_colors(len(ds)); rows = []
    for d, c in zip(ds, cols):
        t = subset_stats(d["G"]); t["N_REF"] = d["N_REF"]; t["shape"] = shape; rows.append(t)
        lab = f"N_REF = {_fmt(d['N_REF'])}"
        axes[0].plot(t.n, t.rms_dev, "o-", color=c, label=lab)
        axes[0].plot(t.n, t.analytic_se, ":", color=c, lw=1.0)
        axes[1].plot(t.n, t.mean_norm, "o-", color=c, label=lab)
        axes[1].axhline(np.linalg.norm(d["G"].mean(0)), color=c, lw=0.8, ls="--")
        axes[2].plot(t.n, t.mean_cos, "o-", color=c, label=lab)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("seeds averaged, n")
    axes[0].set_yscale("log"); axes[0].set_ylabel("rms ‖ĝ_n − ĝ_K‖  (dotted: sd/√n)"); axes[0].set_title(f"noise of the mean gradient, {shape}")
    axes[1].set_ylabel("mean ‖ĝ_n‖  (dashed: ‖ĝ_K‖)"); axes[1].set_title("norm inflated by noise: E‖ĝ‖² = ‖g‖² + trΣ/n")
    axes[2].set_ylabel("mean cos(ĝ_n, ĝ_K)"); axes[2].set_title("direction stability"); axes[2].set_ylim(0, 1.02)
    axes[0].legend(loc="best")
    return save_fig(fig, Path(out_dir) / f"grad_se_vs_nseeds_{shape}", dict(values=pd.concat(rows, ignore_index=True)))


# ---- sweeps ---------------------------------------------------------------------------------------------------
def select_sweep(df, knob, baseline):
    """Rows whose other knobs equal the baseline (so the knob is the only thing that changes)."""
    col = KNOBS.get(knob, knob)
    others = [c for c in KNOBS.values() if c != col]
    m = np.ones(len(df), bool)
    for c in others:
        m &= np.isclose(df[c].astype(float), float(baseline[c]))
    sub = df[m].copy()
    sub["knob_value"] = sub[col]
    return sub.sort_values(["knob_value", "SEED0"])


def fig_convergence(df, knob, baseline, root, out_dir):
    """Normalised loss, drag force and ‖g‖ (with its standard error dashed) vs iteration, coloured by knob value."""
    col = KNOBS.get(knob, knob)
    sub = select_sweep(df, knob, baseline)
    if sub.empty:
        return None
    vals = sorted(sub.knob_value.unique())
    cols = dict(zip(vals, ordinal_colors(len(vals))))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    rows = []; seen = set()
    for _, r in sub.iterrows():
        h = load_history(r.run_dir); it = np.arange(len(h["loss"])); c = cols[r.knob_value]
        rep = (sub.knob_value == r.knob_value).sum() > 1
        lab = f"{KNOB_SHORT[col]} = {_fmt(r.knob_value)}" if r.knob_value not in seen else None
        seen.add(r.knob_value)
        axes[0].plot(it, h["loss"], "-", color=c, lw=1.4 if not rep else 1.0, label=lab)
        axes[1].plot(it, h["force"], "-", color=c, lw=1.4 if not rep else 1.0, label=lab)
        axes[2].plot(it, h["gnorm"], "-", color=c, lw=1.4 if not rep else 1.0, label=lab)
        se = np.asarray(h["grad_se"], float)
        if np.isfinite(se).any():
            axes[2].plot(it, se, "--", color=c, lw=0.8)
        for i in it:
            rows.append(dict(knob=col, knob_value=r.knob_value, SEED0=r.SEED0, iter=int(i), loss=h["loss"][i], force=h["force"][i],
                             gnorm=h["gnorm"][i], grad_se=se[i], grad_signif=h["grad_signif"][i], n_seeds=h["n_seeds"][i]))
    axes[0].set_ylabel("loss  J / J(C_init)"); axes[0].set_title("objective (normalised)")
    axes[1].set_ylabel("drag force F_x"); axes[1].set_title("drag force (penalty removed)")
    axes[2].set_yscale("log"); axes[2].set_ylabel("‖g‖  (dashed: s.e. of g)"); axes[2].set_title("gradient norm")
    for ax in axes:
        ax.set_xlabel("accepted iteration")
    axes[0].legend(loc="best", title=KNOB_LABEL[col])
    fig.suptitle(f"convergence vs {KNOB_LABEL[col]}", y=1.02)
    return save_fig(fig, Path(out_dir) / f"convergence_vs_{knob}", dict(values=pd.DataFrame(rows)))


def fig_final_coeffs(df, knob, baseline, out_dir):
    """Final effective coefficients per mode, one line per knob value (mean ± sd over replicates when present)."""
    col = KNOBS.get(knob, knob)
    sub = select_sweep(df, knob, baseline)
    if sub.empty:
        return None
    vals = sorted(sub.knob_value.unique()); cols = dict(zip(vals, ordinal_colors(len(vals))))
    P = max(len(c) for c in sub.C_opt_eff)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * P + 3), 3.6)); rows = []
    for v in vals:
        Cs = np.array([np.pad(c, (0, P - len(c))) if len(c) < P else c for c in sub[sub.knob_value == v].C_opt_eff])
        # pad in "mode order": a shorter C = [c0, a1..aN, b1..bN] must be re-laid into the larger P
        if any(len(c) < P for c in sub[sub.knob_value == v].C_opt_eff):
            Cs = np.array([_relayout(c, P) for c in sub[sub.knob_value == v].C_opt_eff])
        m = Cs.mean(0); sd = Cs.std(0, ddof=1) if len(Cs) > 1 else np.zeros(P)
        ax.errorbar(np.arange(P), m, yerr=sd if len(Cs) > 1 else None, color=cols[v], marker="o", ms=4, lw=1.2, capsize=2,
                    label=f"{_fmt(v)}" + (f"  (n={len(Cs)})" if len(Cs) > 1 else ""))
        for k in range(P):
            rows.append(dict(knob=col, knob_value=v, mode=mode_labels(P)[k], mean=m[k], sd=sd[k], n_runs=len(Cs)))
    _style_modes(ax, P)
    ax.set_ylabel("final C_eff (simulated shape)"); ax.set_title(f"optimised coefficients vs {KNOB_LABEL[col]}")
    ax.legend(loc="best", title=KNOB_SHORT[col])
    return save_fig(fig, Path(out_dir) / f"final_coeffs_vs_{knob}", dict(values=pd.DataFrame(rows)))


def _relayout(c, P):
    """[c0, a1..an, b1..bn] -> length-P vector with the same modes (zeros for the missing higher modes)."""
    c = np.asarray(c, float); n = (len(c) - 1) // 2; N = (P - 1) // 2
    out = np.zeros(P); out[0] = c[0]; out[1:1 + n] = c[1:1 + n]; out[1 + N:1 + N + n] = c[1 + n:1 + 2 * n]
    return out


def outline(C, n=400):
    C = np.asarray(C, float); N = (len(C) - 1) // 2
    th = np.linspace(0, 1, n)
    r = np.full(n, C[0])
    if N > 0:
        k = np.arange(1, N + 1); ph = 2 * np.pi * np.outer(th, k)
        with np.errstate(all="ignore"):         # Accelerate on macOS emits spurious FP warnings on tiny matmuls
            r = r + np.cos(ph) @ C[1:N + 1] + np.sin(ph) @ C[N + 1:]
    return r * np.cos(2 * np.pi * th), r * np.sin(2 * np.pi * th)


def fig_shapes(df, knob, baseline, out_dir):
    col = KNOBS.get(knob, knob)
    sub = select_sweep(df, knob, baseline)
    if sub.empty:
        return None
    vals = sorted(sub.knob_value.unique()); cols = dict(zip(vals, ordinal_colors(len(vals))))
    fig, ax = plt.subplots(figsize=(5, 5)); rows = []
    x0, y0 = outline([1.0]); ax.plot(x0, y0, "--", color=INK2, lw=1.0, label="initial (unit circle)")
    seen = set()
    for _, r in sub.iterrows():
        x, y = outline(r.C_opt_eff)
        lab = f"{_fmt(r.knob_value)}" if r.knob_value not in seen else None; seen.add(r.knob_value)
        ax.plot(x, y, "-", color=cols[r.knob_value], lw=1.4, alpha=0.9 if lab else 0.5, label=lab)
        rows += [dict(knob_value=r.knob_value, SEED0=r.SEED0, x=xi, y=yi) for xi, yi in zip(x[::4], y[::4])]
    ax.set_aspect("equal"); ax.annotate("flow →", xy=(-1.35, 1.25), color=INK2)
    ax.set_title(f"final shapes vs {KNOB_LABEL[col]}"); ax.legend(loc="lower right", title=KNOB_SHORT[col])
    return save_fig(fig, Path(out_dir) / f"shapes_vs_{knob}", dict(outlines=pd.DataFrame(rows)))


def fig_coeff_trajectories(run_dir, out_dir, name=None):
    """Small multiples: every effective coefficient vs iteration for one run (single hue)."""
    h = load_history(run_dir); C = h["C_eff"]; n, P = C.shape; it = np.arange(n)
    ncol = 7 if P > 11 else max(3, P); nrow = int(np.ceil(P / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.6 * nrow + 0.6), sharex=True, squeeze=False)
    labs = mode_labels(P)
    for k, ax in enumerate(axes.ravel()):
        if k >= P:
            ax.axis("off"); continue
        ax.plot(it, C[:, k], "-", color=SEQ_BLUE[450], lw=1.3); ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(labs[k], fontsize=8, pad=2); ax.tick_params(labelsize=7)
    fig.suptitle(f"coefficient trajectories: {name or Path(run_dir).name}", y=1.0)
    tab = pd.DataFrame(C, columns=labs); tab.insert(0, "iter", it)
    return save_fig(fig, Path(out_dir) / f"coeff_trajectories_{name or Path(run_dir).name}", dict(values=tab))


def fig_frac_effects(df, baseline, out_dir):
    sub = select_sweep(df, "frac", baseline)
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4)); rows = []
    agg = []
    for _, r in sub.iterrows():
        h = load_history(r.run_dir); L = h["loss"]
        d = dict(frac=r.GRAD_INIT_FRAC, SEED0=r.SEED0, first_step_disp=r.first_step_disp, backtracks_it0=r.backtracks_it0,
                 loss_it1=L[1] if len(L) > 1 else np.nan, loss_it5=L[5] if len(L) > 5 else np.nan, loss_final=L[-1],
                 iter_reach_floor=r.iter_reach_floor, verdict=r.verdict, status=r.status)
        agg.append(d)
    t = pd.DataFrame(agg).sort_values("frac")
    axes[0].plot(t.frac, t.first_step_disp, "o-", color=SEQ_BLUE[450]); axes[0].set_ylabel("first accepted |ΔC|"); axes[0].set_title("first step actually taken")
    axes[1].plot(t.frac, t.backtracks_it0, "o-", color=SEQ_BLUE[450]); axes[1].set_ylabel("trials in iteration 0"); axes[1].set_title("backtracking at the start")
    for c, (k, lab) in zip(ordinal_colors(3), [("loss_it1", "after 1"), ("loss_it5", "after 5"), ("loss_final", "final")]):
        axes[2].plot(t.frac, t[k], "o-", color=c, label=lab)
    axes[2].set_ylabel("loss J/J(C_init)"); axes[2].set_title("progress"); axes[2].legend(loc="best", title="iterations")
    y = t.iter_reach_floor.astype(float)
    axes[3].plot(t.frac, y, "o-", color=SEQ_BLUE[450]); axes[3].set_ylabel("first iterate with ‖g‖/s.e. < 2"); axes[3].set_title("reaching the noise floor")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("GRAD_INIT_FRAC")
    return save_fig(fig, Path(out_dir) / "grad_init_frac_effects", dict(values=t))


def fig_wall_per_iter(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    t = df[["N_REF", "N_AVG", "n_workers", "wall_per_iter_s", "wall_min", "name", "group"]].copy()
    for ax, col in zip(axes, ("N_REF", "N_AVG")):
        g = t.groupby(col).wall_per_iter_s.agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(g[col], g["mean"] / 60, yerr=(g["std"] / 60).fillna(0), color=SEQ_BLUE[450], marker="o", capsize=2)
        ax.set_xscale("log", base=2 if col == "N_AVG" else 10); ax.set_xlabel(KNOB_LABEL[col]); ax.set_ylabel("wall time per iteration (min)")
    axes[0].set_title("cost vs particles (all runs)"); axes[1].set_title("cost vs seeds (parallel seeds)")
    return save_fig(fig, Path(out_dir) / "wall_per_iter", dict(values=t))


def success_table(df, out_dir):
    cols = ["group", "name", "N_REF", "N_AVG", "N_FOURIER", "GRAD_INIT_FRAC", "SEED0", "verdict", "decayed", "plateau", "at_noise_floor",
            "status", "n_iter_done", "decay_rel", "gnorm_tail_cv", "signif_tail_median", "loss_last", "force_first", "force_last", "wall_min"]
    t = df[[c for c in cols if c in df.columns]].copy()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "summary_table.csv", index=False)
    # heat-map style table of the three flags + verdict (colour + text, never colour alone)
    flags = ["decayed", "plateau", "at_noise_floor"]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.2)); ax.axis("off")
    cell_colors, cell_text = [], []
    for _, r in t.iterrows():
        rowc, rowt = [SURFACE, SURFACE], [f"{r.group}/{r['name']}", r.verdict]
        for f in flags:
            v = r[f]
            rowc.append(STATUS["good"] if v is True else STATUS["critical"] if v is False else STATUS["none"])
            rowt.append("yes" if v is True else "no" if v is False else "n/a")
        cell_colors.append(rowc); cell_text.append(rowt)
    tb = ax.table(cellText=cell_text, cellColours=cell_colors, colLabels=["run", "verdict"] + flags, loc="center", cellLoc="center",
                  colWidths=[0.36, 0.18, 0.15, 0.15, 0.16])
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.2)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if i > 0 and j >= 2:
            cell.get_text().set_color("#ffffff" if cell_colors[i - 1][j] != STATUS["none"] else INK)
    ax.set_title("success classification per run", pad=8)
    save_fig(fig, out / "success_table")
    return t


# ---- per-run quick look and the report --------------------------------------------------------------------
def plot_run(run_dir, out=None):
    run_dir = Path(run_dir); h = load_history(run_dir); it = np.arange(len(h["loss"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(it, h["loss"], "o-", color=SEQ_BLUE[450], ms=3); axes[0].set_ylabel("loss J/J(C_init)"); axes[0].set_title("objective")
    axes[1].plot(it, h["gnorm"], "o-", color=SEQ_BLUE[450], ms=3, label="‖g‖")
    se = np.asarray(h["grad_se"], float)
    if np.isfinite(se).any():
        axes[1].plot(it, se, "--", color=INK2, lw=1.0, label="s.e. of g")
        axes[1].legend(loc="best")
    axes[1].set_yscale("log"); axes[1].set_title("gradient norm")
    x0, y0 = outline(h["C_eff"][0]); x1, y1 = outline(h["C_eff"][-1])
    axes[2].plot(x0, y0, "--", color=INK2, lw=1.0, label="initial"); axes[2].plot(x1, y1, "-", color=SEQ_BLUE[450], lw=1.6, label="final")
    axes[2].set_aspect("equal"); axes[2].legend(loc="lower right"); axes[2].set_title("simulated shape")
    for ax in axes[:2]:
        ax.set_xlabel("accepted iteration")
    st = str(h.get("status", ""))
    fig.suptitle(f"{run_dir.name}   status: {st}   n_seeds: {int(np.max(h['n_seeds']))}", y=1.02)
    p = Path(out) if out else run_dir / "quicklook.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def make_figures(root, out_dir=None, baseline=None):
    """Every figure that the available results allow. Returns the list of PNG paths."""
    from .grids import BASELINE
    baseline = baseline or BASELINE
    root = Path(root); out_dir = Path(out_dir or root / "figures"); out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    gradstats, gtab = load_gradstats(root)
    if gradstats:
        gtab.to_csv(out_dir / "gradstat_table.csv", index=False)
        made += [fig_grad_mean_vs_nref(gradstats, out_dir), fig_grad_sd_vs_nref(gradstats, out_dir)]
        made += [fig_grad_se_vs_nseeds(gradstats, out_dir, shape=s) for s in sorted({d["shape"] for d in gradstats})]
    df = load_runs(root)
    if len(df):
        success_table(df, out_dir)
        made.append(out_dir / "success_table.png")
        for knob in KNOBS:
            made += [fig_convergence(df, knob, baseline, root, out_dir), fig_final_coeffs(df, knob, baseline, out_dir), fig_shapes(df, knob, baseline, out_dir)]
        made.append(fig_frac_effects(df, baseline, out_dir))
        made.append(fig_wall_per_iter(df, out_dir))
        for _, r in df[df.group == "baseline"].iterrows():
            made.append(fig_coeff_trajectories(r.run_dir, out_dir, name=r["name"]))
    return [m for m in made if m is not None]


def write_report(root, out=None, title="Adjoint-DSMC parameter study"):
    """Self-contained HTML (figures embedded) so results can be read anywhere with no code."""
    root = Path(root); out = Path(out or root / "report.html"); fig_dir = root / "figures"
    pngs = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    df = load_runs(root)
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title><style>body{{font-family:sans-serif;max-width:1200px;margin:20px auto;color:{INK};background:{SURFACE}}}"
             f"img{{max-width:100%;border:1px solid {GRID};margin:6px 0}} table{{border-collapse:collapse;font-size:12px}} td,th{{border:1px solid {GRID};padding:3px 6px}}</style></head><body>",
             f"<h1>{title}</h1><p>generated from <code>{root}</code>; every figure has a CSV twin with the plotted values in <code>figures/</code>.</p>"]
    if len(df):
        cols = ["group", "name", "N_REF", "N_AVG", "N_FOURIER", "GRAD_INIT_FRAC", "SEED0", "verdict", "status", "n_iter_done", "loss_last", "force_first", "force_last", "wall_min"]
        parts.append("<h2>Runs</h2>" + df[[c for c in cols if c in df.columns]].round(4).to_html(index=False))
    for p in pngs:
        b = base64.b64encode(p.read_bytes()).decode()
        parts.append(f"<h2>{p.stem}</h2><img src='data:image/png;base64,{b}'>")
    parts.append("</body></html>")
    out.write_text("\n".join(parts))
    return out
