"""Pass-3 analysis: seed-growth runs, held-out (out-of-sample) evaluation of every final shape, fixed-shape
diagnostics near the optimum, and the extra-mode line probes.   python -m sweep.pass3_analysis

Reads sweep_results/ (pass 1), sweep_results_pass2/, sweep_results_pass3/ and sweep_results_pass3_diag/;
writes CSV tables to sweep_results_pass3/tables/ and prints a text report. Drag values from gradstat jobs are
the objective minus the deterministic curvature penalty, i.e. the mean drag force in raw units.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIMPLE = Path(__file__).resolve().parents[1]
if str(SIMPLE) not in sys.path:
    sys.path.insert(0, str(SIMPLE))

DIAG = SIMPLE / "sweep_results_pass3_diag"
P3 = SIMPLE / "sweep_results_pass3"
LAB10 = ["c0"] + [f"a{k}" for k in range(1, 11)] + [f"b{k}" for k in range(1, 11)]


def _ri():
    import runInflow as ri
    return ri


def pad(C, nf=10):
    C = np.asarray(C, float)
    n = (len(C) - 1) // 2
    out = np.zeros(2 * nf + 1)
    out[0] = C[0]
    m = min(n, nf)
    out[1:1 + m] = C[1:1 + m]
    out[1 + nf:1 + nf + m] = C[1 + n:1 + n + m]
    return out


def gradstat(run_dir):
    """(drag per seed, G, seeds, summary) of one gradstat run dir; drag = objective - curvature penalty."""
    ri = _ri()
    z = np.load(Path(run_dir) / "gradstat.npz")
    s = json.loads((Path(run_dir) / "summary.json").read_text())
    pen = ri.CURV_LAMBDA * ri.curvature_penalty(z["C_eval_eff"]) if ri.CURV_LAMBDA else 0.0
    return dict(drag=z["losses"] - pen, G=z["G"], seeds=z["seeds"], C_eff=z["C_eval_eff"], pen=float(pen), summary=s)


def one(pattern):
    hits = [d for d in glob.glob(str(pattern)) if (Path(d) / "DONE").exists()]
    return hits[0] if hits else None


def growth_runs():
    rows, coeffs, hists = [], {}, {}
    for d in sorted(glob.glob(str(P3 / "adaptive" / "*" / ""))):
        if not (Path(d) / "DONE").exists():
            continue
        s = json.loads((Path(d) / "summary.json").read_text())
        h = np.load(Path(d) / "history.npz")
        ns = h["n_seeds"]
        grow_at = {int(n): int(np.argmax(ns >= n)) for n in (8, 16, 32) if (ns >= n).any()}
        f = h["force"]
        rows.append(dict(name=s["name"], seed0=s["params"]["SEED0"], status=s["status"], iterations=len(f) - 1,
                         seeds_final=int(ns[-1]), grow_to8_at=grow_at.get(8), grow_to16_at=grow_at.get(16), grow_to32_at=grow_at.get(32),
                         drag_change_pct=100 * (f[-1] / f[0] - 1),
                         drag_at50_pct=100 * (f[min(50, len(f) - 1)] / f[0] - 1),
                         grad_signif_end=float(h["grad_signif"][-1]), wall_h=s["wall_total_s"] / 3600))
        coeffs[s["name"]] = np.asarray(s["C_opt_eff"], float)
        hists[s["name"]] = h
    return pd.DataFrame(rows), coeffs, hists


def heldout_table():
    """Held-out drag of every evaluated shape, paired against the circle on the same seeds."""
    circ = one(DIAG / "heldout" / "circle__*")
    if circ is None:
        return pd.DataFrame()
    c = gradstat(circ)
    base = c["drag"]
    rows = []
    for d in sorted(glob.glob(str(DIAG / "heldout" / "*" / ""))):
        if not (Path(d) / "DONE").exists():
            continue
        g = gradstat(d)
        name = g["summary"]["name"]
        assert np.array_equal(g["seeds"], c["seeds"])
        ratio = g["drag"] / base                                     # per-seed paired ratio
        rows.append(dict(shape=name, drag=g["drag"].mean(), drag_se=g["drag"].std(ddof=1) / np.sqrt(len(base)),
                         reduction_pct=100 * (1 - g["drag"].mean() / base.mean()),
                         reduction_se_pct=100 * ratio.std(ddof=1) / np.sqrt(len(ratio))))
    df = pd.DataFrame(rows)
    # in-sample drag change of the run that produced the shape (its own seeds, its own trajectory)
    ins = {}
    for tag, root in (("p1", "sweep_results"), ("p2", "sweep_results_pass2"), ("p3", "sweep_results_pass3")):
        for p in glob.glob(str(SIMPLE / root / "*" / "*" / "summary.json")):
            s = json.loads(Path(p).read_text())
            if s["kind"] == "optimise":
                cl = s.get("classification", {})
                if cl.get("force_first"):
                    ins[f"{tag}_{s['name']}"] = 100 * (1 - cl["force_last"] / cl["force_first"])
    df["in_sample_reduction_pct"] = df["shape"].map(ins)
    df["optimism_pct"] = df["in_sample_reduction_pct"] - df["reduction_pct"]
    return df.sort_values("reduction_pct", ascending=False).reset_index(drop=True)


def probes():
    """Drag along the extra-mode probe lines, paired against the t = 0 shape on the same 16 held-out seeds."""
    ri = _ri()
    sh = json.loads((DIAG / "shapes.json").read_text())
    base = {}
    for key, name in (("C2hi", "C2_250k_nf20"), ("C5hi", "C5_250k_nf10")):
        d = one(DIAG / "diag" / f"{name}__*")
        if d is not None:
            g = gradstat(d)
            base[key] = (g["drag"][:16], g["seeds"][:16], g["pen"], g["C_eff"])
    rows = []
    for d in sorted(glob.glob(str(DIAG / "probe" / "*" / ""))):
        if not (Path(d) / "DONE").exists():
            continue
        g = gradstat(d)
        name = g["summary"]["name"]
        key, t = name.split("_t")
        b_drag, b_seeds, b_pen, b_Ceff = base[key]
        assert np.array_equal(g["seeds"], b_seeds)
        diff = g["drag"] - b_drag
        rows.append(dict(line=key, t=float(t), drag=g["drag"].mean(), d_drag_pct=100 * diff.mean() / b_drag.mean(),
                         d_drag_se_pct=100 * diff.std(ddof=1) / np.sqrt(len(diff)) / b_drag.mean(),
                         d_penalty=g["pen"] - b_pen, c0_eff=float(g["C_eff"][0]), c0_eff_base=float(b_Ceff[0])))
    return pd.DataFrame(rows).sort_values(["line", "t"]).reset_index(drop=True) if rows else pd.DataFrame()


def mode_gradient(run_dir, inner, extra):
    """Noise-corrected gradient norm in two index sets, and the ratio ||g_hat||^2 / noise^2 (about 1 = pure noise)."""
    G = gradstat(run_dir)["G"]
    out = {}
    for tag, idx in (("inner", inner), ("extra", extra)):
        Gs = G[:, idx]
        g = Gs.mean(0)
        tr = Gs.var(0, ddof=1).sum() / len(Gs)
        out[tag] = (float(np.sqrt(max(g @ g - tr, 0.0))), float((g @ g) / tr))
    return out


def _outline(C, n=720):
    C = np.asarray(C, float)
    nf = (len(C) - 1) // 2
    k = np.arange(1, nf + 1)
    th = np.linspace(0, 2 * np.pi, n + 1)
    r = C[0] + np.cos(np.outer(th, k)) @ C[1:nf + 1] + np.sin(np.outer(th, k)) @ C[nf + 1:]
    return r * np.cos(th), r * np.sin(th)


HELDOUT_GROUPS = [   # (row label, shape-name prefixes / exact names)
    ("pass 3: 250k, seeds grow 4 → 32, ≤100 it", lambda s: s.startswith("p3_grow")),
    ("pass 2: 250k × 8 seeds, 50 it", lambda s: s.startswith("p2_nseeds8")),
    ("pass 2: 500k × 4 seeds, 50 it", lambda s: s.startswith("p2_nref500k")),
    ("pass 2: frac 0.1, 4 seeds, 50 it", lambda s: s.startswith("p2_frac0.1")),
    ("pass 2: baseline, 4 seeds, 50 it", lambda s: s.startswith("p2_B0")),
    ("pass 2: 1 seed, 50 it", lambda s: s.startswith("p2_nseeds1")),
    ("pass 2: 5 modes, 50 it", lambda s: s.startswith("p2_nf5")),
    ("pass 1: 10 modes, all knobs, 20 it", lambda s: s.startswith("p1_") and not s.startswith(("p1_nf3", "p1_nf5"))),
    ("pass 1: 5 modes, 20 it", lambda s: s == "p1_nf5"),
    ("pass 1: 3 modes, 20 it", lambda s: s == "p1_nf3"),
]


def make_figures(ho, pr, hists, sh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sweep.analysis import CAT, GRID, INK, INK2, SEQ_BLUE, save_fig   # house palette + rcParams
    fig_dir = P3 / "figures"
    made = []

    # 1. growth runs: drag reduction vs iteration, seed doublings marked
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    rows = []
    marks = {8: ("o", "seeds → 8"), 16: ("s", "seeds → 16"), 32: ("^", "seeds → 32")}
    used = set()
    for name, h in sorted(hists.items()):
        red = 100 * (1 - h["force"] / h["force"][0])
        it = np.arange(len(red))
        ax.plot(it, red, color=SEQ_BLUE[450], lw=1.3, zorder=2)
        ns = h["n_seeds"]
        for n_new, (mk, lab) in marks.items():
            idx = np.nonzero((ns[1:] >= n_new) & (ns[:-1] < n_new))[0] + 1
            for i in idx:
                ax.plot(it[i], red[i], mk, ms=8, mfc=SURFACE_(), mec=INK, mew=1.2, zorder=3,
                        label=(lab if lab not in used else None))
                used.add(lab)
        rows += [dict(run=name, iteration=int(i), drag_reduction_pct=float(v), n_seeds=int(n)) for i, v, n in zip(it, red, ns)]
    ends = [100 * (1 - h["force"][-1] / h["force"][0]) for h in hists.values()]
    ax.text(len(max(hists.values(), key=lambda h: len(h["force"]))["force"]) - 1, max(ends) + 2.5,
            f"4 replicates end at {min(ends):.1f}–{max(ends):.1f}%", ha="right", fontsize=8, color=INK2)
    ax.axhline(62.9, color=INK2, lw=1.0, ls="--", zorder=1)
    ax.text(1, 63.6, "best pass-2 run, 50 iterations (62.9%)", color=INK2, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("drag reduction vs circle (%)")
    ax.set_title("Pass 3: four replicates with seed growth (own seeds, in-sample)")
    ax.set_ylim(0, 78)
    ax.legend(loc="lower right")
    made.append(save_fig(fig, fig_dir / "p3_growth_trajectories", tables=dict(values=pd.DataFrame(rows))))

    # 2. held-out drag reduction by setting (unseen seeds)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rows = []
    groups = [(lab, ho[ho["shape"].map(f)]) for lab, f in HELDOUT_GROUPS]
    groups = [(lab, g) for lab, g in groups if len(g)]
    groups.sort(key=lambda t: t[1]["reduction_pct"].mean())
    for y, (lab, g) in enumerate(groups):
        v = g["reduction_pct"].to_numpy()
        ax.plot([v.min(), v.max()], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter(v, np.full_like(v, y), s=36, color=SEQ_BLUE[450], edgecolor=SURFACE_(), linewidth=1.2, zorder=3)
        ax.plot([v.mean()] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.6, zorder=4)
        ax.annotate(f"{v.mean():.1f}%", (v.max(), y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        rows += [dict(setting=lab, shape=s, reduction_pct=r, se_pct=e) for s, r, e in zip(g["shape"], g["reduction_pct"], g["reduction_se_pct"])]
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([lab for lab, _ in groups])
    ax.set_xlabel("drag reduction vs circle on 16 unseen seeds (%)   ·  dots = runs, bar = mean  (s.e. ≈ 0.1–0.2% per run)")
    ax.set_title("Out-of-sample drag of every optimised shape")
    ax.set_xlim(25, 75)
    ax.grid(axis="y", visible=False)
    made.append(save_fig(fig, fig_dir / "p3_heldout_by_setting", tables=dict(values=pd.DataFrame(rows))))

    # 3. shapes
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    rows = []
    x, y = _outline([1.0] + [0.0] * 20)
    ax.plot(x, y, color=INK2, lw=1.0, ls="--")
    ax.annotate("circle (start)", (0, 1.0), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    shapes = [("C5", "5-mode best (pass 2), −49.5%", CAT[2]), ("C2", "10-mode best (pass 2), −62.5%", CAT[1]),
              ("C3", "10-mode converged (pass 3), −68.7%", CAT[0])]
    for key, lab, col in shapes:
        x, y = _outline(sh[key])
        ax.plot(x, y, color=col, lw=1.8, label=lab)
        rows += [dict(shape=key, x=float(a), y=float(b)) for a, b in zip(x[::4], y[::4])]
    ax.annotate("", xy=(-1.45, -0.9), xytext=(-1.95, -0.9), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    ax.text(-1.95, -0.8, "flow", color=INK2, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 1.4)
    ax.set_ylim(-1.15, 1.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Optimised shapes (fixed perimeter); drag change on unseen seeds")
    made.append(save_fig(fig, fig_dir / "p3_shapes", tables=dict(outlines=pd.DataFrame(rows))))

    # 4. extra-mode probes
    if len(pr):
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        lines = [("C2hi", "10-mode best + modes 11–20", CAT[0]), ("C5hi", "5-mode best + modes 6–10", CAT[1])]
        for key, lab, col in lines:
            g = pr[pr["line"] == key].sort_values("t")
            t = np.r_[0.0, g["t"].to_numpy()]
            d = np.r_[0.0, g["d_drag_pct"].to_numpy()]
            e = np.r_[0.0, g["d_drag_se_pct"].to_numpy()]
            ax.errorbar(t, d, yerr=e, color=col, lw=2.0, marker="o", ms=6, mfc=col, mec=SURFACE_(), mew=1.2, capsize=0, label=lab)
            ax.annotate(lab, (t[-1], d[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("step into the unused modes (coefficient displacement t)")
        ax.set_ylabel("drag change on unseen seeds (%)")
        ax.set_title("Adding Fourier modes to an optimised shape still lowers drag")
        ax.legend(loc="lower left")
        made.append(save_fig(fig, fig_dir / "p3_mode_probes", tables=dict(values=pr)))
    return made


def SURFACE_():
    from sweep.analysis import SURFACE
    return SURFACE


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    out = P3 / "tables"
    out.mkdir(parents=True, exist_ok=True)
    runs, coeffs, hists = growth_runs()
    print("=== pass-3 growth runs ===")
    print(runs.round(3).to_string(index=False))
    runs.to_csv(out / "growth_runs.csv", index=False)
    if coeffs:
        M = np.array([pad(c) for c in coeffs.values()])
        tab = pd.DataFrame(dict(mode=LAB10, mean=M.mean(0), sd=(M.std(0, ddof=1) if len(M) > 1 else np.nan)))
        for n, c in coeffs.items():
            tab[n] = pad(c)
        tab.to_csv(out / "growth_final_coefficients.csv", index=False)
        print("\nfinal coefficients (mean, sd over replicates):")
        print(tab.round(4).to_string(index=False))
        if len(M) > 1:
            dist = [np.linalg.norm(M[i] - M[j]) for i in range(len(M)) for j in range(i + 1, len(M))]
            print(f"pairwise replicate distance: {np.round(dist, 3)}  mean {np.mean(dist):.3f}")
    ho = heldout_table()
    if len(ho):
        print("\n=== held-out drag reduction vs circle (16 unseen seeds, paired) ===")
        print(ho.round(2).to_string(index=False))
        ho.to_csv(out / "heldout_drag.csv", index=False)
    pr = probes()
    if len(pr):
        print("\n=== extra-mode line probes (paired vs t = 0) ===")
        print(pr.round(3).to_string(index=False))
        pr.to_csv(out / "probes.csv", index=False)
    c3 = one(DIAG / "diag" / "C3_250k_nf20__*")
    if c3 is not None:
        inner = [0] + list(range(1, 11)) + list(range(21, 31))
        extra = list(range(11, 21)) + list(range(31, 41))
        mg = mode_gradient(c3, inner, extra)
        print(f"\n=== gradient at the pass-3 consensus C3 (32 held-out seeds) ===\n  modes 1-10: corrected ||g|| {mg['inner'][0]:.0f} "
              f"(ratio {mg['inner'][1]:.1f});  modes 11-20: corrected ||g|| {mg['extra'][0]:.0f} (ratio {mg['extra'][1]:.1f})")
    sh = json.loads((DIAG / "shapes.json").read_text())
    if len(ho) and "C3" in sh:
        for p in make_figures(ho, pr, hists, sh):
            print("figure:", p)


if __name__ == "__main__":
    main()
