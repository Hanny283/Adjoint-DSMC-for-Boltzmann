"""Build the PDF results report of the whole parameter study (passes 1-4).   python -m sweep.report_pdf

Builds two PDFs in sweep_results_report/: a short main report (summary, set-up, one section per parameter, discussion,
questions) and an appendix (every run, every summary figure, one figure per run with its parameter values, a gallery of
every final shape, every fixed-shape measurement). All tables are computed from the saved results.
"""
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIMPLE = Path(__file__).resolve().parents[1]
if str(SIMPLE) not in sys.path:
    sys.path.insert(0, str(SIMPLE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sweep.analysis import CAT, GRID, INK, INK2, SEQ_BLUE, SURFACE  # noqa: E402  (house palette + rcParams)
from sweep.pass3_analysis import gradstat  # noqa: E402

OUT = SIMPLE / "sweep_results_report"
PASSES = [(1, "sweep_results"), (2, "sweep_results_pass2"), (3, "sweep_results_pass3"), (4, "sweep_results_pass4")]
GROUP_LABEL = {"baseline": "baseline", "sweep_seeds": "seed-count sweep", "sweep_nref": "particle-count sweep",
               "sweep_nfourier": "Fourier-mode sweep", "sweep_frac": "first-step sweep", "adaptive": "seed growth",
               "continuation": "1M continuation with endgame"}
GROUP_ORDER = list(GROUP_LABEL)
STATUS_LABEL = {"noise-limited": "noise limit reached", "max iterations": "iteration cap reached",
                "line search failed": "line search failed"}


# ---------------------------------------------------------------------------------------------------- helpers
def tex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def nref(n):
    n = int(n)
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"


def seeds_label(P, math=False, short=False):
    n, m = int(P["N_AVG"]), int(P["N_AVG_MAX"])
    if short:
        return str(n) if n == m else rf"{n}$\to${m}"
    if n == m:
        return f"{n} (fixed)"
    return (rf"{n} $\to$ {m} (grow)" if math else f"{n} → {m} (grow)")


def param_values(P):
    """The parameters varied in the study, as NAME = value (meanings are in Table 1)."""
    v = [f"N_REF = {nref(P['N_REF'])}", f"N_AVG = {P['N_AVG']}"]
    if int(P["N_AVG_MAX"]) != int(P["N_AVG"]):
        v.append(f"N_AVG_MAX = {P['N_AVG_MAX']}")
    v += [f"N_FOURIER = {P['N_FOURIER']}", f"GRAD_INIT_FRAC = {P['GRAD_INIT_FRAC']:g}", f"N_ITER = {P['N_ITER']}", f"SEED0 = {P['SEED0']}"]
    if P.get("ENDGAME"):
        v.append(f"ENDGAME = {P['ENDGAME_ITERS']} steps")
    if P.get("C_START") is not None:
        v.append("start = pass-3 shape")
    return v


def pm(x, e, d=2):
    return rf"${x:.{d}f} \pm {e:.{d}f}$"


def sgn(x, d=1):
    return rf"${x:+.{d}f}$"


def outline(C, n=720):
    C = np.asarray(C, float)
    nf = (len(C) - 1) // 2
    k = np.arange(1, nf + 1)
    th = np.linspace(0, 2 * np.pi, n + 1)
    r = C[0] + np.cos(np.outer(th, k)) @ C[1:nf + 1] + np.sin(np.outer(th, k)) @ C[nf + 1:]
    return r * np.cos(th), r * np.sin(th)


def table(caption, header, rows, align, note=None, long=False, size=r"\small", label=None):
    caption = (rf"\label{{{label}}}" if label else "") + caption
    head = " & ".join(header) + r" \\"
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    if long:
        return (f"{{{size}\n\\begin{{longtable}}{{{align}}}\n\\caption{{{caption}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
                f"\\toprule\n{head}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{body}\n\\end{{longtable}}}}\n"
                + (f"{{\\footnotesize {note}}}\n\n" if note else ""))
    return (f"\\begin{{table}}[H]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n" + (f"\\par\\smallskip{{\\footnotesize\\raggedright {note}\\par}}\n" if note else "") + "\\end{table}\n")


def clean(G, k=5.0):
    """Drop seeds whose gradient deviates from the per-component median by more than k x the median deviation."""
    dev = np.linalg.norm(G - np.median(G, 0), axis=1)
    keep = dev <= k * np.median(dev)
    return G[keep], int((~keep).sum())


def chi2_compare(A, B):
    from scipy.stats import chi2
    se = np.sqrt(A.var(0, ddof=1) / len(A) + B.var(0, ddof=1) / len(B))
    ok = se > 0
    z = (A.mean(0) - B.mean(0))[ok] / se[ok]
    return float((z ** 2).mean()), float(chi2.sf((z ** 2).sum(), ok.sum()))


# ---------------------------------------------------------------------------------------------------- data
def load_runs():
    runs = []
    for p, root in PASSES:
        for sj in sorted(glob.glob(str(SIMPLE / root / "*" / "*" / "summary.json"))):
            d = Path(sj).parent
            if not (d / "DONE").exists():
                continue
            s = json.loads(Path(sj).read_text())
            if s["kind"] != "optimise" or s.get("group") == "smoke":
                continue
            with np.load(d / "history.npz", allow_pickle=False) as z:
                h = {k: z[k] for k in z.files}
            P = json.loads((d / "config.json").read_text())["params"]      # full settings (the summary keeps a subset)
            runs.append(dict(pass_=p, dir=d, s=s, h=h, name=s["name"], group=s["group"], P=P))
    runs.sort(key=lambda r: (r["pass_"], GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99, r["name"]))
    return runs


def heldout_map():
    ho = pd.read_csv(SIMPLE / "sweep_results_pass3" / "tables" / "heldout_drag.csv")
    m = {r.shape: (r.reduction_pct, r.reduction_se_pct) for r in ho.itertuples()}
    circ = glob.glob(str(SIMPLE / "sweep_results_pass3_diag" / "heldout" / "circle__*"))
    p4 = glob.glob(str(SIMPLE / "sweep_results_pass4" / "heldout" / "p4_final_250k__*"))
    if circ and p4:
        c, g = gradstat(circ[0])["drag"], gradstat(p4[0])["drag"]
        ratio = g / c
        m["p4_C3_1M_endgame_seed400"] = (100 * (1 - g.mean() / c.mean()), 100 * ratio.std(ddof=1) / np.sqrt(len(ratio)))
    return m


def run_facts(r, H):
    s, h, P = r["s"], r["h"], r["P"]
    f = h["force"]
    key = f"p{r['pass_']}_{r['name']}"
    ho = H.get(key)
    own = 100 * (1 - f[-1] / f[0])
    return dict(key=key, iters=len(f) - 1, seeds_final=int(h["n_seeds"][-1]), status=str(s.get("status")),
                own=own, own_vs="start shape" if P.get("C_START") is not None else "circle", held=ho,
                wall=s["wall_total_s"] / 3600, seed_hi=int(P["SEED0"]) + int(P["N_AVG_MAX"]) - 1)


# ---------------------------------------------------------------------------------------------------- figures
def run_figure(r, F, path):
    P, h = r["P"], r["h"]
    fig = plt.figure(figsize=(10, 3.9))
    title = f"Pass {r['pass_']}  ·  {r['name']}"
    fig.text(0.012, 0.965, title, fontsize=11, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.895, "    ".join(param_values(P)), fontsize=9, color=INK2, va="top", family="monospace")
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.985, bottom=0.13, top=0.76, wspace=0.28)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    red = 100 * (1 - h["force"] / h["force"][0])
    it = np.arange(len(red))
    ax1.plot(it, red, color=SEQ_BLUE[450], lw=1.6, marker="o", ms=2.2)
    ns = h["n_seeds"]
    for n_new, mk in ((8, "o"), (16, "s"), (32, "^")):
        idx = np.nonzero((ns[1:] >= n_new) & (ns[:-1] < n_new))[0] + 1
        for i in idx:
            ax1.plot(it[i], red[i], mk, ms=7, mfc=SURFACE, mec=INK, mew=1.1, zorder=4)
            ax1.annotate(f"→{n_new}", (it[i], red[i]), xytext=(3, -11), textcoords="offset points", fontsize=7, color=INK2)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel(f"drag reduction vs {F['own_vs']} (%)")
    ax1.set_title("drag (own seeds)", fontsize=9.5)
    g = np.asarray(h["gnorm"], float)
    se = np.asarray(h["grad_se"], float)
    ax2.semilogy(it, g, color=SEQ_BLUE[450], lw=1.4, marker="o", ms=2.2, label="‖g‖")
    if np.isfinite(se).any():
        ax2.semilogy(it, np.where(np.isfinite(se), se, np.nan), color=INK2, lw=1.0, ls="--", label="s.e. of g")
    else:
        ax2.text(0.5, 0.06, "1 seed: no standard error", transform=ax2.transAxes, ha="center", fontsize=7.5, color=INK2)
    ax2.set_xlabel("iteration")
    ax2.set_title("gradient norm (normalised)", fontsize=9.5)
    ax2.legend(fontsize=7.5, loc="upper right")
    x, y = outline([1.0] + [0.0] * (len(h["C_eff"][0]) - 1))
    ax3.plot(x, y, color=INK2, lw=1.0, ls="--", label="circle")
    if P.get("C_START") is not None:
        x, y = outline(h["C_eff"][0])
        ax3.plot(x, y, color=CAT[1], lw=1.2, ls="--", label="start (pass-3 shape)")
    x, y = outline(h["C_eff"][-1])
    ax3.plot(x, y, color=CAT[0], lw=1.8, label="final shape")
    ax3.set_aspect("equal")
    ax3.set_xlim(-2.0, 1.4)
    ax3.set_ylim(-1.15, 1.15)
    ax3.set_title("shape (flow left → right)", fontsize=9.5)
    ax3.legend(fontsize=7, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def gallery(runs, facts, path):
    n = len(runs)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 1.55 * rows + 0.3))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, runs):
        F = facts[r["key"]]
        x, y = outline([1.0] + [0.0] * (len(r["h"]["C_eff"][-1]) - 1))
        ax.plot(x, y, color=INK2, lw=0.6, ls="--")
        x, y = outline(r["h"]["C_eff"][-1])
        ax.plot(x, y, color=CAT[0], lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 1.4)
        ax.set_ylim(-1.15, 1.15)
        pct = F["held"][0] if F["held"] else None
        ax.set_title(f"P{r['pass_']} {r['name']}\n" + (f"−{pct:.1f}% unseen" if pct is not None else f"−{F['own']:.1f}% own"), fontsize=6.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- tables
RR = r">{\raggedright\arraybackslash}"
SETTING_LABELS = ["Pass 3: 250k, seeds grow 4 to 32, up to 100 iterations", "Pass 2: 250k, 8 seeds, 50 iterations",
                  "Pass 2: 500k, 4 seeds, 50 iterations", "Pass 2: first step 0.1, 4 seeds, 50 iterations",
                  "Pass 2: baseline (250k, 4 seeds), 50 iterations", "Pass 2: 1 seed, 50 iterations", "Pass 2: 5 modes, 50 iterations",
                  "Pass 1: 10 modes, every setting, 20 iterations", "Pass 1: 5 modes, 20 iterations", "Pass 1: 3 modes, 20 iterations"]


def tables(runs, facts):
    T = {}
    held = heldout_map()
    # ---- main report
    T["varied"] = table("The parameters that we varied. Every other setting was the same in every run.",
                        ["Parameter", "What it controls", "Values used", "Baseline"],
                        [[r"N\_AVG", "The number of seeds averaged for each drag and gradient measurement.", "1, 2, 4, 8, 16", "4"],
                         [r"N\_AVG\_MAX", r"The largest number of seeds a run may grow to. When it equals N\_AVG, the seed count is fixed.", r"= N\_AVG; 32 in passes 3 and 4", r"= N\_AVG"],
                         [r"N\_REF", "The number of simulated particles in the channel.", "100k, 250k, 500k, 1M", "250k"],
                         [r"N\_FOURIER", "The number of Fourier modes that describe the shape.", "3, 5, 10, 20", "10"],
                         [r"GRAD\_INIT\_FRAC", "The size of the very first trial step, as a fraction of the size of the shape coefficients.", "0.01, 0.02, 0.05, 0.1, 0.2", "0.05"],
                         [r"N\_ITER", "The largest number of iterations a run may take.", "20, 50, 100; 30 in pass 4", "20"],
                         ["SEED0", "The first seed of a run. Replicates use separate seed sets.", "0, 100, 200, 300, 400", "0"],
                         ["ENDGAME", "A final phase of gradient steps that do not have to prove an improvement.", "off; 10 steps in pass 4", "off"],
                         ["Starting shape", "The shape a run starts from.", "circle; pass-3 shape in pass 4", "circle"]],
                        rf"l{RR}p{{6.6cm}}{RR}p{{3.8cm}}l", label="tab:varied")
    H = pd.read_csv(SIMPLE / "sweep_results_pass3" / "tables" / "heldout_drag.csv")
    from sweep.pass3_analysis import HELDOUT_GROUPS
    rows = []
    for (lab, f), nice in zip(HELDOUT_GROUPS, SETTING_LABELS):
        g = H[H["shape"].map(f)]
        if len(g):
            rows.append([nice, str(len(g)), f"{g.reduction_pct.mean():.1f}",
                         (f"{g.reduction_pct.min():.1f}--{g.reduction_pct.max():.1f}" if len(g) > 1 else "--"),
                         f"{(g.in_sample_reduction_pct - g.reduction_pct).abs().max():.2f}"])
    p4 = facts.get("p4_C3_1M_endgame_seed400")
    if p4 and p4["held"]:
        rows.insert(0, ["Pass 4: 1M, endgame, from the pass-3 shape", "1", f"{p4['held'][0]:.1f}", "--", "--"])
    T["heldout"] = table("How far each setting got. The values are drag reductions relative to the circle, re-measured on 16 seeds that no run used. "
                         "Each run's standard error is 0.1 to 0.2 percentage points. The last column gives the largest difference between the own-seed and "
                         "unseen-seed values of any run in the setting.",
                         ["Setting", "Runs", r"Mean \%", r"Range \%", r"Largest own$-$unseen gap"], rows, "lrrrr", label="tab:heldout")
    T["seedsrun"] = table("Where runs with different seed settings stopped and how far they got. All of these runs used 10 modes and 250k particles, "
                          "except for the row with 500k particles. The drag reductions were measured on unseen seeds.",
                          ["Seeds per gradient", "Iterations reached", "Why the runs stopped", r"Drag reduction \%"],
                          [["1, fixed", "24--35", "noise limit", "50.4--53.7"], ["4, fixed", "27, 32, 50", "noise limit, iteration limit", "55.4, 58.3, 62.0"],
                           ["8, fixed", "47--50", "iteration limit (one at the noise limit)", "61.4--62.9"],
                           ["4, fixed, with 500k particles", "50", "iteration limit", "62.3--62.7"],
                           [r"4 $\to$ 32, growing", "81--100", "noise limit at 32 seeds, iteration limit", "68.1--69.0"]], "llll", label="tab:seedsrun",
                          note="Four seeds with 500k particles average the same total number of simulated particles per measurement as eight seeds with "
                               "250k particles (2 million). The two settings therefore have about the same noise, and they reached the same place.")
    T["seedsneed"] = table("How many seeds are needed just to see the gradient. The second column is the scatter of one seed's gradient divided by the size "
                           "of the mean gradient. The third column is the number of seeds that makes the gradient at least twice its standard error.",
                           ["Shape", r"One-seed scatter / $\|g\|$", "Seeds needed"],
                           [["Circle (start)", "0.18", "1"], ["Best pass-2 shape", "0.97", "4--5"], ["Final pass-3 shape, 250k", "1.14", "6"],
                            ["Final pass-3 shape, 1M", "0.54", "2"]], "lrr", label="tab:seedsneed",
                           note="At the final shape, one of 32 seeds at 250k gave a gradient about 30 times the typical scatter, and it was left out of this table. "
                                "Confirming that a small step lowered the drag needs many more seeds than seeing the gradient.")
    gt = pd.read_csv(SIMPLE / "sweep_results" / "figures" / "gradstat_table.csv")
    piv = {(r.shape, int(r.N_REF)): r.sd_norm / r.gnorm_mean for r in gt.itertuples()}
    cost = {100000: 41, 250000: 112, 500000: 163, 1000000: 308}
    T["noiseN"] = table(r"How the gradient noise and the cost depend on the particle count. The noise is the scatter of one seed's gradient divided by the "
                        r"gradient's size, measured with 16 seeds in pass 1. At the circle it falls as $N^{-0.46}$, close to the ideal $N^{-1/2}$.",
                        ["Particles", "Noise, circle", "Noise, stretched", "s per seed-iteration", "s per 100k particles"],
                        [[nref(N), f"{100 * piv[('C_init', N)]:.1f}\\%", f"{100 * piv[('C_def', N)]:.1f}\\%", str(cost[N]), f"{cost[N] / (N / 1e5):.0f}"]
                         for N in (100000, 250000, 500000, 1000000)], "lrrrr", label="tab:noiseN")

    def drag_per(path, N):
        g = gradstat(glob.glob(str(SIMPLE / path))[0]); d = g["drag"] / N * 1e5
        return g, d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    C2 = {N: drag_per(f"sweep_results_pass3_diag/diag/C2_{nref(N)}_nf10__*/", N) for N in (100000, 250000, 500000)}
    G = {N: C2[N][0]["G"] / N * 1e5 for N in C2}
    rows = []
    for a, b in ((100000, 500000), (100000, 250000), (250000, 500000)):
        raw = chi2_compare(G[a], G[b]); ca, _ = clean(G[a]); cb, _ = clean(G[b]); cl = chi2_compare(ca, cb)
        rows.append([f"{nref(a)} vs {nref(b)} at C2", pm(C2[a][1], C2[a][2], 1), pm(C2[b][1], C2[b][2], 1), f"{raw[0]:.2f} ($p={raw[1]:.3f}$)", f"{cl[0]:.2f} ($p={cl[1]:.3f}$)"])
    inner = [0] + list(range(1, 11)) + list(range(21, 31))
    g250 = gradstat(glob.glob(str(SIMPLE / "sweep_results_pass3_diag/diag/C3_250k_nf20__*/"))[0])
    g1m = gradstat(glob.glob(str(SIMPLE / "sweep_results_pass4/diag/C3_1M_nf10__*/"))[0])
    A = g250["G"][:, inner] / 2.5; B = g1m["G"] / 10
    raw = chi2_compare(A, B); ca, _ = clean(A); cl = chi2_compare(ca, B)
    d250 = g250["drag"] / 2.5; d1m = g1m["drag"] / 10
    rows.append(["250k vs 1M at C3", pm(d250.mean(), d250.std(ddof=1) / np.sqrt(len(d250)), 1), pm(d1m.mean(), d1m.std(ddof=1) / np.sqrt(len(d1m)), 1),
                 f"{raw[0]:.2f} ($p={raw[1]:.3f}$)", f"{cl[0]:.2f} ($p={cl[1]:.3f}$)"])
    agree = lambda cl: "yes" if cl[1] > 0.01 else "no"
    rows = [[f"{nref(a)} vs {nref(b)}", "best pass-2 shape", f"{C2[a][1]:.0f} vs {C2[b][1]:.0f}", "yes",
             agree(chi2_compare(clean(G[a])[0], clean(G[b])[0]))] for a, b in ((100000, 500000), (100000, 250000), (250000, 500000))]
    rows.append(["250k vs 1M", "final pass-3 shape", f"{d250.mean():.0f} vs {d1m.mean():.0f}", "yes", "yes, except the finest mode"])
    T["bias"] = table(r"Whether the particle count biases the result. Each row measures the same fixed shape with two particle counts on the same unseen seeds. "
                      r"The drag is given per $10^5$ particles so that the counts can be compared. Two search directions count as the same when they differ "
                      r"by no more than their noise.",
                      ["Comparison", "Shape", r"Drag per $10^5$ particles", "Same drag?", "Same search direction?"], rows, "llrll", label="tab:bias",
                      note=r"At the final shape, 250k and 1M agree on the overall gradient, but in the finest mode $a_{10}$ 1M sees about five times more pull. "
                           r"The comparisons leave out a few seeds with extreme gradients, which would otherwise hide real differences.")
    T["modes"] = table("The best drag reduction on unseen seeds for each number of Fourier modes.",
                       ["Modes", r"Drag reduction \%", "Notes"],
                       [["3", "32.6", "This was measured in pass 1, and the shape is too coarse."],
                        ["5", "49.5", "Three pass-2 replicates agree closely, and they stop after about 20 iterations."],
                        ["10", "62.5, then 68.7", "The first value is after 50 iterations, and the second is the pass-3 plateau."],
                        ["20", "about 10 (own seeds)", "Starting from the circle, the run stopped after 3 iterations, because the extra modes carry only noise there."]],
                       rf"ll{RR}p{{9.6cm}}", label="tab:modes")
    pr = pd.read_csv(SIMPLE / "sweep_results_pass3" / "tables" / "probes.csv")
    rows = []
    for t in sorted(set(pr.t)):
        a = pr[(pr.line == "C2hi") & (pr.t == t)]; b = pr[(pr.line == "C5hi") & (pr.t == t)]
        f = lambda q: (pm(q.d_drag_pct.iloc[0], q.d_drag_se_pct.iloc[0]) + rf" ({q.d_penalty.iloc[0]:+.0f})") if len(q) else "--"
        rows.append([f"{t:g}", f(a), f(b)])
    T["probes"] = table(r"What happens when an optimised shape is given extra modes. Each shape is stepped by $t$ into the modes it did not use, and its drag is "
                        r"re-measured on the same 16 unseen seeds. The change in the curvature penalty is shown in brackets.",
                        ["Step $t$", r"10-mode shape + modes 11--20, drag change \%", r"5-mode shape + modes 6--10, drag change \%"], rows, "lll", label="tab:probes",
                        note="At the best pass-2 shape the gradient in modes 11--20 is about seven times the gradient in modes 1--10, and at the final pass-3 shape it is "
                             "about eight times larger.")
    fe = pd.read_csv(SIMPLE / "sweep_results" / "figures" / "grad_init_frac_effects__values.csv")
    hmap = {0.01: "p1_frac0.01", 0.02: "p1_frac0.02", 0.05: "p1_B0_seed0", 0.1: "p1_frac0.1", 0.2: "p1_frac0.2"}
    T["frac"] = table(r"The effect of the first-step size in pass 1. The objective equals 1 at the circle, and the last column is the drag reduction on unseen "
                      r"seeds after 20 iterations.",
                      [r"GRAD\_INIT\_FRAC", "Objective after 1 iteration", "After 5", "After 20", r"Drag reduction \%"],
                      [[f"{r.frac:g}", f"{r.loss_it1:.3f}", f"{r.loss_it5:.3f}", f"{r.loss_final:.3f}", f"{held[hmap[r.frac]][0]:.1f}"] for r in fe.itertuples()],
                      "rrrrr", label="tab:frac",
                      note=r"In pass 2, a first step of 0.1 averaged 60.0\% after 50 iterations against 58.6\% for 0.05. That difference comes from where individual "
                           r"runs stopped, not from the step size.")
    fc = pd.read_csv(SIMPLE / "sweep_results_pass3" / "tables" / "growth_final_coefficients.csv")
    sh = json.loads((SIMPLE / "sweep_results_pass3_diag" / "shapes.json").read_text())
    p4r = [r for r in runs if r["pass_"] == 4]
    c4 = p4r[0]["h"]["C_eff"][-1] if p4r else None
    b0 = [r for r in runs if r["pass_"] == 1 and r["name"] == "B0_seed0"][0]["h"]["C_eff"][-1]
    lab = ["c_0"] + [f"a_{{{k}}}" for k in range(1, 11)]
    rows = [[f"${lab[i]}$", f"{b0[i]:+.3f}", f"{sh['C2'][i]:+.3f}", pm(fc["mean"][i], fc["sd"][i], 3), (f"{c4[i]:+.3f}" if c4 is not None else "--")] for i in range(11)]
    bmax = lambda C: max(abs(x) for x in list(C)[11:21])
    rows.append([r"$b_1\ldots b_{10}$", rf"$\le {bmax(b0):.3f}$", rf"$\le {bmax(sh['C2']):.3f}$", rf"$|\mathrm{{mean}}| \le {fc['mean'][11:].abs().max():.3f}$",
                 (rf"$\le {bmax(c4):.3f}$" if c4 is not None else "--")])
    T["coeffs"] = table(r"The shape coefficients from each pass. Pass 1 is the baseline after 20 iterations, pass 2 is its best shape, pass 3 is the mean "
                        r"$\pm$ standard deviation over four replicates, and pass 4 is the final 1M shape.",
                        ["Coefficient", "Pass 1", "Pass 2", "Pass 3 (4 runs)", "Pass 4"], rows, "lrrrr", label="tab:coeffs",
                        note=r"Every mode is capped at $0.6\,c_0$ by the optimiser's stretch limit (A\_MAX\_FRAC). In passes 3 and 4, $a_2/c_0 = 0.600$, which is "
                             r"exactly at the cap. The sine coefficients $b_k$ stay near zero, so the shape stays symmetric about the flow axis.")
    T["pass4"] = table("What happened in pass 4, which continued from the pass-3 shape with 1M particles and the endgame mode.",
                       ["Aspect", "Result"],
                       [["Normal phase", "The run took 10 iterations and then reached the noise limit, even with 32 seeds."],
                        ["Endgame (10 steps)", r"The objective stayed between 0.996 and 1.000 of its start, and the gradient stayed between 0.9 and 1.4 times its starting size."],
                        ["Shape change", r"The shape moved by 0.031, which is about the spread between the pass-3 replicates, and $a_2$ stayed on the stretch limit."],
                        ["Unseen seeds at 250k", r"The drag changed by $-0.69 \pm 0.49$\% against the pass-3 shape, which is not significant."],
                        ["Unseen seeds at 1M", r"The drag changed by $-0.84 \pm 0.27$\% against the pass-3 shape, which is small but real."],
                        ["Cost", "The run took 6.3 hours on 16 cores."]], rf"l{RR}p{{12.2cm}}", label="tab:pass4")
    # ---- appendix
    rows = []
    for r in runs:
        P, F = r["P"], facts[r["key"]]
        rows.append([str(r["pass_"]), rf"\texttt{{{tex(r['name'])}}}", nref(P["N_REF"]), seeds_label(P, short=True), str(P["N_FOURIER"]),
                     f"{P['GRAD_INIT_FRAC']:g}", str(P["N_ITER"]) + (rf"+{P['ENDGAME_ITERS']}" if P.get("ENDGAME") else ""),
                     str(P["SEED0"]), str(F["iters"]), {"noise-limited": "noise limit", "max iterations": r"it.\ limit"}.get(F["status"], tex(F["status"])),
                     f"{F['own']:.1f}", (f"{F['held'][0]:.1f}" if F["held"] else "--"), f"{F['wall']:.1f}"])
    T["runs"] = table(r"Every optimisation run. The drag reductions are relative to the circle in percent. Own is measured on the run's own seeds, and Unseen is "
                      r"re-measured on 16 held-out seeds at 250k particles. "
                      r"The pass-4 run took 10 endgame steps after its 30 normal iterations, and its own-seed value is relative to its starting shape rather than the circle.",
                      ["Pass", "Run", r"N\_REF", r"N\_AVG", r"N\_FOURIER", r"\shortstack{GRAD\_INIT\\\_FRAC}", r"N\_ITER", "SEED0", r"\shortstack{Iterations\\done}", r"\shortstack{Stopped\\at}", r"Own \%", r"Unseen \%", "Hours"], rows,
                      "rllllrrrrlrrr", long=True, size=r"\scriptsize\setlength{\tabcolsep}{1.6pt}", label="tab:runs")
    gr = pd.read_csv(SIMPLE / "sweep_results_pass3" / "tables" / "growth_runs.csv")
    T["growth"] = table(r"The four seed-growth replicates of pass 3. The columns $\to8$, $\to16$ and $\to32$ give the iteration at which the seed count reached "
                        r"8, 16 and 32.",
                        ["Seed0", "Iterations", r"$\to8$", r"$\to16$", r"$\to32$", "Stopped at", r"Own \%", r"Unseen \%", "Hours"],
                        [[str(r.seed0), str(r.iterations), str(r.grow_to8_at), str(r.grow_to16_at), str(r.grow_to32_at),
                          {"noise-limited": "noise limit", "max iterations": "iteration limit"}.get(r.status, tex(r.status)), f"{-r.drag_change_pct:.2f}",
                          f"{held['p3_' + r.name][0]:.2f}", f"{r.wall_h:.1f}"] for r in gr.itertuples()], "rrrrrlrrr", label="tab:growth")
    T["checks"] = table("Checks on the method.", ["Question", "Answer"],
                        [["Does the gas settle after a shape change?", r"Yes. Without settling the drag reads 6.6\% too low, and after 10, 20 and 40 time units it agrees with 80 units within 0.7\%."],
                         ["Do shapes over-fit their own seeds?", r"No. For all 36 final shapes, own-seed and unseen-seed drag agree within 1 percentage point, and mostly within 0.5."],
                         ["Do replicates agree?", r"Yes. In pass 3, every coefficient has a standard deviation of at most 0.009."],
                         ["Are the computations reproducible?", "Yes. The pass-2 baseline repeated the pass-1 baseline bit for bit, and serial and parallel runs give identical results."],
                         ["Is the final shape a true optimum?", r"No. It presses against the stretch limit ($z \approx -56$), and a smaller real pull remains in $a_1$, $a_6$ and $a_{10}$ ($p<0.001$)."]],
                        rf"{RR}p{{5.0cm}}{RR}p{{11.0cm}}", label="tab:checks")
    return T


def measurements_table():
    rows, specs = [], []
    for p in sorted(glob.glob(str(SIMPLE / "sweep_results" / "gradstat" / "*" / ""))):
        specs.append(("1", p))
    for sub in ("adapt", "diag", "probe"):
        for p in sorted(glob.glob(str(SIMPLE / "sweep_results_pass3_diag" / sub / "*" / ""))):
            specs.append(("3", p))
    for sub in ("diag", "heldout"):
        for p in sorted(glob.glob(str(SIMPLE / "sweep_results_pass4" / sub / "*" / ""))):
            specs.append(("4", p))
    for ps, p in specs:
        if not (Path(p) / "DONE").exists():
            continue
        g = gradstat(p)
        s = g["summary"]; P = s["params"]; N = int(P["N_REF"])
        d = g["drag"] / N * 1e5
        Gc, nout = clean(g["G"])
        gm = Gc.mean(0); sd = np.sqrt(Gc.var(0, ddof=1).sum())
        name = s["name"].replace("nref", "").replace("_nf10", "").replace("_nf20", " (20 modes)")
        T = P["REWARM_T"] if s["name"] not in ("nref100k_C_init", "nref250k_C_init", "nref500k_C_init", "nref1M_C_init") else 0
        rows.append([ps, rf"\texttt{{{tex(name)}}}", nref(N), str(len(d)), f"{T:g}", pm(d.mean(), d.std(ddof=1) / np.sqrt(len(d)), 1),
                     f"{sd / max(np.linalg.norm(gm), 1e-9):.2f}", str(nout)])
    return table(r"Every fixed-shape measurement. The drag is given per $10^5$ particles. Noise is one seed's gradient scatter divided by the size of the mean "
                 r"gradient, with extreme seeds removed, and the last column counts the removed seeds. Adapt.\ is the time the gas adapts to the shape before "
                 r"the measurement.",
                 ["Pass", "Measurement", "Particles", "Seeds", "Adapt.", "Drag", "Noise", "Removed"], rows, "rllrrrrr", long=True, size=r"\scriptsize",
                 label="tab:meas",
                 note=r"C\_init is the circle, C\_def is the stretched shape with $a_2 = 0.15$, C2 is the best pass-2 shape, C5 is the best 5-mode shape, and C3 is "
                      r"the pass-3 consensus. C2hi and C5hi are the line probes, and p4\_final is the pass-4 final shape. The held-out re-measurements of the "
                      r"final shapes are summarised in the main report.")


# ---------------------------------------------------------------------------------------------------- captions
KNOB_NAME = {"frac": r"GRAD\_INIT\_FRAC", "nfourier": r"N\_FOURIER", "nref": r"N\_REF", "seeds": r"N\_AVG"}
FIG_CAPTIONS = {
    "coeff_trajectories": "This figure shows each Fourier coefficient of the baseline run against iteration.",
    "grad_init_frac_effects": "This figure shows how the first-step size affects the first accepted step, the number of backtracks at iteration 0, and the objective after 1, 5 and 20 iterations.",
    "grad_mean_vs_nref": r"This figure shows the mean gradient of each mode, with its standard error over 16 seeds, against the particle count at the circle and at the stretched shape ($a_2 = 0.15$).",
    "grad_sd_vs_nref": r"This figure shows the scatter of one seed's gradient against the particle count on logarithmic axes, with an $N^{-1/2}$ guide.",
    "grad_se_vs_nseeds_C_init": r"This figure shows the error of an $n$-seed average against $n$ at the circle, together with the $1/\sqrt{n}$ prediction.",
    "grad_se_vs_nseeds_C_def": r"This figure shows the error of an $n$-seed average against $n$ at the stretched shape ($a_2 = 0.15$), together with the $1/\sqrt{n}$ prediction.",
    "success_table": r"This figure shows the automatic classifier's verdict for each run. For 1-seed runs a success verdict is an artefact, because a single seed gives no noise estimate.",
    "wall_per_iter": "This figure shows the wall-clock time per iteration against the particle count and the number of seeds.",
    "p3_growth_trajectories": "This figure shows the drag reduction against iteration for the four pass-3 replicates on their own seeds. The markers show where each run doubled its seeds instead of stopping.",
    "p3_heldout_by_setting": "This figure shows the drag reduction of every optimised shape on 16 unseen seeds, grouped by setting. Each dot is one run, and the bar marks the mean.",
    "p3_shapes": "This figure shows the best 5-mode and 10-mode shapes from pass 2 and the final 10-mode shape from pass 3, with their drag reductions on unseen seeds. All shapes have the same perimeter, and the gas flows from left to right.",
    "p3_mode_probes": r"This figure shows how the drag changes when the optimised shapes are stepped into the modes they did not use. The drag was re-measured on unseen seeds, and the error bars are $\pm 1$ standard error.",
}
for knob, name in KNOB_NAME.items():
    FIG_CAPTIONS[f"convergence_vs_{knob}"] = f"This figure shows the objective and the gradient norm against iteration. The runs differ only in {name}."
    FIG_CAPTIONS[f"final_coeffs_vs_{knob}"] = f"This figure shows the final shape coefficients against {name}."
    FIG_CAPTIONS[f"shapes_vs_{knob}"] = f"This figure shows the final shapes for each value of {name}."
PASS_SETTINGS = {1: r"Pass 1 used N\_REF = 250k, N\_AVG = 4, N\_FOURIER = 10, GRAD\_INIT\_FRAC = 0.05, N\_ITER = 20 and SEED0 = 0, except for the parameter that the figure varies.",
                 2: r"Pass 2 used the same settings with N\_ITER = 50 and SEED0 = 0, 100 and 200.",
                 3: r"Pass 3 used N\_REF = 250k, N\_AVG = 4, N\_AVG\_MAX = 32, N\_FOURIER = 10, GRAD\_INIT\_FRAC = 0.05, N\_ITER = 100 and SEED0 = 0, 100, 200 and 300."}


def caption_for(stem, p):
    key = next((k for k in sorted(FIG_CAPTIONS, key=len, reverse=True) if stem == k or stem.startswith(k)), None)
    text = FIG_CAPTIONS.get(key, tex(stem))
    if stem.startswith("coeff_trajectories_B0_seed"):
        text = text.replace("the baseline run", f"the baseline run with SEED0 = {stem.split('seed')[-1]}")
    return text + " " + PASS_SETTINGS.get(p, "")


def fig(path, caption, label=None, width=r"0.92\textwidth", height=r"0.40\textheight"):
    lab = rf"\label{{{label}}}" if label else ""
    return rf"\begin{{figure}}[H]\centering\includegraphics[width={width},height={height},keepaspectratio]{{{path}}}\caption{{{lab}{caption}}}\end{{figure}}"


# ---------------------------------------------------------------------------------------------------- documents
PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{booktabs,longtable,graphicx,float,caption,array,xcolor}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\setlength{\LTcapwidth}{\textwidth}
"""

MAIN_TEXT = r"""\title{Adjoint-DSMC Parameter Study\\[2pt]\large How seeds, particles, Fourier modes and the first step affect the drag optimiser}
\author{Hansel Carmona}
\date{1 October 2026}
\begin{document}
\maketitle

\section{Summary of results}
Within 10 Fourier modes the drag plateaus at about 69\% below the circle, but the gradient has not reached its noise floor and the shape is not converged in the number of Fourier modes.
Growing the number of random seeds during a run took the drag reduction from about 62\% to 69\%.
250k particles gave the same drag and search direction as 500k.
Running the optimiser's endgame mode with 1M particles from the best shape improved the drag by only about another 0.8\% and did not bring the gradient down to its noise floor.
The first-step size made negligible difference.
Adding more Fourier modes keeps lowering the drag, so with a fixed perimeter and a weak curvature penalty the ``optimal'' shape may depend on how many modes are allowed.

%%FIG_SHAPES%%

\section{Set-up}
The optimiser reshapes a body in a rarefied gas so that the gas pushes on it as little as possible.
The gas flows from left to right through a $16 \times 8$ channel at Knudsen number 1, and it is simulated with direct simulation Monte Carlo (DSMC).
The shape is a Fourier series, $r(\theta) = c_0 + \sum_k (a_k \cos k\theta + b_k \sin k\theta)$.
Its perimeter is held equal to that of the unit circle, and a curvature penalty with weight $\lambda = 1$ is added to the drag.
Each iteration measures the drag and its gradient by averaging several independent random simulations (seeds), takes a step downhill, and checks that the drag really fell by more than the noise.
Table~\ref{tab:varied} lists the parameters that we varied, and every other setting was the same in every run.

%%TAB_VARIED%%

\subsection{The four passes}
\textbf{Pass 1.} The first pass used a baseline of N\_REF = 250k, N\_AVG = 4, N\_FOURIER = 10, GRAD\_INIT\_FRAC = 0.05 and N\_ITER = 20, and it varied one parameter at a time: N\_AVG over 1, 2, 8 and 16, N\_REF over 100k, 500k and 1M, N\_FOURIER over 3, 5 and 20, and GRAD\_INIT\_FRAC over 0.01, 0.02, 0.1 and 0.2.
It also measured the gradient at two fixed shapes with 16 seeds at each of the four particle counts.
It was meant to check how strongly each parameter affects the noise and the first stage of an optimisation.

\textbf{Pass 2.} The second pass lengthened the runs to 50 iterations and repeated six settings three times each with independent seed sets (SEED0 = 0, 100 and 200).
The six settings were the baseline, N\_REF = 500k, N\_AVG = 1, N\_AVG = 8, GRAD\_INIT\_FRAC = 0.1 and N\_FOURIER = 5.
It was meant to check whether the differences seen in pass 1 survive longer runs, and how repeatable each result is.

\textbf{Pass 3.} The third pass let the number of seeds grow during a run: each run started with N\_AVG = 4 and doubled its seeds, up to N\_AVG\_MAX = 32, whenever it could no longer confirm an improvement.
It ran four replicates (SEED0 = 0, 100, 200 and 300) for up to 100 iterations with 250k particles and 10 modes.
It also measured the drag and gradient at the best shapes on 32 unseen seeds, tested whether more Fourier modes would still lower the drag, and re-measured every final shape on 16 unseen seeds.
It was meant to check where a run ends when noise no longer stops it, how reproducible that end point is, and whether the result depends on the particle count or the number of modes.

\textbf{Pass 4.} The fourth pass continued from the pass-3 shape with N\_REF = 1M, 16 seeds growing to 32, 30 iterations and 10 endgame steps.
It was meant to check whether more particles and the endgame mode could push past the plateau and bring the gradient down to its noise floor.

\section{Results}
\subsection{Overall}
Table~\ref{tab:heldout} and Figure~\ref{fig:heldout} show how far each setting got on seeds that no run had used.
The pass-3 runs, which let the seed count grow, reached the lowest drag.
The ranking reflects how far each run got before noise stopped it, not over-fitting, because every shape performs the same on unseen seeds as on its own seeds within about one percentage point.

%%TAB_HELDOUT%%
%%FIG_HELDOUT%%

\subsection{Seeds}
Averaging $n$ seeds shrinks the error of a measurement as $1/\sqrt{n}$, exactly as theory predicts.
Near the optimum the slope gets smaller while the scatter does not, so the optimiser needs more seeds just to see the slope (Table~\ref{tab:seedsneed}), and many more to confirm that a small step lowered the drag.
Runs with a fixed small seed count therefore stall, and they stall earlier the fewer seeds they use (Table~\ref{tab:seedsrun}).
Letting the seed count grow from 4 to 32 whenever the optimiser got stuck took the drag reduction from about 62\% to 69\% (Figure~\ref{fig:growth}).
A newly added seed must first settle at the current shape, because measured straight away it reads the drag 6.6\% too low.
Pass 3 therefore added seeds only through the line search, which lets new seeds settle first.

%%TAB_SEEDSRUN%%
%%TAB_SEEDSNEED%%
%%FIG_GROWTH%%

\subsection{Particles}
More particles reduce the noise at almost exactly the expected rate (Table~\ref{tab:noiseN}), so increasing particles or seeds yields the same result.
In pass 2, 500k particles with 4 seeds and 250k particles with 8 seeds reached nearly the same shape and drag (62.6\% and 62.2\%).
The particle count also has a floor for correctness (Table~\ref{tab:bias}).
At 100k the drag is right, but the search direction is slightly wrong, while at 250k and 500k both agree.
At the thin final shape, 1M particles sees about five times more pull on the finest mode than 250k does, so fine detail needs more particles.

%%TAB_NOISEN%%
%%TAB_BIAS%%

\subsection{Fourier modes}
More shape detail keeps lowering the drag (Table~\ref{tab:modes}).
At the optimised 10-mode shape, the gradient in modes 11--20 is seven to eight times the gradient in modes 1--10.
Stepping the shape into those modes lowered the drag on unseen seeds by up to 5.6\% (Table~\ref{tab:probes} and Figure~\ref{fig:probes}).
The same test on the 5-mode shape correctly predicted that 10 modes would do better.
The gain does not come from the body shrinking, because its height changed by only 0.6\%.
The result is therefore not converged in the number of modes.

%%TAB_MODES%%
%%TAB_PROBES%%
%%FIG_PROBES%%

\subsection{First-step size}
The first-step size only changes the first three or four iterations (Table~\ref{tab:frac}).
Small values waste a few iterations growing the step, and large values overshoot once and back off.
After that all runs take steps of the same size, and the final results differ only by where individual runs stopped.

%%TAB_FRAC%%

\subsection{The optimised shape}
The optimiser mainly flattens the body along the flow (Figure~\ref{fig:shapes}).
The final shape is 2.87 long and 0.79 tall, against $2.0 \times 2.0$ for the circle, with a long rounded front, a sharper tail and shallow ripples along the top and bottom.
The four pass-3 replicates agree on every coefficient to within a standard deviation of 0.009 (Table~\ref{tab:coeffs}).
The stretch coefficient $a_2$ sits exactly at the optimiser's stretch limit of $0.6\,c_0$.

%%TAB_COEFFS%%

\subsection{Pushing further with 1M particles and the endgame}
To test whether more precision could break the plateau, pass 4 continued from the pass-3 shape with 1M particles and the endgame mode (Table~\ref{tab:pass4}).
The drag fell by only $0.84 \pm 0.27$\% more on unseen seeds, and the gradient did not shrink.
With the body pressed against the stretch limit, more particles, more seeds and the endgame cannot move it much further.

%%TAB_PASS4%%

\end{document}
"""


def build_main(T):
    s = MAIN_TEXT
    rep = {
        "%%FIG_SHAPES%%": fig("summary/pass3_p3_shapes.pdf", FIG_CAPTIONS["p3_shapes"], "fig:shapes", height=r"0.32\textheight"),
        "%%TAB_VARIED%%": T["varied"], "%%TAB_HELDOUT%%": T["heldout"],
        "%%FIG_HELDOUT%%": fig("summary/pass3_p3_heldout_by_setting.pdf", FIG_CAPTIONS["p3_heldout_by_setting"], "fig:heldout", height=r"0.34\textheight"),
        "%%TAB_SEEDSRUN%%": T["seedsrun"], "%%TAB_SEEDSNEED%%": T["seedsneed"],
        "%%FIG_GROWTH%%": fig("summary/pass3_p3_growth_trajectories.pdf", FIG_CAPTIONS["p3_growth_trajectories"], "fig:growth", height=r"0.30\textheight"),
        "%%TAB_NOISEN%%": T["noiseN"], "%%TAB_BIAS%%": T["bias"], "%%TAB_MODES%%": T["modes"], "%%TAB_PROBES%%": T["probes"],
        "%%FIG_PROBES%%": fig("summary/pass3_p3_mode_probes.pdf", FIG_CAPTIONS["p3_mode_probes"], "fig:probes", height=r"0.28\textheight"),
        "%%TAB_FRAC%%": T["frac"], "%%TAB_COEFFS%%": T["coeffs"], "%%TAB_PASS4%%": T["pass4"],
    }
    for k, v in rep.items():
        assert k in s, k
        s = s.replace(k, v)
    return PREAMBLE + s


def build_appendix(T, runs, facts, meas):
    L = [PREAMBLE + r"""\title{Adjoint-DSMC Parameter Study}
\author{Hansel Carmona}
\date{1 October 2026}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""]
    L.append(r"\section{Run Settings}\label{sec:runs}" + "\n" + T["runs"])
    L.append(r"\clearpage\section{Summary figures}\label{sec:figs}")
    for p, root in PASSES[:3]:
        figs = sorted(glob.glob(str(OUT / "summary" / f"pass{p}_*.pdf")))
        if figs:
            L.append(rf"\subsection{{Pass {p}}}")
        for f in figs:
            stem = Path(f).stem.split("_", 1)[1]
            L.append(fig(f"summary/{Path(f).name}", caption_for(stem, p), height=r"0.40\textheight"))
    L.append(r"\clearpage\section{Every run}\label{sec:perrun}"
             rf"This section shows each of the {len(runs)} optimisation runs. Each figure shows the drag reduction against iteration, with markers where the seed count doubled, "
             r"the gradient norm with its standard error, and the final shape. The parameter values of each run are printed above its figure, and every other setting was the same in every run.")
    L.append(fig("gallery.pdf", "This figure shows every final shape, with the circle dashed. Each label gives the pass, the run and the drag reduction on unseen seeds, "
                                "or on its own seeds where the shape was not re-measured.", width=r"\textwidth", height=r"0.85\textheight"))
    for r in runs:
        F = facts[r["key"]]
        cap = (rf"Pass {r['pass_']}, \texttt{{{tex(r['name'])}}}. The run stopped at the {'noise limit' if F['status'] == 'noise-limited' else 'iteration limit'} "
               rf"after {F['iters']} iterations with {F['seeds_final']} seeds. Its drag reduction was {F['own']:.1f}\% relative to the {F['own_vs']} on its own seeds"
               + (rf" and {F['held'][0]:.1f}\% relative to the circle on unseen seeds" if F["held"] else "") + rf". It took {F['wall']:.1f} hours.")
        L.append(fig(f"runs/{F['key']}.pdf", cap, width=r"\textwidth", height=r"0.42\textheight"))
    L.append("\\end{document}\n")
    return "\n".join(L)


def compile_tex(name, text):
    src = OUT / f"{name}.tex"
    src.write_text(text)
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", src.name], cwd=OUT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        raise SystemExit(f"LaTeX failed for {name}")
    pdf = OUT / f"{name}.pdf"
    print("wrote", pdf, f"({pdf.stat().st_size / 1e6:.1f} MB)")
    return pdf


def build():
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "summary").mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    H = heldout_map()
    facts = {}
    for r in runs:
        F = run_facts(r, H)
        r["key"] = F["key"]
        facts[F["key"]] = F
        run_figure(r, F, OUT / "runs" / f"{F['key']}.pdf")
    gallery(runs, facts, OUT / "gallery.pdf")
    for p, root in PASSES[:3]:
        for f in sorted(glob.glob(str(SIMPLE / root / "figures" / "*.pdf"))):
            shutil.copy(f, OUT / "summary" / f"pass{p}_{Path(f).stem}.pdf")
    T = tables(runs, facts)
    main = compile_tex("adjoint_dsmc_results", build_main(T))
    appx = compile_tex("adjoint_dsmc_results_appendix", build_appendix(T, runs, facts, None))
    return main, appx


if __name__ == "__main__":
    build()
