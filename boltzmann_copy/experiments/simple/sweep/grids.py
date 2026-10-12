"""Sweep definitions: the baseline, the minimal first pass (one run per knob value), phase-2 replicates, smoke configs."""
import numpy as np

BASELINE = dict(N_REF=250_000, N_FOURIER=10, N_AVG=4, N_AVG_MAX=4, ADAPTIVE_SEEDS=False, SEED0=0,
                GRAD_INIT_FRAC=0.05, N_ITER=20, STOP_ON_SIGNIFICANCE=False, ENDGAME=False, NORMALIZE_MODE="both")
NREF_GRID = [100_000, 250_000, 500_000, 1_000_000]
SEEDS_GRID = [1, 2, 4, 8, 16]
NF_GRID = [3, 5, 10, 20]
FRAC_GRID = [0.01, 0.02, 0.05, 0.1, 0.2]
GRADSTAT_SEEDS = list(range(16))


def _k(n):
    return f"{n // 1000}k" if n < 1_000_000 else f"{n // 1_000_000}M"


def optimise_cfg(group, name, **over):
    p = dict(BASELINE)
    p.update(over)
    if "N_AVG" in over and "N_AVG_MAX" not in over:
        p["N_AVG_MAX"] = p["N_AVG"]
    return dict(kind="optimise", group=group, name=name, params=p, options=dict(reeval_init=False, make_gif=False))


def c_def(n_fourier=10, a2=0.15):
    """A deformed shape for the gradient statistics: a_2 mode (streamlined along x), inside project_C's cap."""
    C = np.zeros(2 * n_fourier + 1)
    C[0] = 1.0
    C[2] = a2
    return C.tolist()


def gradstat_cfg(n_ref, shape="C_init", seeds=GRADSTAT_SEEDS, n_fourier=10, group="gradstat"):
    p = dict(BASELINE)
    p.update(N_REF=n_ref, N_FOURIER=n_fourier, N_AVG=len(seeds), N_AVG_MAX=len(seeds), ADAPTIVE_SEEDS=False, NORMALIZE_MODE=None)
    gs = dict(seeds=list(seeds), C_eval=(None if shape == "C_init" else c_def(n_fourier)), rewarm=True)
    return dict(kind="gradstat", group=group, name=f"nref{_k(n_ref)}_{shape}", params=p, gradstat=gs, options={})


def baseline_cfg(seed0=0, group="baseline"):
    return optimise_cfg(group, f"B0_seed{seed0}", SEED0=seed0)


def first_pass():
    """Minimal first pass: gradient statistics at 4 particle counts x 2 shapes, the baseline, and one run per
    knob value. Ordered cheap -> expensive; the gradstat jobs also build the warm caches the runs reuse."""
    q = []
    for n in (100_000, 250_000):
        q += [gradstat_cfg(n, "C_init"), gradstat_cfg(n, "C_def")]
    q.append(baseline_cfg(0))
    q += [optimise_cfg("sweep_frac", f"frac{f:g}", GRAD_INIT_FRAC=f) for f in FRAC_GRID if f != BASELINE["GRAD_INIT_FRAC"]]
    q += [optimise_cfg("sweep_seeds", f"nseeds{s}", N_AVG=s, N_AVG_MAX=s) for s in SEEDS_GRID if s != BASELINE["N_AVG"]]
    q += [optimise_cfg("sweep_nfourier", f"nf{k}", N_FOURIER=k) for k in NF_GRID if k != BASELINE["N_FOURIER"]]
    for n in (500_000, 1_000_000):
        q += [gradstat_cfg(n, "C_init"), gradstat_cfg(n, "C_def")]
    q += [optimise_cfg("sweep_nref", f"nref{_k(n)}", N_REF=n) for n in NREF_GRID if n != BASELINE["N_REF"]]
    return q


def phase2_replicates(seed0s=(100, 200)):
    """Replicates (different seed sets) of the baseline and of every knob value: across-run mean/variance of
    the optimised coefficients. Same run ids as the first pass otherwise, so finished runs are skipped."""
    q = []
    for s0 in seed0s:
        q.append(baseline_cfg(s0))
        q += [optimise_cfg("sweep_frac", f"frac{f:g}_seed{s0}", GRAD_INIT_FRAC=f, SEED0=s0) for f in FRAC_GRID if f != BASELINE["GRAD_INIT_FRAC"]]
        q += [optimise_cfg("sweep_seeds", f"nseeds{s}_seed{s0}", N_AVG=s, N_AVG_MAX=s, SEED0=s0) for s in SEEDS_GRID if s != BASELINE["N_AVG"]]
        q += [optimise_cfg("sweep_nfourier", f"nf{k}_seed{s0}", N_FOURIER=k, SEED0=s0) for k in NF_GRID if k != BASELINE["N_FOURIER"]]
        q += [optimise_cfg("sweep_nref", f"nref{_k(n)}_seed{s0}", N_REF=n, SEED0=s0) for n in NREF_GRID if n != BASELINE["N_REF"]]
    return q


PASS2_N_ITER = 50
PASS2_SEED0S = (0, 100, 200)


def warm_cfg(name, seeds, group="warm", **over):
    p = dict(BASELINE)
    p.update(over)
    return dict(kind="warm", group=group, name=name, params=p, warm=dict(seeds=list(seeds)), options={})


def second_pass(n_iter=PASS2_N_ITER, seed0s=PASS2_SEED0S):
    """Second pass (after the first pass showed every run was iteration-limited at N_ITER=20): longer runs of the
    baseline and of the promising knob values, each replicated over independent seed sets (SEED0) so the
    across-run mean/variance of the optimised coefficients can be measured. Use its own results root, since
    the analysis matches runs by knob values and not by N_ITER."""
    new = [s0 for s0 in seed0s if s0 != 0]
    q = [warm_cfg("w250k_nf10", [s0 + i for s0 in new for i in range(8)]),
         warm_cfg("w500k_nf10", [s0 + i for s0 in new for i in range(4)], N_REF=500_000),
         warm_cfg("w250k_nf5", [s0 + i for s0 in new for i in range(4)], N_FOURIER=5)]
    for s0 in seed0s:
        q.append(optimise_cfg("sweep_nref", f"nref500k_seed{s0}", N_REF=500_000, SEED0=s0, N_ITER=n_iter))
    for s0 in seed0s:
        q.append(optimise_cfg("sweep_seeds", f"nseeds1_seed{s0}", N_AVG=1, SEED0=s0, N_ITER=n_iter))
    for s0 in seed0s:
        q.append(optimise_cfg("sweep_seeds", f"nseeds8_seed{s0}", N_AVG=8, SEED0=s0, N_ITER=n_iter))
    for s0 in seed0s:
        q.append(optimise_cfg("baseline", f"B0_seed{s0}", SEED0=s0, N_ITER=n_iter))
    for s0 in seed0s:
        q.append(optimise_cfg("sweep_frac", f"frac0.1_seed{s0}", GRAD_INIT_FRAC=0.1, SEED0=s0, N_ITER=n_iter))
    for s0 in seed0s:
        q.append(optimise_cfg("sweep_nfourier", f"nf5_seed{s0}", N_FOURIER=5, SEED0=s0, N_ITER=n_iter))
    return q


HELDOUT_SEEDS = list(range(1000, 1032))   # never used by an optimisation run (those use SEED0 + 0..31, SEED0 <= 300)


def pad_modes(C, n_fourier):
    """Coefficients [c0, a1..aN, b1..bN] zero-padded to n_fourier modes (N <= n_fourier)."""
    src = np.asarray(C, dtype=float)
    nf = (len(src) - 1) // 2
    assert nf <= n_fourier, "pad_modes only pads up"
    out = np.zeros(2 * n_fourier + 1)
    out[0] = src[0]
    out[1:1 + nf] = src[1:1 + nf]
    out[1 + n_fourier:1 + n_fourier + nf] = src[1 + nf:]
    return out.tolist()


def gradstat_at(C_eval, name, n_ref=250_000, seeds=HELDOUT_SEEDS, n_fourier=10, rewarm_t=None, group="diag", n_workers=None):
    """Per-seed gradients and objective at a fixed shape (raw units), measured on held-out seeds after the
    seeds' C_init warm states are advanced by REWARM_T at that shape. Used for the noise/bias diagnostics
    near the optimum and for the out-of-sample evaluation of optimised shapes."""
    p = dict(BASELINE)
    p.update(N_REF=n_ref, N_FOURIER=n_fourier, N_AVG=len(seeds), N_AVG_MAX=len(seeds), ADAPTIVE_SEEDS=False, NORMALIZE_MODE=None)
    if rewarm_t is not None:
        p["REWARM_T"] = rewarm_t
    return dict(kind="gradstat", group=group, name=name, params=p,
                gradstat=dict(seeds=list(seeds), C_eval=pad_modes(C_eval, n_fourier), rewarm=True),
                options=(dict(n_workers=n_workers) if n_workers else {}))


def pass3_pilot(C2):
    """Pilot for pass 3: warm the held-out seeds, then measure how the objective at the pass-2 optimum C2
    depends on the adaptation time REWARM_T of a C_init warm state (0 = a freshly added seed)."""
    q = [warm_cfg("w250k_nf10_heldout", HELDOUT_SEEDS)]
    q += [gradstat_at(C2, f"adapt_T{t:g}_C2", seeds=HELDOUT_SEEDS[:16], rewarm_t=float(t), group="adapt") for t in (0, 10, 20, 40, 80)]
    q += [warm_cfg("w100k_nf10_heldout", HELDOUT_SEEDS, N_REF=100_000),
          warm_cfg("w500k_nf10_heldout", HELDOUT_SEEDS[:16], N_REF=500_000),
          warm_cfg("w250k_nf20_heldout", HELDOUT_SEEDS, N_FOURIER=20)]
    return q


def pass3_diagnostics(C2, C5, rewarm_t):
    """Fixed-shape measurements near the optimum on held-out seeds:
    noise near the optimum (C2, 250k, 32 seeds); particle-count bias of objective and gradient per particle
    (C2 at 100k / 500k); mode sufficiency: the gradient in modes 11-20 at the 10-mode optimum (C2 padded to 20
    modes) and in modes 6-10 at the 5-mode optimum (C5 padded to 10 modes). Non-zero gradient in the omitted
    modes means more modes would lower the objective further."""
    return [gradstat_at(C2, "C2_250k_nf10", rewarm_t=rewarm_t),
            gradstat_at(C2, "C2_250k_nf20", n_fourier=20, rewarm_t=rewarm_t),
            gradstat_at(C5, "C5_250k_nf10", rewarm_t=rewarm_t),
            gradstat_at(C2, "C2_100k_nf10", n_ref=100_000, rewarm_t=rewarm_t),
            gradstat_at(C2, "C2_500k_nf10", n_ref=500_000, seeds=HELDOUT_SEEDS[:16], rewarm_t=rewarm_t)]


PASS3_SEED0S = (0, 100, 200, 300)


def pass3_adaptive(seed0s=PASS3_SEED0S, n_avg=4, n_avg_max=32, n_iter=100, n_workers=4):
    """Pass 3: the baseline (250k, 10 modes, frac 0.05) with seed growth instead of stopping. Starts at n_avg
    seeds; when the line search is noise-limited it doubles the seeds (up to n_avg_max) and retries at the same
    point, and the run stops only when it is noise-limited at n_avg_max seeds (converged at that resolution)
    or after n_iter iterations. The retry is a full gradient call, which first advances every seed's warm
    state by REWARM_T at the current shape, so new seeds enter adapted. ADAPTIVE_SEEDS stays off: it grows
    seeds INSIDE a gradient call and evaluates them from the un-adapted C_init state (pilot: -6.6% drag,
    gradient cosine 0.93 at the pass-2 optimum; REWARM_T = 10 removes it). STOP_ON_SIGNIFICANCE is off
    because without ADAPTIVE_SEEDS it would stop at any seed count instead of growing. Replicates use
    disjoint seed sets (SEED0 + 0..n_avg_max-1). The pool size is fixed at the start (<= n_avg): 4 workers
    each lets all four replicates run side by side on 16 cores; grown seed lists are evaluated in rounds."""
    return [dict(kind="optimise", group="adaptive", name=f"grow{n_avg}to{n_avg_max}_seed{s0}",
                 params=dict(BASELINE, N_AVG=n_avg, N_AVG_MAX=n_avg_max, ADAPTIVE_SEEDS=False, STOP_ON_SIGNIFICANCE=False,
                             N_ITER=n_iter, SEED0=s0),
                 options=dict(reeval_init=False, make_gif=False, n_workers=n_workers))
            for s0 in seed0s]


def heldout_evaluations(shapes, rewarm_t, seeds=HELDOUT_SEEDS[:16], n_workers=8):
    """Out-of-sample evaluation: every optimised shape (dict name -> C_eff, any number of modes <= 10) and the
    circle, on the same held-out seeds at 250k. Same seeds for every shape = paired comparisons across shapes;
    seeds never used in any optimisation = no in-sample optimism (overfitting to the optimiser's own seeds)."""
    q = [gradstat_at(C_init_circle(), "circle", seeds=seeds, rewarm_t=rewarm_t, group="heldout", n_workers=n_workers)]
    q += [gradstat_at(C, name, seeds=seeds, rewarm_t=rewarm_t, group="heldout", n_workers=n_workers) for name, C in shapes.items()]
    return q


def pass3_final(finals, C3, probes, rewarm_t):
    """After the pass-3 growth runs: (i) line probes from the pass-2 consensus shapes along the negative gradient
    in the modes the shape does not use (does adding modes really lower the held-out objective, and by how much?);
    (ii) the gradient in modes 11-20 at the pass-3 consensus C3 (does it persist at the converged 10-mode shape?);
    (iii) held-out evaluation of every pass-3 final shape and of C3.
    finals: dict name -> C_eff; probes: dict name -> dict(C=..., n_fourier=...)."""
    q = [gradstat_at(p["C"], name, n_fourier=p["n_fourier"], seeds=HELDOUT_SEEDS[:16], rewarm_t=rewarm_t, group="probe")
         for name, p in probes.items()]
    q.append(gradstat_at(C3, "C3_250k_nf20", n_fourier=20, rewarm_t=rewarm_t))
    q += [gradstat_at(C, name, seeds=HELDOUT_SEEDS[:16], rewarm_t=rewarm_t, group="heldout") for name, C in finals.items()]
    q.append(gradstat_at(C3, "p3_consensus_C3", seeds=HELDOUT_SEEDS[:16], rewarm_t=rewarm_t, group="heldout"))
    return q


def pass4_1M(C_start, seed0=400, n_avg=16, n_avg_max=32, n_iter=30, endgame_iters=10, frac=0.01, n_workers=16):
    """Pass 4: continue from the pass-3 consensus shape at N_REF = 1M with the endgame on.
    (i) a fixed-shape check at 1M on the held-out seeds (is 250k biased at the final shape?);
    (ii) a continuation run: C_START = the pass-3 shape (rescaled to the circle's perimeter by the caller),
    seeds 16 -> 32 by line-search growth (16 = one seed per core; the pool size is fixed at the start), up to n_iter line-search iterations, then endgame_iters endgame steps
    (curvature-scaled gradient steps that are only vetoed when the paired loss is significantly worse).
    frac sets the first step and the endgame step cap to frac * ||C_start||: late pass-3 steps were 0.003-0.01.
    Fresh seeds (SEED0 = 400) keep the run independent of every earlier optimisation and of the held-out seeds."""
    check = gradstat_at(C_start, "C3_1M_nf10", n_ref=1_000_000, seeds=HELDOUT_SEEDS[:16], rewarm_t=20.0, group="diag")
    run = dict(kind="optimise", group="continuation", name=f"C3_1M_endgame_seed{seed0}",
               params=dict(BASELINE, N_REF=1_000_000, C_START=list(C_start), N_AVG=n_avg, N_AVG_MAX=n_avg_max,
                           ADAPTIVE_SEEDS=False, STOP_ON_SIGNIFICANCE=False, N_ITER=n_iter, ENDGAME=True,
                           ENDGAME_ITERS=endgame_iters, GRAD_INIT_FRAC=frac, SEED0=seed0),
               options=dict(reeval_init=False, make_gif=False, n_workers=n_workers))
    return [check, run]


PASS5_CAPS = ((1.0, 500), (2.0, 500), (1e9, 500), (1e9, 600))     # (A_MAX_FRAC, SEED0); 1e9 = no cap


def pass5_cap(C_start, caps=PASS5_CAPS, n_avg=4, n_avg_max=32, n_iter=60, frac=0.01, n_workers=4):
    """Pass 5: does the stretch limit A_MAX_FRAC (every mode <= A_MAX_FRAC * c0, default 0.6) cause the stall?
    Continue from the pass-3 consensus shape (rescaled to the circle's perimeter by the caller), which sits exactly on
    the cap (a2 = 0.6 c0), with the cap raised to 1, 2 and effectively infinity (1e9). The first three runs share one
    seed set, so their differences come from the cap alone; the fourth repeats the uncapped run on another seed set.
    Self-intersection cannot occur without the cap: the boundary is a polar curve r(theta) and is_valid keeps r > 0.05.
    A warm-up job builds the seeds' states at the start shape once, instead of three runs building the same ones."""
    seeds = sorted({s0 + i for _, s0 in caps for i in range(n_avg_max)})
    q = [warm_cfg("w250k_C3_pass5", seeds, C_START=list(C_start))]
    for cap, s0 in caps:
        tag = "inf" if cap >= 1e8 else f"{cap:g}"
        q.append(dict(kind="optimise", group="stretch_cap", name=f"cap{tag}_seed{s0}",
                      params=dict(BASELINE, C_START=list(C_start), A_MAX_FRAC=cap, N_AVG=n_avg, N_AVG_MAX=n_avg_max, ADAPTIVE_SEEDS=False,
                                  STOP_ON_SIGNIFICANCE=False, N_ITER=n_iter, GRAD_INIT_FRAC=frac, SEED0=s0),
                      options=dict(reeval_init=False, make_gif=False, n_workers=n_workers)))
    return q


def C_init_circle(n_fourier=10):
    C = np.zeros(2 * n_fourier + 1)
    C[0] = 1.0
    return C.tolist()


SMOKE = dict(LX=6.0, LY=4.0, N_COLL_CELLS=[48, 32], N_REF=20_000, CAP=30_000, T_WARM=5.0, REWARM_T=1.0, TOBS=2.0,
             N_ITER=3, N_AVG=2, N_AVG_MAX=2, ADAPTIVE_SEEDS=False, N_FOURIER=3, STOP_ON_SIGNIFICANCE=False, ENDGAME=False)


def smoke_optimise(name="smoke", n_workers=None, **over):
    p = dict(SMOKE)
    p.update(over)
    return dict(kind="optimise", group="smoke", name=name, params=p, options=dict(reeval_init=False, make_gif=False, n_workers=n_workers))


def smoke_gradstat(n_ref=20_000, seeds=(0, 1, 2), n_workers=None):
    p = dict(SMOKE)
    p.update(N_REF=n_ref, CAP=int(round(1.5 * n_ref)), N_AVG=len(seeds), N_AVG_MAX=len(seeds), NORMALIZE_MODE=None)
    return dict(kind="gradstat", group="smoke", name=f"gradstat{n_ref}", params=p,
                gradstat=dict(seeds=list(seeds), C_eval=None, rewarm=True), options=dict(n_workers=n_workers))
