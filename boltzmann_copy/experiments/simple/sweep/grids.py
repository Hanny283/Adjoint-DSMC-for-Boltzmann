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
