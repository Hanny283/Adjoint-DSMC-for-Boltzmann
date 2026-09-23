"""Set runInflow's module globals from a params dict, in the order the derived quantities require."""
import contextlib

import numpy as np

from .config import TUNABLES, defaults

_DERIVED = ["C_init", "N_STEPS", "M_P", "N0", "_L0_VALUE", "_A0_VALUE"]


def reset_caches(ri):
    ri._WARM_CACHE.clear()
    ri._WARM_STEP.clear()
    ri._GAS_FRAC_CACHE.clear()
    ri.LAST_DIAG = None


def apply_params(ri, params):
    """Apply `params` (runInflow global names -> values) to the imported module `ri`. Returns the saved
    previous values (for restore). Order matters: N_STEPS follows TOBS/DT, M_P/N0 follow N_REF (via
    set_particle_count), C_init follows N_FOURIER, the constraint targets follow C_init."""
    saved = {k: getattr(ri, k) for k in TUNABLES + _DERIVED}
    p = defaults(ri)                 # partial dicts are fine: missing tunables keep runInflow's defaults
    p.update(params)
    if p.get("CAP") is None:
        p["CAP"] = int(round(1.3 * p["N_REF"]))
    # 1. box and time grid
    for k in ("LX", "LY", "DT", "T_START", "TOBS", "T_WARM", "REWARM_T"):
        setattr(ri, k, float(p[k]))
    ri.N_COLL_CELLS = tuple(int(c) for c in p["N_COLL_CELLS"])
    ri.N_STEPS = int(round(ri.TOBS / ri.DT))
    assert ri._k_start() < ri.N_STEPS, "empty impulse window: T_START >= TOBS"
    # 2. physics and boundary conditions
    for k in ("U0", "T0", "RHO0", "KN"):
        setattr(ri, k, float(p[k]))
    for k in ("COLLISIONS", "RIGHT_BACKFLOW", "PREFILL", "WARM_START", "ADD_EDGE_CORRECTION"):
        setattr(ri, k, bool(p[k]))
    ri.VEL_DIM = int(p["VEL_DIM"])
    ri.SIDE_BC = p["SIDE_BC"]
    ri._FLUX_TABLE = None
    ri._BACK_TABLE = None
    # 3. resolution (recomputes M_P and N0)
    ri.set_particle_count(int(p["N_REF"]), int(p["CAP"]))
    # 4. shape parameterisation: the unit circle with N_FOURIER modes
    ri.N_FOURIER = int(p["N_FOURIER"])
    ri.C_init = np.zeros(2 * ri.N_FOURIER + 1)
    ri.C_init[0] = 1.0
    # 5. seeds and optimiser knobs
    for k in ("N_AVG", "N_AVG_MAX", "SEED0", "N_ITER", "ENDGAME_ITERS"):
        setattr(ri, k, int(p[k]))
    for k in ("ADAPTIVE_SEEDS", "STOP_ON_SIGNIFICANCE", "ENDGAME"):
        setattr(ri, k, bool(p[k]))
    for k in ("GRAD_SIGNIFICANCE", "GRAD_INIT_FRAC", "GRAD_TOL_REL", "MIN_DISPLACEMENT", "ARMIJO_C", "NOISE_SNR",
              "STEP_INC", "STEP_DEC", "A_MAX_FRAC", "ENDGAME_SIGNIF", "CURV_LAMBDA", "WEIGHT"):
        setattr(ri, k, float(p[k]))
    ri.NORMALIZE_MODE = p["NORMALIZE_MODE"]
    ri.OBJECTIVE = p["OBJECTIVE"]
    ri.CONSTRAINT = p["CONSTRAINT"]
    # 6. constraint targets and caches
    ri.set_reference_shape(ri.C_init)
    reset_caches(ri)
    return saved


def restore(ri, saved):
    for k, v in saved.items():
        setattr(ri, k, v)
    reset_caches(ri)


@contextlib.contextmanager
def apply_config(ri, params):
    saved = apply_params(ri, params)
    try:
        yield ri
    finally:
        restore(ri, saved)
