"""Config schema: {"kind", "group", "name", "params": {runInflow globals}, "gradstat": {...}, "options": {...}}.

`params` keys ARE runInflow global names (see TUNABLES); missing keys are filled from runInflow's import-time
defaults, so two configs that mean the same thing get the same run id whether or not they spell everything out.
"""
import hashlib
import json
import os
from pathlib import Path

from . import HARNESS_VERSION

TUNABLES = [
    # box / time grid
    "LX", "LY", "DT", "T_START", "TOBS", "T_WARM", "REWARM_T", "N_COLL_CELLS",
    # physics / boundary conditions
    "U0", "T0", "RHO0", "KN", "COLLISIONS", "VEL_DIM", "SIDE_BC", "RIGHT_BACKFLOW", "PREFILL", "WARM_START",
    # resolution
    "N_REF", "CAP",
    # shape parameterisation
    "N_FOURIER",
    # seeds
    "N_AVG", "N_AVG_MAX", "ADAPTIVE_SEEDS", "SEED0", "GRAD_SIGNIFICANCE",
    # optimiser
    "N_ITER", "GRAD_INIT_FRAC", "NORMALIZE_MODE", "STOP_ON_SIGNIFICANCE", "GRAD_TOL_REL", "MIN_DISPLACEMENT",
    "ARMIJO_C", "NOISE_SNR", "STEP_INC", "STEP_DEC", "A_MAX_FRAC", "ENDGAME", "ENDGAME_ITERS", "ENDGAME_SIGNIF",
    # objective
    "OBJECTIVE", "CONSTRAINT", "CURV_LAMBDA", "WEIGHT", "ADD_EDGE_CORRECTION",
]
KINDS = ("optimise", "gradstat", "warm")
_DEFAULTS = None


def _jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def defaults(ri):
    """Snapshot of runInflow's import-time values of every tunable (taken once per process)."""
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = {k: _jsonable(getattr(ri, k)) for k in TUNABLES}
    return dict(_DEFAULTS)


def resolve(cfg, ri):
    """Return a fully specified copy of cfg (all tunables present, derived values filled, validated)."""
    cfg = json.loads(json.dumps(cfg, default=_jsonable))          # deep copy, json-clean
    kind = cfg.get("kind", "optimise")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    params = defaults(ri)
    unknown = set(cfg.get("params", {})) - set(TUNABLES)
    if unknown:
        raise ValueError(f"unknown params (not runInflow tunables): {sorted(unknown)}")
    params.update(cfg.get("params", {}))
    if params.get("CAP") is None:
        params["CAP"] = int(round(1.3 * params["N_REF"]))
    params["N_COLL_CELLS"] = [int(c) for c in params["N_COLL_CELLS"]]
    if params["N_AVG"] > params["N_AVG_MAX"]:
        raise ValueError("N_AVG must be <= N_AVG_MAX")
    if params["CAP"] < 1.2 * params["N_REF"]:
        raise ValueError("CAP should be >= 1.2 * N_REF (pool capacity)")
    if abs(params["TOBS"] / params["DT"] - round(params["TOBS"] / params["DT"])) > 1e-9:
        raise ValueError("TOBS must be a multiple of DT")
    if params["NORMALIZE_MODE"] not in ("both", "objective", "gradient", None):
        raise ValueError("bad NORMALIZE_MODE")
    cfg["kind"] = kind
    cfg["params"] = params
    cfg.setdefault("group", "default")
    cfg.setdefault("name", kind)
    cfg.setdefault("options", {})
    if kind == "gradstat":
        gs = cfg.setdefault("gradstat", {})
        gs.setdefault("seeds", list(range(params["SEED0"], params["SEED0"] + params["N_AVG"])))
        gs.setdefault("C_eval", None)          # None -> C_init; else a list of 2N+1 coefficients
        gs.setdefault("rewarm", True)          # mirror the optimiser's moving warm start at a deformed shape
    if kind == "warm":
        ws = cfg.setdefault("warm", {})
        ws.setdefault("seeds", list(range(params["SEED0"], params["SEED0"] + params["N_AVG"])))
    cfg["run_id"] = run_id(cfg, ri)
    return cfg


def run_id(cfg, ri):
    """Hash of everything that changes the numbers: kind, resolved params, gradstat/warm block, code versions."""
    params = defaults(ri)
    params.update(cfg.get("params", {}))
    if params.get("CAP") is None:
        params["CAP"] = int(round(1.3 * params["N_REF"]))
    key = {"kind": cfg.get("kind", "optimise"), "params": params,
           "gradstat": cfg.get("gradstat"), "warm": cfg.get("warm"),
           "WARM_VERSION": getattr(ri, "WARM_VERSION", None), "HARNESS_VERSION": HARNESS_VERSION}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=_jsonable).encode()).hexdigest()[:10]


def run_dir(root, cfg):
    return Path(root) / cfg.get("group", "default") / f"{cfg.get('name', cfg.get('kind', 'run'))}__{cfg['run_id'][:8]}"


def save(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1, sort_keys=True, default=_jsonable))
    return path


def load(path):
    return json.loads(Path(path).read_text())


def seeds_of(cfg):
    p = cfg["params"]
    return list(range(int(p["SEED0"]), int(p["SEED0"]) + int(p["N_AVG"])))


def n_workers_of(cfg, kind_seeds=None):
    """Worker processes for the seed pool: options.n_workers, else min(n_seeds, physical cores)."""
    n = cfg.get("options", {}).get("n_workers")
    n_seeds = len(kind_seeds) if kind_seeds is not None else int(cfg["params"]["N_AVG_MAX"])
    if n is None:
        n = min(n_seeds, max(1, (os.cpu_count() or 2) // 2))
    return max(1, min(int(n), n_seeds))


def mem_estimate_gb(cfg, n_workers=None):
    """Rough peak RSS: ~2.3 GB per concurrent seed at N_REF = 1e6 (docstring of runInflow, scaled) + overhead."""
    n = n_workers if n_workers is not None else n_workers_of(cfg)
    per_seed = 2.3 * cfg["params"]["N_REF"] / 1e6 + 0.2
    return 0.3 + n * per_seed
