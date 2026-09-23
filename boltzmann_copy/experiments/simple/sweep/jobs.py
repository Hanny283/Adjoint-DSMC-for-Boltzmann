"""The three job kinds: optimise (a full run), gradstat (per-seed gradients at a fixed shape), warm (cache build)."""
import json
import time
from pathlib import Path

import numpy as np

from . import HARNESS_VERSION
from . import config as C
from .apply import apply_params
from .classify import classify_run
from .pool import SeedPool
from .recorder import HISTORY_README, Recorder

SIMPLE_DIR = Path(__file__).resolve().parents[1]


def _pool_for(cfg, seeds):
    nw = C.n_workers_of(cfg, seeds)
    return SeedPool(cfg["params"], nw, SIMPLE_DIR) if nw > 1 else None


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def _base_summary(cfg, wall, extra=None):
    s = dict(run_id=cfg["run_id"], kind=cfg["kind"], group=cfg["group"], name=cfg["name"], harness=HARNESS_VERSION,
             wall_total_s=float(wall), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
             params={k: cfg["params"][k] for k in ("N_REF", "CAP", "N_FOURIER", "N_AVG", "N_AVG_MAX", "ADAPTIVE_SEEDS", "SEED0",
                                                   "GRAD_INIT_FRAC", "N_ITER", "NORMALIZE_MODE", "CONSTRAINT", "CURV_LAMBDA",
                                                   "TOBS", "T_WARM", "REWARM_T", "LX", "LY", "COLLISIONS")})
    if extra:
        s.update(extra)
    return s


def run_optimise(cfg, run_dir, ri):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    apply_params(ri, cfg["params"])
    seeds = C.seeds_of(cfg)
    opts = cfg.get("options", {})
    (run_dir / "history_README.md").write_text(HISTORY_README)
    rec = Recorder(ri, run_dir=run_dir, n_iter=ri.N_ITER, save_every=int(opts.get("save_every", 1)))
    pool = _pool_for(cfg, seeds)
    print(f"[sweep] optimise {cfg['group']}/{cfg['name']} id={cfg['run_id']} seeds={seeds} workers={pool.n_workers if pool else 1}")
    t0 = time.time()
    try:
        res = ri.run_optimisation(seeds=seeds, pool=pool, wrap_evaluate=rec.wrap, make_plots=False,
                                  reeval_init=bool(opts.get("reeval_init", False)))
        history = rec.build_history(res)
        np.savez(run_dir / "history.npz", **history)
        sc = float(history["scale"])
        ri.plot_results(ri._effective(ri.C_init), ri._effective(res["C_opt"]), res["hist"], sc, out=str(run_dir / "convergence.png"))
        if opts.get("make_gif"):
            nf = min(len(res["shape_hist"]), len(res["hist"]["loss"]))
            force_hist = [ri.loss_to_force(res["hist"]["loss"][i], sc, res["shape_hist"][i]) for i in range(nf)]
            ri.shape_evolution_gif(res["shape_hist"][:nf], force_hist=force_hist, out=str(run_dir / "shape_evolution.gif"))
    finally:
        if pool is not None:
            pool.close()
    cl = classify_run(history)
    summary = _base_summary(cfg, time.time() - t0, dict(status=str(history["status"]), seeds=list(res["seeds"]),
                                                       n_workers=(pool.n_workers if pool else 1),
                                                       C_opt=np.asarray(res["C_opt"]).tolist(),
                                                       C_opt_eff=np.asarray(ri._effective(res["C_opt"])).tolist(),
                                                       classification=cl))
    _write_json(run_dir / "summary.json", summary)
    try:
        from .analysis import plot_run
        plot_run(run_dir)
    except Exception as e:                       # the quick-look is a convenience, never a failure
        print(f"[sweep] quicklook skipped: {e!r}")
    (run_dir / "progress.json").write_text(json.dumps(rec.progress("done"), indent=1))
    return summary


def run_gradstat(cfg, run_dir, ri):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    apply_params(ri, cfg["params"])
    gs = cfg["gradstat"]
    seeds = [int(s) for s in gs["seeds"]]
    C_eval = ri.C_init.copy() if gs.get("C_eval") is None else np.asarray(gs["C_eval"], dtype=float)
    assert len(C_eval) == len(ri.C_init), "C_eval must have 2*N_FOURIER+1 entries"
    C_sim = ri._effective(C_eval)
    pool = _pool_for(cfg, seeds)
    print(f"[sweep] gradstat {cfg['group']}/{cfg['name']} id={cfg['run_id']} N_REF={ri.N_REF} seeds={seeds} workers={pool.n_workers if pool else 1}")
    t0 = time.time()
    try:
        if pool is not None:
            pool.warm(seeds)
        else:
            for s in seeds:
                ri.warm_state(s)
        t_warm = time.time() - t0
        if gs.get("rewarm", True) and ri.REWARM_T > 0 and not np.array_equal(C_eval, ri.C_init):
            if pool is not None:
                pool.rewarm(C_sim, seeds)
            else:
                for s in seeds:
                    ri.rewarm(s, C_sim)
        ev = ri.make_evaluate_adjoint(seeds, pool=pool)
        t1 = time.time()
        L, g = ev(C_eval, want_grad=True)
        t_grad = time.time() - t1
    finally:
        if pool is not None:
            pool.close()
    st = ev.state
    sc = float(st["scale"] or 1.0)
    gsc = float(st["gscale"] or 1.0) if ri.NORMALIZE_MODE == "both" else 1.0
    G = np.asarray(st["G"], float) * sc * gsc                      # raw units regardless of NORMALIZE_MODE
    losses = np.asarray(st["losses"], float) * sc
    from adjoint.shape_gradient import curvature_penalty_gradient
    g_pen = ri._constrained_grad(C_eval, ri.CURV_LAMBDA * curvature_penalty_gradient(C_sim)) if ri.CURV_LAMBDA else np.zeros_like(C_eval)
    out = dict(G=G, losses=losses, seeds=np.asarray(seeds), C_eval=C_eval, C_eval_eff=np.asarray(C_sim), g_pen=g_pen,
               g_mean=G.mean(axis=0), N_REF=np.array(ri.N_REF), N_FOURIER=np.array(ri.N_FOURIER),
               t_warm=np.array(t_warm), t_grad=np.array(t_grad), n_workers=np.array(pool.n_workers if pool else 1))
    np.savez(run_dir / "gradstat.npz", **out)
    (run_dir / "gradstat_README.md").write_text(
        "# gradstat.npz -- per-seed gradients at one fixed shape (raw units)\n"
        "- G (K, P): per-seed gradient dJ/dC (constraint-rescaled chain rule applied), K seeds\n"
        "- losses (K,): per-seed objective J (drag force + curvature penalty)\n"
        "- seeds (K,), C_eval (P,) raw shape, C_eval_eff (P,) simulated (rescaled) shape\n"
        "- g_pen (P,): deterministic curvature-penalty part of the gradient (G - g_pen = flow part)\n"
        "- g_mean (P,), N_REF, N_FOURIER, t_warm (s, cache build/load), t_grad (s, all seeds), n_workers\n")
    summary = _base_summary(cfg, time.time() - t0, dict(
        shape=("C_init" if gs.get("C_eval") is None else "C_def"), seeds=seeds, n_workers=(pool.n_workers if pool else 1),
        t_warm_s=t_warm, t_grad_s=t_grad, g_mean=G.mean(axis=0).tolist(),
        g_sd=(G.std(axis=0, ddof=1).tolist() if len(seeds) > 1 else None),
        loss_mean=float(losses.mean()), loss_sd=(float(losses.std(ddof=1)) if len(seeds) > 1 else None)))
    _write_json(run_dir / "summary.json", summary)
    return summary


def run_warm(cfg, run_dir, ri):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    apply_params(ri, cfg["params"])
    seeds = [int(s) for s in cfg["warm"]["seeds"]]
    pool = _pool_for(cfg, seeds)
    print(f"[sweep] warm {cfg['group']}/{cfg['name']} N_REF={ri.N_REF} seeds={seeds} workers={pool.n_workers if pool else 1}")
    t0 = time.time()
    try:
        if pool is not None:
            pool.warm(seeds)
        else:
            for s in seeds:
                ri.warm_state(s)
    finally:
        if pool is not None:
            pool.close()
    summary = _base_summary(cfg, time.time() - t0, dict(seeds=seeds, n_workers=(pool.n_workers if pool else 1)))
    _write_json(run_dir / "summary.json", summary)
    return summary


def run_job(cfg, run_dir, ri):
    return {"optimise": run_optimise, "gradstat": run_gradstat, "warm": run_warm}[cfg["kind"]](cfg, run_dir, ri)
