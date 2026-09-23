"""Recorder: wraps the optimiser's evaluate() and records every gradient call and line-search trial."""
import json
import time
from pathlib import Path

import numpy as np

HISTORY_README = """# history.npz -- one optimisation run (arrays; n = iterates visited incl. the start, P = 2*N_FOURIER+1,
S = max number of seeds, n_g = gradient calls, n_t = loss-only line-search trials)

Per iterate (index i = 0 is C_init; i >= 1 are accepted line-search steps in order):
- loss (n,)           normalised objective (J / J(C_init) in NORMALIZE_MODE "both"/"objective"; raw otherwise)
- objective_raw (n,)  loss * scale  (J in raw force units, incl. the curvature penalty if CURV_LAMBDA)
- force (n,)          mean drag force F_x (penalty removed) via runInflow.loss_to_force
- gnorm (n,)          ||g|| in the run's gradient units (||g(C_init)|| = 1 in mode "both")
- grad_se (n,)        pooled standard error of the mean gradient (inf/nan for 1 seed)
- grad_se_unpooled    the same from this iterate's per-seed gradients only
- grad_signif (n,)    ||g|| / grad_se;  gnorm_corr (n,) = sqrt(max(||g||^2 - grad_se^2, 0))
- loss_se (n,)        standard error of the per-seed losses
- n_seeds (n,)        seeds averaged at this iterate;  seeds_used (n, S) the seed labels (-1 = unused)
- C_raw (n, P)        optimiser parameters;  C_eff (n, P) the SIMULATED shape (constraint-rescaled)
- grad (n, P)         mean gradient;  G (n, S, P) per-seed gradients (NaN-padded);  losses_seed (n, S)
- perimeter_eff, area_eff, curv_penalty_eff (n,)  geometry of C_eff
- t_iter (n,)         seconds since the run started at this iterate's gradient call;  wall_iter (n,) = diff
- n_trials (n,)       loss-only trials evaluated before this iterate was accepted (backtracks + 1)
- n_refines (n,)      extra gradient re-evaluations at the same point (noise-limited seed growth)
- step, disp (n,)     optimiser trial step and accepted displacement ||C_i - C_{i-1}|| (nan at i = 0)
- accepted (n_attempts,) bool per attempted iteration;  h_secant (n_acc,) secant curvature estimates
All gradient calls / trials (for diagnostics):
- gcall_C (n_g, P), gcall_loss, gcall_gnorm, gcall_n_seeds, gcall_t, gcall_is_refine
- trial_C (n_t, P), trial_loss, trial_losses_seed (n_t, S), trial_disp, trial_iter, trial_accepted, trial_t
Scalars: scale, gscale, slope_scale, gnorm0, L0, A0, L0_same, wall_total, status (string), n_iter_requested
"""


def _pad(rows, S, fill=np.nan):
    out = np.full((len(rows), S) + (np.shape(rows[0])[1:] if len(rows) and np.ndim(rows[0]) > 1 else ()), fill, dtype=float)
    for i, r in enumerate(rows):
        r = np.asarray(r, dtype=float)
        if r.size:
            out[i, :len(r)] = r
    return out


class Recorder:
    """evaluate-like wrapper. Use Recorder.wrap as `wrap_evaluate` of runInflow.run_optimisation."""

    def __init__(self, ri, run_dir=None, n_iter=None, save_every=1):
        self.ri = ri
        self.run_dir = Path(run_dir) if run_dir else None
        self.n_iter = n_iter
        self.save_every = save_every
        self.t0 = time.time()
        self.gcalls, self.trials = [], []
        self._trials_since = 0
        self.inner = None

    def wrap(self, evaluate):
        self.inner = evaluate
        self.t0 = time.time()
        return self

    # attributes the optimiser reads
    @property
    def state(self):
        return self.inner.state

    @property
    def seeds(self):
        return self.inner.seeds

    @property
    def grow(self):
        return self.inner.grow

    def n_iterates(self):
        return len({np.asarray(g["C_raw"]).tobytes() for g in self.gcalls})

    def __call__(self, C, want_grad=True):
        C = np.asarray(C, dtype=float)
        loss, g = self.inner(C, want_grad)
        st = self.inner.state
        t = time.time() - self.t0
        if want_grad:
            prev = self.gcalls[-1]["C_raw"] if self.gcalls else None
            rec = dict(C_raw=C.copy(), C_eff=np.asarray(self.ri._effective(C), dtype=float), loss=float(loss),
                       grad=np.asarray(g, dtype=float).copy(), t=t, n_trials=self._trials_since,
                       is_refine=bool(prev is not None and np.array_equal(prev, C)),
                       n_seeds=int(st["n_seeds"]), grad_se=float(st["grad_se"]), grad_signif=float(st["grad_signif"]),
                       gnorm_corr=float(st["gnorm_corr"]), loss_se=float(st["loss_se"]),
                       losses=np.asarray(st["losses"], dtype=float).copy(),
                       G=np.asarray(st["G"], dtype=float).copy(), seeds_used=list(st["seeds_used"]),
                       scale=st["scale"], gscale=st["gscale"], slope_scale=st["slope_scale"], gnorm0=st["gnorm0"])
            self.gcalls.append(rec)
            self._trials_since = 0
            if self.run_dir is not None and (len(self.gcalls) % self.save_every == 0):
                self.save_partial()
        else:
            self.trials.append(dict(C=C.copy(), loss=float(loss), losses=np.asarray(st["losses"], dtype=float).copy(),
                                    t=t, iter=self.n_iterates() - 1))
            self._trials_since += 1
        return loss, g

    # -- outputs ---------------------------------------------------------------------------------
    def progress(self, status="running"):
        it = max(self.n_iterates() - 1, 0)
        el = time.time() - self.t0
        last = self.gcalls[-1] if self.gcalls else None
        eta = (el / it * (self.n_iter - it)) if (self.n_iter and it > 0) else None
        return dict(status=status, iter=it, n_iter=self.n_iter, loss=(last["loss"] if last else None),
                    gnorm=(float(np.linalg.norm(last["grad"])) if last else None),
                    n_seeds=(last["n_seeds"] if last else None), elapsed_s=el, eta_s=eta,
                    n_gradient_calls=len(self.gcalls), n_trials=len(self.trials), pid=__import__("os").getpid(),
                    last_update=time.strftime("%Y-%m-%d %H:%M:%S"))

    def save_partial(self, status="running"):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "progress.json").write_text(json.dumps(self.progress(status), indent=1))
        np.savez(self.run_dir / "history.npz", **self.build_history(None))

    def _iterate_records(self, shape_hist=None):
        """Per-iterate gradient records: the LAST gradient call at each distinct C (a noise-limited refine
        replaces the earlier record, like backtracking_gd's history). Order = order of first appearance,
        or shape_hist's order when given."""
        by_key, order = {}, []
        for k, g in enumerate(self.gcalls):
            key = g["C_raw"].tobytes()
            if key not in by_key:
                order.append(key)
            by_key[key] = (k, g)
        if shape_hist is not None:
            keys = [np.asarray(C, dtype=float).tobytes() for C in shape_hist]
            keys = [k for k in keys if k in by_key]
            if keys:
                order = keys
        return [by_key[k] for k in order]

    def build_history(self, res):
        ri = self.ri
        hist = res["hist"] if res else None
        recs = self._iterate_records(res["shape_hist"] if res else None)
        n = len(recs)
        S = max([len(g["losses"]) for _, g in recs] + [len(t["losses"]) for t in self.trials] + [1])
        P = len(recs[0][1]["C_raw"]) if n else len(ri.C_init)
        last = recs[-1][1] if n else None
        scale = float(last["scale"] or 1.0) if last else 1.0
        h = {}
        h["loss"] = np.array([g["loss"] for _, g in recs])
        h["objective_raw"] = h["loss"] * scale
        h["force"] = np.array([ri.loss_to_force(g["loss"], scale, g["C_raw"]) for _, g in recs])
        h["gnorm"] = np.array([np.linalg.norm(g["grad"]) for _, g in recs])
        for k in ("grad_se", "grad_signif", "gnorm_corr", "loss_se", "t_iter"):
            h[k] = np.array([g[k if k != "t_iter" else "t"] for _, g in recs], dtype=float)
        h["grad_se_unpooled"] = np.array([np.sqrt(g["G"].var(axis=0, ddof=1).sum() / len(g["G"])) if len(g["G"]) > 1 else np.nan
                                          for _, g in recs])
        h["n_seeds"] = np.array([g["n_seeds"] for _, g in recs], dtype=int)
        h["seeds_used"] = _pad([g["seeds_used"] for _, g in recs], S, fill=-1).astype(int) if n else np.zeros((0, S), int)
        h["C_raw"] = np.array([g["C_raw"] for _, g in recs]).reshape(n, P)
        h["C_eff"] = np.array([g["C_eff"] for _, g in recs]).reshape(n, P)
        h["grad"] = np.array([g["grad"] for _, g in recs]).reshape(n, P)
        h["losses_seed"] = _pad([g["losses"] for _, g in recs], S) if n else np.zeros((0, S))
        G = np.full((n, S, P), np.nan)
        for i, (_, g) in enumerate(recs):
            G[i, :len(g["G"])] = g["G"]
        h["G"] = G
        h["perimeter_eff"] = np.array([ri.perimeter(g["C_eff"]) for _, g in recs])
        h["area_eff"] = np.array([ri.area(g["C_eff"]) for _, g in recs])
        h["curv_penalty_eff"] = np.array([ri.curvature_penalty(g["C_eff"]) for _, g in recs])
        h["wall_iter"] = np.diff(h["t_iter"], prepend=0.0) if n else np.zeros(0)
        # trials between consecutive iterate gradient calls; refines = repeated gradient calls at the same C
        gidx = [k for k, _ in recs]
        n_trials, n_ref = np.zeros(n, int), np.zeros(n, int)
        for i, k in enumerate(gidx):
            lo = gidx[i - 1] if i > 0 else -1
            n_trials[i] = sum(1 for gg in self.gcalls[lo + 1:k + 1] for _ in [0]) and sum(
                self.gcalls[j]["n_trials"] for j in range(lo + 1, k + 1))
            n_ref[i] = sum(1 for j in range(lo + 1, k + 1) if self.gcalls[j]["is_refine"])
        h["n_trials"], h["n_refines"] = n_trials, n_ref
        # optimiser-side records
        if hist is not None:
            m = min(n, len(hist["loss"]))
            h["step"] = np.full(n, np.nan); h["step"][:m] = np.asarray(hist["step"][:m], float)
            h["disp"] = np.full(n, np.nan); h["disp"][:m] = np.asarray(hist["disp"][:m], float)
            h["accepted"] = np.asarray(hist["accepted"], bool)
            h["h_secant"] = np.asarray(hist.get("h_secant", []), float)
            h["status"] = np.array(str(hist["status"]))
        else:
            h["step"] = np.full(n, np.nan); h["disp"] = np.full(n, np.nan)
            h["accepted"] = np.zeros(0, bool); h["h_secant"] = np.zeros(0); h["status"] = np.array("running")
        if n >= 2:
            h["disp"][1:] = np.where(np.isnan(h["disp"][1:]), np.linalg.norm(np.diff(h["C_raw"], axis=0), axis=1), h["disp"][1:])
        # all calls
        h["gcall_C"] = np.array([g["C_raw"] for g in self.gcalls]).reshape(len(self.gcalls), P)
        h["gcall_loss"] = np.array([g["loss"] for g in self.gcalls])
        h["gcall_gnorm"] = np.array([np.linalg.norm(g["grad"]) for g in self.gcalls])
        h["gcall_n_seeds"] = np.array([g["n_seeds"] for g in self.gcalls], int)
        h["gcall_t"] = np.array([g["t"] for g in self.gcalls])
        h["gcall_is_refine"] = np.array([g["is_refine"] for g in self.gcalls], bool)
        nt = len(self.trials)
        h["trial_C"] = np.array([t["C"] for t in self.trials]).reshape(nt, P)
        h["trial_loss"] = np.array([t["loss"] for t in self.trials])
        h["trial_losses_seed"] = _pad([t["losses"] for t in self.trials], S) if nt else np.zeros((0, S))
        h["trial_t"] = np.array([t["t"] for t in self.trials])
        h["trial_iter"] = np.array([t["iter"] for t in self.trials], int)
        iter_keys = {g["C_raw"].tobytes(): i for i, (_, g) in enumerate(recs)}
        h["trial_accepted"] = np.array([t["C"].tobytes() in iter_keys for t in self.trials], bool)
        base = [recs[t["iter"]][1]["C_raw"] if 0 <= t["iter"] < n else t["C"] for t in self.trials]
        h["trial_disp"] = np.array([np.linalg.norm(t["C"] - b) for t, b in zip(self.trials, base)])
        # scalars
        h["scale"] = np.array(scale)
        h["gscale"] = np.array(float(last["gscale"]) if (last and last["gscale"]) else 1.0)
        h["slope_scale"] = np.array(float(last["slope_scale"]) if last else 1.0)
        h["gnorm0"] = np.array(float(last["gnorm0"]) if (last and last["gnorm0"] is not None) else np.nan)
        h["L0"] = np.array(float(ri._L0_VALUE) if ri._L0_VALUE is not None else np.nan)
        h["A0"] = np.array(float(ri._A0_VALUE) if ri._A0_VALUE is not None else np.nan)
        h["L0_same"] = np.array(float(res["L0_same"]) if res else np.nan)
        h["wall_total"] = np.array(float(res["wall"]) if res else time.time() - self.t0)
        h["n_iter_requested"] = np.array(int(self.n_iter) if self.n_iter else -1)
        return h
