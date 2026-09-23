"""Success classifier for one optimisation run, a pure function of the saved history arrays.

decayed : the objective went down significantly and is not rising again at the end
plateau : the gradient norm is flat over the tail (small coefficient of variation, no net drift)
at_noise_floor : ||g|| is statistically indistinguishable from its own standard error (needs >= 2 seeds)
verdict : success | decaying_noisy | stalled | diverged | incomplete
"""
import numpy as np


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def classify_run(h, *, cv_max=0.35, drift_max=np.log(1.5), z_min=3.0, rel_min=0.02, signif_max=2.0):
    loss = np.asarray(h["loss"], float)
    gnorm = np.asarray(h["gnorm"], float)
    n = len(loss)
    out = dict(n_iter_done=max(n - 1, 0), status=str(h["status"]) if "status" in h else "unknown")
    if n < 2:
        out.update(verdict="incomplete", decayed=None, plateau=None, at_noise_floor=None)
        return out
    m = max(3, n // 3)
    m = min(m, n)
    S = int(np.max(h["n_seeds"])) if "n_seeds" in h else 1
    # ---- decay ----
    tail = loss[-m:]
    out["decay_rel"] = float((loss[0] - np.median(tail)) / abs(loss[0])) if loss[0] != 0 else float("nan")
    ls = np.asarray(h["losses_seed"], float) if "losses_seed" in h else None
    if ls is not None and ls.shape[1] >= 2 and np.isfinite(ls[0]).sum() >= 2 and np.isfinite(ls[-1]).sum() >= 2:
        k = min(int(np.isfinite(ls[0]).sum()), int(np.isfinite(ls[-1]).sum()))
        D = ls[-1, :k] - ls[0, :k]                       # paired CRN differences, same seeds
        sd = float(np.std(D, ddof=1)) if k > 1 else float("nan")
        out["decay_z"] = float(-D.mean() / (sd / np.sqrt(k))) if sd > 0 else float("inf")
    else:
        out["decay_z"] = float("nan")
    prev = loss[-2 * m:-m] if n >= 2 * m else loss[:max(n - m, 1)]
    out["late_increase"] = float(np.median(tail) - np.median(prev))
    lse = _finite(h["loss_se"][-m:]) if "loss_se" in h else np.zeros(0)
    late_ok = (out["late_increase"] <= 2.0 * float(np.median(lse))) if lse.size else (out["late_increase"] <= 0.0)
    if np.isfinite(out["decay_z"]):
        out["decayed"] = bool(out["decay_z"] > z_min and late_ok)
    else:
        out["decayed"] = bool(out["decay_rel"] > rel_min and late_ok)
    # ---- plateau of ||g|| ----
    gt = gnorm[-m:]
    out["gnorm_tail_mean"] = float(np.mean(gt))
    out["gnorm_tail_cv"] = float(np.std(gt) / np.mean(gt)) if np.mean(gt) > 0 else float("nan")
    if m >= 2 and np.all(gt > 0):
        x = np.arange(m)
        slope = float(np.polyfit(x, np.log(gt), 1)[0])
    else:
        slope = float("nan")
    out["gnorm_tail_slope"] = slope
    out["plateau"] = bool(out["gnorm_tail_cv"] < cv_max and (np.isnan(slope) or abs(slope) * (m - 1) < drift_max))
    # ---- noise floor ----
    sig = _finite(h["grad_signif"][-m:]) if "grad_signif" in h else np.zeros(0)
    if S >= 2 and sig.size:
        out["signif_tail_median"] = float(np.median(sig))
        out["at_noise_floor"] = bool(out["signif_tail_median"] < signif_max)
        reach = np.nonzero(np.asarray(h["grad_signif"], float) < signif_max)[0]
        out["iter_reach_floor"] = int(reach[0]) if reach.size else None
    else:
        out["signif_tail_median"] = float("nan")
        out["at_noise_floor"] = None
        out["iter_reach_floor"] = None
    # ---- verdict ----
    st = out["status"]
    n_req = int(h["n_iter_requested"]) if "n_iter_requested" in h and int(h["n_iter_requested"]) > 0 else None
    if loss[-1] > loss[0]:
        v = "diverged"
    elif out["decayed"] and out["plateau"] and out["at_noise_floor"] is not False:
        v = "success"
    elif out["decayed"]:
        v = "decaying_noisy"
    elif st in ("line search failed", "noise-limited") and (n_req is None or n - 1 < n_req / 2):
        v = "stalled"
    else:
        v = "incomplete"
    out["verdict"] = v
    # ---- handy scalars ----
    out["loss_first"], out["loss_last"] = float(loss[0]), float(loss[-1])
    if "force" in h:
        out["force_first"], out["force_last"] = float(h["force"][0]), float(h["force"][-1])
    if "disp" in h and n >= 2:
        out["first_step_disp"] = float(h["disp"][1])
    if "n_trials" in h and n >= 2:
        out["backtracks_it0"] = int(h["n_trials"][1])
    if "wall_iter" in h and n >= 2:
        out["wall_per_iter_s"] = float(np.mean(h["wall_iter"][1:]))
    return out
