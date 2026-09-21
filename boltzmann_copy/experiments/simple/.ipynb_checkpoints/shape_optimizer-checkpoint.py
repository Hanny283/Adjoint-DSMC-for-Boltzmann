"""
shape_optimizer.py
==================
backtracking_gd(C0, evaluate, ...) where
evaluate(C, want_grad) -> (loss, grad_or_None)

MUST evaluate L (and grad) using the SAME fixed seeds on every call, so that
losses at different C are directly comparable.
"""
import numpy as np


class _PooledNoise:
    """dof-weighted pooled variance of the paired CRN loss differences D_s across trials
    (a 3-seed sample has 2 dof, far too few on its own), robustified against the heavy,
    bimodal tails these differences have (max of sample variance and MAD^2), with
    forgetting so the estimate can follow the displacement scale of recent trials."""

    def __init__(self, memory_dof=30):
        self.SS, self.dof, self.mem = 0.0, 0, memory_dof

    def update(self, D):
        n = len(D)
        if n < 2:
            return
        m = D.mean()
        s2 = float(((D - m) ** 2).sum() / (n - 1))
        mad2 = float((1.4826 * np.median(np.abs(D - m))) ** 2)
        self.SS += (n - 1) * max(s2, mad2)
        self.dof += n - 1
        if self.dof > self.mem:
            f = self.mem / self.dof
            self.SS *= f
            self.dof = self.mem

    def sd(self):
        return float(np.sqrt(self.SS / self.dof)) if self.dof >= 2 else None


def backtracking_gd(
    C0,
    evaluate,
    *,
    n_iter=25,
    init_step=0.1,
    step_increase=2.0,
    step_decrease=0.5,
    armijo_c=1e-4,
    max_backtrack=30,
    project=None,
    is_valid=None,
    grad_tol=0.0,
    verbose=True,
    on_accept=None,
    init_displacement=None,
    noise_tol=0.0,
    min_displacement=None,
    converged=None,
    noise_aware=False,
    noise_t=1.645,
    noise_snr=2.0,
    max_refine=3,
    direction_norm=None,
):
    """
    Gradient descent with Armijo backtracking, optionally noise-aware for stochastic
    (Monte-Carlo) objectives evaluated with common random numbers.

    Parameters
    ----------
    C0          : initial parameter vector.
    evaluate    : callable(C, want_grad) -> (loss, grad). Uses fixed seeds (CRN). For a
                  Monte-Carlo objective `grad` may be an estimate of the gradient of the
                  EXPECTED loss (e.g. runDraft adds a horizon term), i.e. not exactly the
                  derivative of the returned sample loss; the Armijo prediction then uses
                  that slope, which is what the noise band is for.
    init_step   : initial line-search step (reused/grown across iterations).
    init_displacement : if given, overrides init_step by the SCALE-AWARE choice
                  init_step = init_displacement / ||grad(C0)||, i.e. the first trial
                  step moves C by `init_displacement` in norm. Use this when the
                  gradient magnitude is not known in advance (e.g. extensive
                  objectives like a raw impulse sum, whose gradient is O(1e4)):
                  a fixed init_step then either does nothing or shoots straight
                  into the feasibility projection.
    direction_norm : if given, the search direction is NORMALISED, d = -grad/||grad|| * direction_norm
                  (a trial with alpha = 1 moves C by direction_norm), while the Armijo test and the
                  noise test keep the TRUE slope g.d. Use it to normalise the gradient without changing
                  the objective (e.g. loss = J/J(C_init) and first steps of 5% of ||C_init||). None
                  (default): d = -grad. All displacement rules below are stated with ||d|| and |g.d|,
                  which reduce to ||g|| and ||g||^2 for the plain direction.
    armijo_c    : sufficient-decrease constant in  L(C+ad) <= L(C) + c*a*kappa*(g.d), where kappa =
                  evaluate.state["slope_scale"] (1 if absent) converts the returned gradient, which may
                  be a constant multiple of the true derivative (e.g. normalised to ||g(C0)|| = 1), back
                  to the true slope of the returned loss. The noise test uses the same kappa.
    noise_tol   : constant relaxation of the Armijo test (accept when
                  L(C+ad) <= L(C) + c*a*(g.d) + noise_tol). 0 = classic rule.
    noise_aware : if True and `evaluate.state["losses"]` exposes the PER-SEED losses of
                  the last call, every trial is judged with the PAIRED differences
                  D_s = L_s(C+ad) - L_s(C) (same seed at both points):
                    se_D = std(D_s, ddof=1)/sqrt(n)
                    accept  iff  mean(D) <= c*a*(g.d) + max(noise_tol, noise_t*se_D)
                                                       (relaxed Armijo, Berahas-Cao-
                                                        Scheinberg style)
                  se_D uses max(sample sd, pooled sd over recent trials) -- see _PooledNoise.
                    noise-limited  iff  a*|g.d| < noise_snr*se_D   (the predicted
                        decrease of this trial cannot be resolved; halving a only makes
                        it worse). Then, if `evaluate.grow()` exists and returns True
                        (more seeds appended), the reference loss/gradient are
                        re-evaluated and the iteration is retried (<= max_refine
                        times); otherwise the run stops with status "noise-limited".
                  Without per-seed losses this reduces to the classic rule.
    min_displacement : stop when an accepted step moved C by less than this norm
                  (||C_new - C_old|| AFTER projection, i.e. the move actually made),
                  and exit the trial loop early once a*||d|| falls below it.
    converged   : optional callable(it, C, loss, grad) -> bool, checked after every
                  accepted step; return True to stop.
    project     : optional callable(C) -> C (feasibility projection).
    is_valid    : optional callable(C) -> bool (reject invalid trial shapes).
    grad_tol    : stop when ||grad|| <= grad_tol.
    on_accept   : optional callable(it, C, loss, grad) after each accepted step.

    Returns
    -------
    C_opt, history : history lists loss/gnorm/step/n_seeds/disp have one entry per
        visited iterate INCLUDING the returned C_opt (so [-1] describes C_opt);
        `accepted` has one entry per attempted iteration; `status` is one of
        "max iterations", "grad_tol", "line search failed", "noise-limited",
        "min_displacement", "converged". n_seeds/disp are None/nan when unavailable.
    """
    C = np.array(C0, dtype=float)
    step = float(init_step)
    history = {"loss": [], "gnorm": [], "step": [], "n_seeds": [], "disp": [],
               "accepted": [], "status": None, "h_secant": []}
    st = getattr(evaluate, "state", None)

    def _losses():
        if noise_aware and isinstance(st, dict) and st.get("losses") is not None:
            v = np.asarray(st["losses"], dtype=float)
            return v if v.size >= 2 else None
        return None

    def _nseeds():
        return int(st["n_seeds"]) if isinstance(st, dict) and "n_seeds" in st else None

    pool = _PooledNoise()

    def _slope_scale():
        """evaluate.state["slope_scale"]: the true directional derivative of the returned loss along d
        is slope_scale * (g . d) when the returned gradient is a constant multiple of the derivative
        (e.g. normalised to unit norm at the start). 1 when absent."""
        v = st.get("slope_scale") if isinstance(st, dict) else None
        return float(v) if v is not None else 1.0

    def _direction(g):
        d = -np.asarray(g, dtype=float)
        if direction_norm is not None:
            d *= float(direction_norm) / max(float(np.linalg.norm(g)), 1e-300)
        return d

    loss, grad = evaluate(C, want_grad=True)
    ref_losses = _losses()
    if init_displacement is not None:
        step = float(init_displacement) / max(float(np.linalg.norm(_direction(grad))), 1e-300)
        if verbose:
            print(f"init: ||grad||={np.linalg.norm(grad):.3e} -> init_step={step:.3e} "
                  f"(first trial moves C by {init_displacement})")

    def _record(disp):
        history["loss"].append(loss)
        history["gnorm"].append(float(np.linalg.norm(grad)))
        history["step"].append(step)
        history["n_seeds"].append(_nseeds())
        history["disp"].append(disp)

    last_disp = float("nan")
    pending = False
    status = "max iterations"
    for it in range(n_iter):
        gnorm = float(np.linalg.norm(grad))
        _record(last_disp)
        pending = False
        if gnorm <= grad_tol:
            if verbose:
                print(f"iter {it:3d}: ||grad||={gnorm:.3e} <= grad_tol, converged")
            history["accepted"].append(False)
            status = "grad_tol"
            break
        d = _direction(grad)
        gd = float(grad @ d)               # true slope along d: -||grad||^2 (plain) or -||grad||*norm (normalised)
        dnorm = float(np.linalg.norm(d))   # |dC| per unit alpha
        alpha = step
        accepted = False
        refines = 0
        C_prev = C
        while True:
            noise_limited = False
            reason = ""
            for _bt in range(max_backtrack):
                if min_displacement is not None and alpha * dnorm < min_displacement:
                    noise_limited = True
                    reason = f"trial displacement {alpha*dnorm:.2e} < min_displacement {min_displacement:g}"
                    break
                C_try = C + alpha * d
                if project is not None:
                    C_try = np.asarray(project(C_try), dtype=float)
                if is_valid is not None and not is_valid(C_try):
                    alpha *= step_decrease
                    continue
                # CRN: same seeds inside evaluate -> comparable to `loss`
                loss_try, _ = evaluate(C_try, want_grad=False)
                band = noise_tol
                try_losses = _losses()
                if try_losses is not None and ref_losses is not None and len(try_losses) == len(ref_losses):
                    D = try_losses - ref_losses
                    pool.update(D)
                    sd = max(float(np.std(D, ddof=1)), pool.sd() or 0.0)
                    se_D = sd / np.sqrt(len(D))
                    band = max(noise_tol, noise_t * se_D)
                    if alpha * (-gd) * _slope_scale() < noise_snr * se_D:   # predicted decrease alpha*kappa*|g.d|
                        noise_limited = True
                        reason = (f"predicted decrease {alpha*(-gd)*_slope_scale():.3g} < {noise_snr:g} x paired "
                                  f"loss s.e. {se_D:.3g} at |dC|={alpha*dnorm:.3g}")
                        break
                if loss_try <= loss + armijo_c * alpha * _slope_scale() * gd + band:   # true slope kappa*(g.d)
                    if project is not None and verbose:
                        clip = float(np.linalg.norm(C_try - (C + alpha * d)))
                        if clip > 1e-12:
                            print(f"iter {it:3d}: feasibility projection active (moved trial point by {clip:.3g}); "
                                  f"the raw gradient norm need not vanish at a constrained optimum")
                    C = C_try
                    step = alpha * step_increase     # grow the trial step for next time
                    accepted = True
                    break
                alpha *= step_decrease
            if accepted or not noise_limited:
                break
            grow = getattr(evaluate, "grow", None)
            if grow is not None and refines < max_refine and grow():
                refines += 1
                loss, grad = evaluate(C, want_grad=True)      # same point, more seeds
                ref_losses = _losses()
                gnorm = float(np.linalg.norm(grad))
                d = _direction(grad)
                gd = float(grad @ d)
                dnorm = float(np.linalg.norm(d))
                if verbose:
                    print(f"iter {it:3d}: noise-limited ({reason}) -> seeds grown to {_nseeds()}, retrying")
                continue
            status = "noise-limited"
            if verbose:
                print(f"iter {it:3d}: noise-limited ({reason}) and no more precision available -> stopping")
            break
        history["accepted"].append(accepted)
        if not accepted:
            if status != "noise-limited":
                status = "line search failed"
                if verbose:
                    print(f"iter {it:3d}: line search failed (no sufficient decrease) -- stopping")
            break

        # gradient (and refreshed loss) at the accepted point, same fixed seeds
        grad_prev = grad
        loss, grad = evaluate(C, want_grad=True)
        ref_losses = _losses()
        last_disp = float(np.linalg.norm(C - C_prev))   # the move actually made (after projection)
        s_vec = C - C_prev
        if last_disp > 0:                                # secant curvature along the accepted step
            history["h_secant"].append(float((grad - grad_prev) @ s_vec) / float(s_vec @ s_vec))
        pending = True
        if on_accept is not None:
            on_accept(it, C, loss, grad)
        if verbose:
            print(f"iter {it:3d}: L={loss:.6f}  ||grad||={np.linalg.norm(grad):.4e}  "
                  f"step={step:.4g}  alpha={alpha:.4g}  |dC|={last_disp:.3g}")
        if min_displacement is not None and last_disp < min_displacement:
            if verbose:
                print(f"iter {it:3d}: accepted step moved C by {last_disp:.2e} < min_displacement="
                      f"{min_displacement:g} -> stopping (noise floor reached)")
            status = "min_displacement"
            break
        if converged is not None and converged(it, C, loss, grad):
            if verbose:
                print(f"iter {it:3d}: convergence test satisfied -> stopping")
            status = "converged"
            break
    if pending:
        _record(last_disp)                # so history[-1] describes the returned C
    elif history["loss"]:
        # same iterate as the last record, but loss/grad/seeds may have been refreshed
        # (noise-limited refine re-evaluates C with more seeds before giving up)
        history["loss"][-1] = loss
        history["gnorm"][-1] = float(np.linalg.norm(grad))
        history["n_seeds"][-1] = _nseeds()
    history["status"] = status
    return C, history


def stochastic_endgame(
    C0,
    evaluate,
    *,
    n_iter=6,
    max_disp=0.05,
    min_disp=1e-3,
    signif_stop=1.5,
    reject_z=2.0,
    h_bounds=None,
    project=None,
    is_valid=None,
    verbose=True,
    on_accept=None,
    h_init=None,
    grad_tol=None,
):
    """
    Near the optimum of a stiff objective the loss change of a correct step (~|g|^2/2H) drops
    below the loss noise while the gradient estimate still carries a clear signal, so a
    loss-based Armijo test stalls at ||g|| ~ 2 s.e.(loss)/|dC|_max even though the gradient
    is far from statistically zero. Here steps are d = -g/h with h a damped secant (BB)
    curvature estimate along the path, capped at max_disp; the paired per-seed loss change
    only VETOES a step that is significantly worse (mean > reject_z * s.e.), in which case
    h is doubled and the step retried. The run stops when the gradient is statistically
    consistent with zero, ||g|| / s.e.(g) < signif_stop -- the meaningful "||g|| -> 0" for
    a Monte-Carlo gradient, whose raw norm can never fall below its own standard error --
    or when a step moves C by less than min_disp.

    Requires evaluate.state with 'losses' (per-seed losses of the last call), 'grad_signif'
    and 'gnorm_corr' (as runDraft.make_evaluate_adjoint provides).
    Returns C, history(dict: loss, gnorm, gnorm_corr, signif, disp, h, status).
    """
    st = evaluate.state
    C = np.array(C0, dtype=float)
    loss, grad = evaluate(C, want_grad=True)
    ref = np.asarray(st["losses"], dtype=float)
    hist = {"loss": [loss], "gnorm": [float(np.linalg.norm(grad))], "gnorm_corr": [st["gnorm_corr"]],
            "signif": [st["grad_signif"]], "disp": [float("nan")], "h": [float("nan")], "status": "max iterations"}
    h = float(np.linalg.norm(grad)) / max_disp          # first step moves at most max_disp ...
    if h_init is not None and np.isfinite(h_init) and h_init > 0:
        h = max(float(h_init), h)                        # ... unless a curvature estimate says less
    for it in range(n_iter):
        if st["grad_signif"] < signif_stop:
            hist["status"] = "gradient consistent with zero"
            if verbose:
                print(f"endgame {it:2d}: ||g||/s.e.={st['grad_signif']:.2f} < {signif_stop} -> gradient "
                      f"statistically indistinguishable from zero at {st['n_seeds']} seeds; stopping")
            break
        if grad_tol is not None and np.isfinite(st.get("gnorm_corr", np.nan)) and st["gnorm_corr"] <= grad_tol:
            hist["status"] = "gradient tolerance reached"
            if verbose:
                print(f"endgame {it:2d}: noise-corrected ||g||={st['gnorm_corr']:.4g} <= grad_tol={grad_tol:.4g}; stopping")
            break
        accepted = False
        for _try in range(4):
            d = -grad / h
            nd = float(np.linalg.norm(d))
            if nd > max_disp:
                d *= max_disp / nd
            C_try = C + d
            if project is not None:
                C_try = np.asarray(project(C_try), dtype=float)
            if is_valid is not None and not is_valid(C_try):
                h *= 2.0
                continue
            loss_try, _ = evaluate(C_try, want_grad=False)
            D = np.asarray(st["losses"], dtype=float) - ref
            se_D = float(np.std(D, ddof=1) / np.sqrt(len(D))) if len(D) > 1 else float("inf")
            if len(D) > 1 and D.mean() > reject_z * se_D:
                if verbose:
                    print(f"endgame {it:2d}: step |dC|={np.linalg.norm(C_try - C):.3g} significantly worse "
                          f"(dL={D.mean():+.3g} vs s.e. {se_D:.3g}) -> curvature h {h:.3g} -> {2*h:.3g}, retry")
                h *= 2.0
                continue
            accepted = True
            break
        if not accepted:
            hist["status"] = "no acceptable step"
            break
        s_vec = C_try - C
        loss_new, grad_new = evaluate(C_try, want_grad=True)
        ref = np.asarray(st["losses"], dtype=float)
        y = grad_new - grad
        hs = float(y @ s_vec) / float(s_vec @ s_vec)          # secant curvature along the step
        if np.isfinite(hs) and hs > 0:
            if h_bounds is not None:
                hs = float(np.clip(hs, *h_bounds))
            h = 0.5 * h + 0.5 * hs                           # damped: gradient noise makes hs ~50% noisy
        disp = float(np.linalg.norm(s_vec))
        C, loss, grad = C_try, loss_new, grad_new
        hist["loss"].append(loss); hist["gnorm"].append(float(np.linalg.norm(grad)))
        hist["gnorm_corr"].append(st["gnorm_corr"]); hist["signif"].append(st["grad_signif"])
        hist["disp"].append(disp); hist["h"].append(h)
        if on_accept is not None:
            on_accept(it, C, loss, grad)
        if verbose:
            print(f"endgame {it:2d}: L={loss:.4g}  ||g||={np.linalg.norm(grad):.4g}  ||g||corr={st['gnorm_corr']:.4g}  "
                  f"||g||/s.e.={st['grad_signif']:.2f}  |dC|={disp:.3g}  h={h:.3g}")
        if disp < min_disp:
            hist["status"] = "min_disp"
            break
    else:
        if st["grad_signif"] < signif_stop:
            hist["status"] = "gradient consistent with zero"
    return C, hist
