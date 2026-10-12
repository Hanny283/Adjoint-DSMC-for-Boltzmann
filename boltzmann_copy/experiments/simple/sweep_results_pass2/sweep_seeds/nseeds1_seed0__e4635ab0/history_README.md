# history.npz -- one optimisation run (arrays; n = iterates visited incl. the start, P = 2*N_FOURIER+1,
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
