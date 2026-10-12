# gradstat.npz -- per-seed gradients at one fixed shape (raw units)
- G (K, P): per-seed gradient dJ/dC (constraint-rescaled chain rule applied), K seeds
- losses (K,): per-seed objective J (drag force + curvature penalty)
- seeds (K,), C_eval (P,) raw shape, C_eval_eff (P,) simulated (rescaled) shape
- g_pen (P,): deterministic curvature-penalty part of the gradient (G - g_pen = flow part)
- g_mean (P,), N_REF, N_FOURIER, t_warm (s, cache build/load), t_grad (s, all seeds), n_workers
