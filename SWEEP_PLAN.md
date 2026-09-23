# Parameter study for the adjoint-DSMC shape optimiser (`experiments/simple/runInflow.py`)

## Context

`runInflow.py` minimises the drag on a Fourier-parameterised body in an open DSMC channel with an
exact pathwise adjoint gradient and a noise-aware Armijo line search (`shape_optimizer.py`). The
user needs to (1) see how four knobs — number of CRN seeds, number of particles, number of Fourier
modes, `GRAD_INIT_FRAC` — affect a run, (2) get publication-quality figures with the plotted
numbers saved, and (3) drive everything from Jupyter on a 16-core Cornell machine. Today the script
saves only a PNG and a GIF, evaluates seeds serially on one core, and hard-codes every knob as a
module global. A run counts as "successful" when the objective decays and the gradient norm plateaus
at a constant (its Monte-Carlo noise floor).

Decisions taken with the user: study BOTH the per-mode gradient statistics at a fixed shape (cheap)
and the optimised coefficients from full runs; baseline **N_REF = 250k, 4 seeds, N_FOURIER = 10,
GRAD_INIT_FRAC = 0.05, N_ITER = 20**; **minimal first pass** (baseline + one run per knob value, no
replicates; replicates can be added later because results are cached per config); detached
background processes are allowed; the boundary-crossing ("single crossing") check is deferred — the
plan only guarantees that the per-iterate shape history is saved so it can be added post hoc.

## 0. Deployment target, access and transport (decided with the user, 2026-09-22)

- **Machine:** `fibonacci.math.cornell.edu` (AMD Ryzen 9 5950X, 16 cores / 32 threads, 128 GB RAM, shared).
  Alternatives from the department list if it is busy: `ramsey`, `boole` (same class), `hopper` (16 c, 256 GB),
  `kraken` (64 c, 512 GB). Always check `uptime` / `who` before starting 16 workers.
- **Access:** no SSH from the Mac (department SSH is filtered except the portal gateway; the CU VPN
  departmental login fails). The user works in the Math Portal's Jupyter and terminal on the machine, where
  Claude Code is installed (`~/.local/bin/claude`). Math account `hec66`, home `/homes/hec66`.
- **Transport = git via GitHub:** repo `Hanny283/Adjoint-DSMC-for-Boltzmann` (public; root holds
  `boltzmann_copy/`). The Mac pushes over SSH (`git@github.com:...`, key works as Hanny283). The machine
  clones over HTTPS and pushes with a fine-grained personal access token stored by
  `git config --global credential.helper store`. Code flows Mac → GitHub → machine (`git pull` in the
  terminal or a `!git pull` notebook cell); results flow back with `git push` of `sweep_results/` (plain files
  only) or a zip download from Jupyter. This plan lives in the repo as `SWEEP_PLAN.md`.
- **Repo hygiene:** `.gitignore` excludes `__pycache__`, `.ipynb_checkpoints`, `warm_cache/`, `*.prof`,
  `*.gif`, `*.mp4`. On the machine clone to `~/Adjoint-DSMC-for-Boltzmann`; if an older copy with a
  `warm_cache/` exists under `~`, symlink it to `boltzmann_copy/experiments/simple/warm_cache` (warm states
  are keyed by physics + seed, so reuse is safe).
- **Self-describing outputs (no code travels with results):** every run dir has `config.json`, `history.npz`
  plus `README.md` describing each array, `summary.json`, `quicklook.png`; the analysis writes
  `summary_table.csv`, every figure as PNG + CSV of its values, and a rendered `report.html`. The analysis
  depends only on numpy, pandas and matplotlib and runs on either machine (Mac: Python 3.12; machine: 3.11).
- **Sizing:** 16 cores bound the concurrency, memory does not (128 GB vs ≈10 GB peak for the first pass).

## 1. Where the parameters live (all module globals in `experiments/simple/runInflow.py`)

| Knob | Global(s) | Lines | Notes |
|---|---|---|---|
| particles | `N_REF`, `CAP` | 60–61 | change ONLY via `set_particle_count(n_ref, cap)` (l.144) — it recomputes `M_P`, `N0`; setting `N_REF` alone silently corrupts the collision rate and inflow |
| Fourier modes | `N_FOURIER`, `C_init` | 74–75 | set both (`C_init = zeros(2N+1); C_init[0] = 1`); `backward_pass_impulse` / `edge_term` read the global `N_FOURIER` (l.307, 448, 451, 475) |
| seeds | `N_AVG` (initial), `N_AVG_MAX` (cap), `ADAPTIVE_SEEDS`, `GRAD_SIGNIFICANCE` | 94–97 | seed list is `range(N_AVG)` hard-coded in `main()` l.1559; grows via `grow()` l.598. **No seed-offset knob exists** |
| grad_init frac | `GRAD_INIT_FRAC` | 128 | first line-search trial moves C by `GRAD_INIT_FRAC·‖C_init‖` (`init_displacement`, l.1558/1607); also the endgame step cap (l.1616) and the scale in `NORMALIZE_MODE="gradient"` (l.578). It is transient: after the first trial the step evolves by ×1.5 / ×0.5 |
| iterations / stopping | `N_ITER` 88, `STOP_ON_SIGNIFICANCE` 130, `GRAD_TOL_REL` 129, `MIN_DISPLACEMENT` 98, `ENDGAME` 100 | | sweeps use `STOP_ON_SIGNIFICANCE=False`, `ENDGAME=False` |
| objective / constraint | `OBJECTIVE` 105, `CONSTRAINT` 109, `CURV_LAMBDA` 114, `NORMALIZE_MODE` 118 | | keep baseline values |
| warm start | `T_WARM` 64, `REWARM_T` 1081, `WARM_VERSION` 1080 | | cache key `_warm_key` (l.1087) includes `N_REF`, `CAP`, box and the `C_init` tuple, so each N_FOURIER re-warms the same circle once (≈3 min/seed at 250k; accepted) |
| time window | `TOBS` 55, `DT` 50, `N_STEPS` 91 | | `N_STEPS` is computed at import; a harness that changes `TOBS` must set `N_STEPS` too |
| outputs | `OUT_DIR`, `PLOT_NAME`, `SHAPE_GIF_NAME` | 36–38 | hand-edited names, re-runs overwrite; `OUT_DIR` is also the warm-cache root (l.1103) so it must not be repointed per run |

Facts that matter: with the current `N_AVG = N_AVG_MAX = 1` every variance quantity is inert
(`grad_se = inf`, `gnorm_corr = nan`, no noise band, `converged()` can never fire — visible as
`||g||/s.e.=inf` in the saved notebook output). That output is also the cost anchor:
**≈290 s per accepted iteration per seed at N_REF = 1M** (iteration 8 at 2713 s).

## 2. Can it run in parallel? Yes, by processes at two levels (threads give nothing)

- One process = one core: the hot paths are numpy ufuncs, `lexsort`, `bincount`, fancy indexing and a
  pure-Python per-hit loop in `backward_pass_impulse` (l.362). The only BLAS calls are small `@`
  products in `_batch_radius_and_slope`; BLAS threads must be pinned to 1 (oversubscription and
  bitwise reproducibility).
- Seeds are independent: all randomness is the label hash `u01(seed, tag, …)` (l.740) and
  `_reflection_rng(seed, k)` (l.238); no global RNG. The only per-seed mutable state is the moving
  warm state `_WARM_CACHE[_warm_key(seed)]` + `_WARM_STEP[seed]`, mutated in place by `rewarm` (l.1138).
  Shared caches (`_GAS_FRAC_CACHE`, `_FLUX_TABLE`, `_BACK_TABLE`) are pure functions of shape/physics.
- The serial seed loops are l.635 (loss-only trials), l.639 (`_per_seed` gradients) and l.1580
  (`rewarm` per seed in `_Evaluate.__call__`). Parallelising them cuts wall time per iteration by
  ≈n_seeds and gives bitwise-identical results.
- Memory per concurrent seed (docstring l.69–72, scaled): ≈0.6 GB at 250k, 1.2 GB at 500k, 2.3 GB at 1M.

**Level 1 – seeds inside a run** (`sweep/pool.py`, `SeedPool`): a `ProcessPoolExecutor` (spawn context,
`initializer` imports `runInflow`, applies the config and pins BLAS threads). Workers are stateless;
the main process owns each seed's warm state and hands it in and out explicitly (13 MB at 250k, 50 MB
at 1M per call — negligible against ~100 s of compute). This avoids the divergence a fork-per-call
pool would suffer from `rewarm`'s in-place mutation. API: `warm(seeds)`, `rewarm_and_grad(C, C_sim,
seeds) -> [(J, g, g_tilde, e)]` (worker installs the state into `ri._WARM_CACHE/_WARM_STEP`, calls
`rewarm` then `per_seed_eval`, returns results + updated state), `losses(C_sim, seeds)`.
`n_workers` may be smaller than `n_seeds` when memory is tight. `pool=None` keeps the serial path.

**Level 2 – runs as separate OS processes** (`sweep/launcher.py`): each run is
`python -m sweep.run --config …` started detached (`start_new_session=True`, stdout → `log.txt`,
`OMP/OPENBLAS/MKL_NUM_THREADS=1`, `MPLBACKEND=Agg`, `os.nice(10)`); a detached scheduler keeps
`Σ n_workers ≤ cores` and `Σ memory ≤ budget` (from `/proc/meminfo MemAvailable`), so on fibonacci's 16 cores
e.g. four 4-seed runs at 250k run at once (≈10 GB of its 128 GB). `--max-cores` defaults to
`os.cpu_count()//2` (physical cores; the machine is shared) and is raised explicitly when it is idle.

Later speed-ups, not in scope: vectorising the per-hit adjoint loop (≈54 of 85 s per gradient at 500k).

## 3. Minimal, backward-compatible edits to `runInflow.py` (`python runInflow.py` unchanged)

1. `SEED0 = 0` global next to `N_AVG`; the seed list becomes `range(SEED0, SEED0 + N_AVG)`.
2. Lift the closure `_per_seed` (l.614–629) to module level as `per_seed_eval(C, C_sim, s)`;
   `make_evaluate_adjoint(seeds, pool=None)` uses `pool.rewarm_and_grad` / `pool.losses` when given,
   else the existing list comprehensions. `grow()` unchanged.
3. Export per-seed data in `state` (l.661): `G = G/(sc*gs)` (same units as `g`, so `G.mean(0) == g`)
   and `seeds_used = list(seeds)`.
4. Gradient length from `len(C)` instead of the global at l.307, 448, 451, 475 (no behaviour change;
   removes a crash when `C_init` and `N_FOURIER` disagree).
5. Factor `main()` (l.1542–1632) into
   `run_optimisation(seeds=None, *, pool=None, wrap_evaluate=None, make_plots=True, plot_out=None,
   gif_out=None, reeval_init=True) -> dict(C_opt, hist, shape_hist, evaluate, scale, L0_same, seeds, wall)`
   with the body unchanged; `_Evaluate.__call__` routes `rewarm` through the pool when given;
   `wrap_evaluate` lets the harness wrap the evaluator (it then sees every gradient call and every
   loss-only trial); `reeval_init=False` skips the extra forward pass at l.1622; `main()` calls
   `run_optimisation()`. `plot_results(..., out=None)` gets an explicit output path
   (`shape_evolution_gif` already accepts one).
6. `warm_state`: atomic cache write (`tmp` + `os.replace`) and treat a corrupt file as a miss.

`shape_optimizer.py`: no changes (its `history` plus the recorder cover everything).

## 4. New package `experiments/simple/sweep/` + notebooks

| File | Responsibility |
|---|---|
| `config.py` | `TUNABLES` (explicit list of overridable runInflow globals), `defaults()` snapshot at import, `resolve(cfg)` (fills missing keys, `CAP = round(1.3·N_REF)` when null, validates), `run_id(cfg)` = sha1 of `{kind, params, seed0, gradstat, WARM_VERSION, HARNESS_VERSION}`[:10], `run_dir(root, cfg)` |
| `apply.py` | `apply_config(ri, params)` context manager (save → set in the safe order → restore + `reset_caches`). Order: box/times then `N_STEPS = round(TOBS/DT)` → physics flags → `set_particle_count` → `N_FOURIER` + rebuilt unit-circle `C_init` → `set_reference_shape` → optimiser knobs → `reset_caches` (`_WARM_CACHE`, `_WARM_STEP`, `_GAS_FRAC_CACHE`, `LAST_DIAG`) |
| `pool.py` | `SeedPool` (§2) |
| `recorder.py` | `Recorder(evaluate)` wrapper: records every gradient call and loss-only trial (C raw/effective, loss, g, state scalars, `G`, `losses`, wall clock), builds the `history.npz` arrays, rewrites `history.npz` + `progress.json` after every accepted iterate |
| `jobs.py` | `run_optimise(cfg, run_dir)`, `run_gradstat(cfg, run_dir)`, `run_warm(cfg, run_dir)` |
| `run.py` | CLI `python -m sweep.run --config <dir>/config.json [--force]`; skips when `DONE` exists; writes `FAILED` + traceback on exception |
| `launcher.py` | detached scheduler `python -m sweep.launcher --queue <group>/queue.json --max-cores 16 --mem-budget-gb <n>`; `status(root)` DataFrame from `progress.json`; `tail(run)` |
| `classify.py` | success classifier (§6), pure functions of the saved arrays |
| `analysis.py` | `load_runs(root) -> DataFrame`, `load_history`, gradstat aggregation, one function per figure (§7), each saving `.png` + `.pdf` + `.csv`/`.npz` of the plotted values |
| `grids.py` | baseline + one-factor-at-a-time grids (§5) → list of configs; `queue.json` writer |
| `selftest.py` | pure unit checks (< 5 s) |
| `01_smoke_test.ipynb`, `02_run_sweeps.ipynb`, `03_analysis.ipynb` | §8 |

Config JSON: `{"kind": "optimise"|"gradstat"|"warm", "group": "...", "name": "...", "params": {<runInflow
global names>: value, ...}, "seed0": 0, "gradstat": {"seeds": [...], "C_eval": null|[...]},
"options": {"n_workers": null, "reeval_init": false, "make_gif": false}}` (`group`, `name`, `options`
are not hashed). Results: `experiments/simple/sweep_results/<group>/<name>__<id8>/` containing
`config.json`, `log.txt`, `progress.json` (status, iter, loss, gnorm, elapsed, ETA, pid), `history.npz`,
`summary.json`, `quicklook.png`, `DONE`/`FAILED`. The warm cache stays shared in `OUT_DIR/warm_cache`.

`history.npz` (n = iterates, P = 2N+1, S = max seeds, n_g gradient calls, n_t loss-only trials):
`loss, gnorm, step, disp, grad_se, grad_signif, gnorm_corr, loss_se, grad_se_unpooled, objective_raw,
force, perimeter_eff, area_eff, curv_penalty_eff, t_iter, wall_iter (n,)`; `n_seeds, n_trials,
n_refines (n,) int`; `C_raw, C_eff, grad (n, P)`; `losses_seed (n, S)`, `G (n, S, P)` NaN-padded,
`seeds_used (n, S)`; `accepted (n_attempts,)`, `h_secant`; trial records `trial_C (n_t, P), trial_loss,
trial_losses_seed, trial_disp, trial_iter, trial_accepted, trial_t`; gradient-call records `gcall_C,
gcall_loss, gcall_gnorm, gcall_n_seeds, gcall_t`; scalars `scale, gscale, slope_scale, gnorm0, L0, A0,
L0_same, wall_total, status`. `gradstat.npz`: `G (K, P)` raw units, `losses, Ix, seeds (K,)`, `C_eval,
C_eval_eff, g_pen, g_mean (P,)`, `wall_seed (K,)`.

## 5. Experiments — minimal first pass (baseline B0 = 250k, 4 seeds, N_FOURIER 10, frac 0.05, 20 it, seed0 0)

Cost model (from the notebook anchor): iteration ≈ 5 min × N_REF/1M per seed, seeds in parallel;
warm-up ≈ 12 min × N_REF/1M per seed, cached on disk. All sweep runs: `ADAPTIVE_SEEDS=False`,
`N_AVG = N_AVG_MAX = n_seeds`, `STOP_ON_SIGNIFICANCE=False`, `ENDGAME=False`, `reeval_init=False`,
`NORMALIZE_MODE="both"`.

| Exp | Purpose | Configs | Wall each (seed-parallel) | Cores |
|---|---|---|---|---|
| E0 warm | pre-build caches so concurrent runs never build the same (seed, N_REF) | seeds 0–15 at 100k/250k/500k/1M, and seeds 0–3 for N_FOURIER 3/5/20 at 250k | 1–12 min per N_REF (16 seeds in parallel) | 16 |
| E1 gradstat | mean/variance of every gradient component vs particles and seeds at a fixed shape; noise floor of ‖ĝ‖; direction stability | N_REF ∈ {100k, 250k, 500k, 1M} × K = 16 seeds × shapes {`C_init`, `C_def = [1, 0, 0.15, 0, …]` (a₂ mode, rewarmed like the optimiser)}, `NORMALIZE_MODE=None` (raw units — in "both" mode ‖g(C_init)‖ = 1 by construction) | 3–15 min per (N_REF, shape) | 16 |
| B0 | baseline trajectory | 1 run | ≈25 min | 4 |
| E2 seeds | convergence and final coefficients vs number of CRN seeds | n_seeds ∈ {1, 2, 8, 16} | ≈25 min | 1/2/8/16 |
| E3 particles | same vs N_REF | {100k, 500k, 1M} | 10 / 50 / 100 min | 4 |
| E4 modes | same vs N_FOURIER | {3, 5, 20} | ≈25 min | 4 |
| E5 grad_init_frac | first-step size vs behaviour (backtracks at iteration 0, first accepted displacement, loss after 1/5/20 iterations, whole trajectory kept) | {0.01, 0.02, 0.1, 0.2} | ≈25 min | 4 |

15 optimisation runs ≈ 2 200 core-minutes → **≈3–4 h wall on 16 cores** including E0/E1. Order:
E0 → E1 → B0 → E5 → E2 → E4 → E3 (1M last). Phase 2 (later, same harness): replicates
`seed0 ∈ {100, 200}` of B0 and of each knob value give the across-run mean/variance of the optimised
coefficients; `grids.py` already defines them so they are one `launch()` call away.

Built-in sanity checks for E1: at `C_init` the perimeter-constrained gradient's `c0` component is
exactly 0 and the `b_k` components have zero mean (y-symmetry); a₂ at `C_def` breaks both.

## 6. Success classifier (`sweep/classify.py`; tail `m = max(3, n//3)`, thresholds are kwargs)

- Decay: `decay_rel = (loss[0] − median(loss[−m:]))/|loss[0]|`; paired CRN z-score
  `decay_z = −mean(D)/(std(D)/√S)` with `D_s = losses_seed[−1,s] − losses_seed[0,s]` (NaN if S < 2);
  `late_increase = median(loss[−m:]) − median(loss[−2m:−m])`.
  `decayed = (decay_z > 3 if S ≥ 2 else decay_rel > 0.02) and late_increase ≤ 2·median(loss_se[−m:])`.
- Plateau: `gnorm_tail_cv = std/mean of gnorm[−m:] < 0.35` and `|slope of log gnorm over tail|·(m−1) < log 1.5`;
  `at_noise_floor = median(grad_signif[−m:]) < GRAD_SIGNIFICANCE` (None when S < 2);
  `iter_reach_floor` = first iterate with `grad_signif < 2`.
- `verdict`: `success` (decayed ∧ plateau ∧ at_noise_floor ≠ False), `decaying_noisy`, `stalled`
  (line search failed / noise-limited before n_iter/2), `diverged` (`loss[−1] > loss[0]`), `incomplete`.
  Also stored: `status`, `n_iter_done`, `first_step_disp = disp[1]`, `backtracks_it0 = n_trials[1]`,
  force first/last, wall per iteration.

## 7. Figures (`analysis.py`; each saves png + pdf + csv/npz of the plotted arrays; house palette `SEQ_BLUE`; load the `dataviz` skill when implementing)

1. `grad_mean_vs_nref` — per mode: mean ± s.e. over 16 seeds vs N_REF (also flow-only `G − g_pen`), both shapes.
2. `grad_sd_vs_nref` — per-seed sd (norm and per mode) vs N_REF, log-log, `N_REF^-1/2` guide.
3. `grad_se_vs_nseeds` — s.e. of the n-seed mean gradient vs n (500 random subsets) with the `sd/√n`
   line, plus ‖ĝ_n‖ vs n showing the noise floor `E‖ĝ‖² = ‖g‖² + tr Σ/n`, and cosine(n-seed mean, 16-seed mean).
4. `loss_gnorm_vs_iter_<knob>` — loss (normalised and drag force) and ‖g‖ with `grad_se` dashed vs
   iteration, lines coloured by knob value; one figure per knob (E2–E5).
5. `final_coeffs_vs_<knob>` — `C_eff` at the last iterate per mode vs knob value (padded to the largest
   N for the mode sweep); coefficient trajectories `C_k(it)` for B0.
6. `shapes_<knob>` — initial circle dashed, final effective outlines (`ri.body_outline`) per knob value.
7. `grad_init_frac_effects` — first accepted displacement, backtracks at iteration 0, loss after 1/5/20
   iterations, `iter_reach_floor` vs GRAD_INIT_FRAC.
8. `wall_per_iter` — wall time per iteration vs N_REF and n_seeds (validates the cost model).
9. `success_table.csv` + heatmap of `decayed / plateau / at_noise_floor / status` per run.

## 8. Notebooks (`experiments/simple/`, run with cwd there so `import runInflow` works)

0. `00_bootstrap.ipynb`: `!git pull`, environment cell (cores, RAM, load, Python, numpy/scipy/matplotlib/pandas,
   ffmpeg), warm-cache symlink if an older copy exists, geometry selftest.
1. `01_smoke_test.ipynb` (< 5 min): `python -m sweep.selftest`; smoke config (`LX,LY = 6,4`,
   `N_COLL_CELLS=[48,32]`, `N_REF=20_000`, `T_WARM=5`, `REWARM_T=1`, `TOBS=2`, `N_ITER=3`, `n_seeds=2`,
   `N_FOURIER=3`, group `smoke`); run in-process (exercises `apply_config` restore) and via subprocess;
   assert array shapes, `DONE`, skip on re-launch; `--force` into a second dir → `array_equal` on `loss`
   and `G` (CRN determinism); serial (`n_workers=1`) vs pool (`n_workers=2`) → bitwise-identical
   `history.npz`; smoke gradstat (`N_REF ∈ {10k, 20k}`, 3 seeds) → `G.shape == (3, 7)` and
   `allclose(G.mean(0), g)`; `analysis.plot_run` renders.
2. `02_run_sweeps.ipynb`: environment cell (cores, RAM, Python, ffmpeg, geometry selftest);
   `grids.first_pass()` → `queue.json` with E0 warm jobs first, then E1, B0, E5, E2, E4, E3; start the
   detached launcher; `launcher.status()` table; `launcher.tail()`; prune cell. Phase-2 replicate grid
   defined but not queued.
3. `03_analysis.ipynb`: `load_runs`, classifier table, figures 1–9, `summary_table.csv`, renders itself to
   `sweep_results/report.html`, and ends with a cell that commits `sweep_results/` and pushes to GitHub.

## 9. Verification

1. `python -m sweep.selftest`: `resolve()`/`run_id()` idempotent and spelling-independent; `apply_config`
   round trip on a fresh `import runInflow` restores every tunable, `C_init`, `N_STEPS`, `M_P`, `N0`;
   inside the context `len(C_init) == 2N+1`, `|M_P·N0 − RHO0| < 1e-12`, `N_STEPS == round(TOBS/DT)`,
   caches empty; classifier on synthetic histories (monotone decay + flat gnorm → success; rising
   loss → diverged).
2. Backward compatibility after the `runInflow.py` edits: `python -c "import runInflow as ri;
   ri.validate_adjoint()"` (cos > 0.999) and the same with a 3-mode `C` (exercises the `len(C)` change);
   `python runInflow.py` path checked with a throw-away tiny override.
3. Smoke notebook passes locally (macOS, 10 cores) and then on the Cornell machine.
4. Launcher: two smoke jobs with a 1 GB budget run serially, with 4 GB concurrently; killing one
   mid-run leaves `progress.json status=killed`, a loadable partial `history.npz`, no `DONE`, and a
   re-launch reruns it. The warm cache file count does not grow on a second run.

## 10. Pitfalls to respect while implementing

- 1-seed runs are a different algorithm (no noise band, `grad_se = inf`): judge them only on
  `decay_rel` and `gnorm_tail_cv`; the E2 n=1 point is exactly what the current production setting does.
- `grad_se` is pooled over the last 3 gradient evaluations (l.585–596) — also store `grad_se_unpooled` from `G`.
- Normalisation is fixed at the first evaluation on the run's own seeds (l.569–584): loss/gnorm are
  comparable within a run; across runs use `objective_raw`, `force`, and the raw-unit gradstat.
- A tiny `GRAD_INIT_FRAC` can trip the `NOISE_SNR` test at iteration 0 (predicted decrease below the
  paired noise) → status `noise-limited` at iterate 0; a large one hits `project_C` (`A_MAX_FRAC·c0`)
  / `is_valid` and backtracks. Both are recorded (`n_trials`, `trial_disp`, `status`), not failures.
- `N_FOURIER = 20`: `is_valid` samples 200 θ points (10 per period of mode 20) — fine; per-component
  noise is mode-independent so `grad_se ∝ √(2N+1)` and the noise floor is reached earlier.
- Shape statistics, overlays and coefficient plots use `C_eff` (what is simulated), not raw `C`.
- The smoke box (6×4) keeps `GAS_AREA_REF` of the 16×8 reference — same convention as
  `validate_adjoint`; fine for harness tests, not physics.

## Out of scope (user's call)
- Single-crossing assumption detector; `C_raw`/`C_eff` per iterate are saved so it can be added later.
- Speeding up the adjoint loop; the pathwise-vs-expectation gradient bias (`check_expectation_gradient`).
