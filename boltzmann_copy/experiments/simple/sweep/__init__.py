"""Parameter-study harness for experiments/simple/runInflow.py (see SWEEP_PLAN.md at the repo root).

Modules: config (schema, ids), apply (set runInflow globals safely), pool (per-seed worker processes),
recorder (records every evaluate call -> history.npz), jobs (optimise / gradstat / warm), run (CLI),
launcher (detached scheduler + status), classify (success rules), grids (sweep definitions),
analysis (load, aggregate, figures), selftest (quick checks)."""
HARNESS_VERSION = "2026-09-22a"
