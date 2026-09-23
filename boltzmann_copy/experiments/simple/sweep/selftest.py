"""Quick unit checks (< 5 s, no simulation):  python -m sweep.selftest"""
import sys
from pathlib import Path

import numpy as np

SIMPLE_DIR = Path(__file__).resolve().parents[1]
if str(SIMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(SIMPLE_DIR))


def main():
    import runInflow as ri
    from sweep import config as C, grids
    from sweep.apply import apply_config
    from sweep.classify import classify_run
    # 1. resolve / run_id idempotent and spelling-independent
    a = C.resolve(grids.optimise_cfg("t", "a", N_REF=100_000), ri)
    b = C.resolve(dict(kind="optimise", group="t", name="b", params=dict(a["params"])), ri)
    assert a["run_id"] == b["run_id"] == C.resolve(a, ri)["run_id"], "run_id must depend only on the resolved params"
    c = C.resolve(grids.optimise_cfg("t", "c", N_REF=100_000, SEED0=100), ri)
    assert c["run_id"] != a["run_id"], "seed offset must change the id"
    # 2. apply_config round trip
    before = {k: getattr(ri, k) for k in C.TUNABLES + ["C_init", "N_STEPS", "M_P", "N0"]}
    with apply_config(ri, grids.smoke_optimise()["params"] | {"N_REF": 20_000, "CAP": 30_000, "TOBS": 2.0, "DT": 0.1, "N_FOURIER": 3}):
        assert len(ri.C_init) == 7 and ri.C_init[0] == 1.0
        assert abs(ri.M_P * ri.N0 - ri.RHO0) < 1e-12
        assert ri.N_STEPS == 20 and ri.N_REF == 20_000 and ri.CAP == 30_000
        assert ri.N_AVG == ri.N_AVG_MAX == 2 and not ri.ADAPTIVE_SEEDS
        assert not ri._WARM_CACHE and not ri._GAS_FRAC_CACHE
        assert abs(ri._L0_VALUE - ri.perimeter(ri.C_init)) < 1e-12
    after = {k: getattr(ri, k) for k in before}
    for k in before:
        assert np.array_equal(np.asarray(before[k]), np.asarray(after[k])) if hasattr(before[k], "__len__") and not isinstance(before[k], str) else before[k] == after[k], f"{k} not restored"
    # 3. classifier on synthetic histories
    n, P = 21, 7
    it = np.arange(n)
    good = dict(loss=1.0 * np.exp(-0.2 * it) + 0.4, gnorm=np.r_[np.linspace(1, 0.3, 8), 0.3 + 0.02 * np.random.default_rng(0).standard_normal(n - 8)],
                n_seeds=np.full(n, 4), losses_seed=np.column_stack([1.0 * np.exp(-0.2 * it) + 0.4 + 0.01 * j for j in range(4)]),
                loss_se=np.full(n, 0.005), grad_signif=np.r_[np.full(8, 5.0), np.full(n - 8, 1.2)], status=np.array("max iterations"),
                n_iter_requested=np.array(20), force=np.ones(n), disp=np.r_[np.nan, np.full(n - 1, 0.05)], n_trials=np.ones(n, int), wall_iter=np.ones(n))
    assert classify_run(good)["verdict"] == "success", classify_run(good)
    bad = dict(good); bad["loss"] = 1.0 + 0.05 * it; bad["losses_seed"] = np.column_stack([1.0 + 0.05 * it + 0.01 * j for j in range(4)])
    assert classify_run(bad)["verdict"] == "diverged", classify_run(bad)
    print("sweep selftest: OK")


if __name__ == "__main__":
    main()
