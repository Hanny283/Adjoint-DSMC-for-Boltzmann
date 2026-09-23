"""SeedPool: per-seed simulations, rewarm and gradients in worker processes.

Workers are stateless. The main process owns every seed's moving warm state (x, v, alive, pid, counter)
and its step label, hands them to the worker for each call and takes the updated state back after a
rewarm. Every seed is an independent computation, so results are bitwise identical to the serial path
in runInflow.make_evaluate_adjoint. Workers pin BLAS to one thread (determinism, no oversubscription).
"""
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

_RI = None


def _pin_threads():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def _init_worker(params, simple_dir):
    global _RI
    _pin_threads()
    os.environ.setdefault("MPLBACKEND", "Agg")
    if simple_dir not in sys.path:
        sys.path.insert(0, simple_dir)
    import runInflow as ri
    from sweep.apply import apply_params
    apply_params(ri, params)
    _RI = ri


def _install(seed, state, step):
    if state is not None:
        _RI._WARM_CACHE[_RI._warm_key(seed)] = state
        _RI._WARM_STEP[seed] = step


def _w_warm(seed):
    st = _RI.warm_state(seed)                      # builds (T_WARM at C_init) or loads the file cache
    return seed, st, _RI._WARM_STEP[seed]


def _w_rewarm(seed, state, step, C_sim):
    _install(seed, state, step)
    _RI.rewarm(seed, C_sim)                          # in place on the installed state
    return seed, _RI._WARM_CACHE[_RI._warm_key(seed)], _RI._WARM_STEP[seed]


def _w_grad(seed, state, step, C, C_sim):
    _install(seed, state, step)
    return seed, _RI.per_seed_eval(C, C_sim, seed)


def _w_loss(seed, state, step, C_sim):
    _install(seed, state, step)
    return seed, _RI.per_seed_loss(C_sim, seed)


class SeedPool:
    def __init__(self, params, n_workers, simple_dir, mp_context="spawn"):
        self.params = dict(params)
        self.n_workers = max(1, int(n_workers))
        self.states = {}                              # seed -> (state tuple, step label)
        self._ex = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=mp.get_context(mp_context),
                                       initializer=_init_worker, initargs=(self.params, str(simple_dir)))

    # -- helpers -------------------------------------------------------------------------------
    def _run(self, fn, seeds, *args):
        futs = [self._ex.submit(fn, s, *self.states.get(s, (None, None)), *args) for s in seeds]
        return [f.result() for f in futs]             # in seed order; exceptions propagate

    def warm(self, seeds):
        missing = [s for s in seeds if s not in self.states]
        if missing:
            for seed, st, step in [f.result() for f in [self._ex.submit(_w_warm, s) for s in missing]]:
                self.states[seed] = (st, step)

    # -- API used by runInflow ---------------------------------------------------------------
    def rewarm(self, C_sim, seeds):
        seeds = list(seeds)
        self.warm(seeds)
        for seed, st, step in self._run(_w_rewarm, seeds, C_sim):
            self.states[seed] = (st, step)

    def grads(self, C, C_sim, seeds):
        seeds = list(seeds)
        self.warm(seeds)
        return [r for _, r in self._run(_w_grad, seeds, C, C_sim)]

    def losses(self, C_sim, seeds):
        seeds = list(seeds)
        self.warm(seeds)
        return [r for _, r in self._run(_w_loss, seeds, C_sim)]

    def close(self):
        self._ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
