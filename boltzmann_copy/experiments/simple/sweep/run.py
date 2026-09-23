"""One job per interpreter:  python -m sweep.run --config <dir>/config.json [--out ROOT] [--force]

Skips when <run_dir>/DONE exists (unless --force). Writes log.txt, progress.json, the job's outputs,
summary.json and DONE; on an exception FAILED holds the traceback. SIGTERM marks progress.json 'killed'.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

SIMPLE_DIR = Path(__file__).resolve().parents[1]
if str(SIMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(SIMPLE_DIR))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="results root; default: the config's own directory is the run dir")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        os.nice(5)
    except Exception:
        pass
    import runInflow as ri
    from sweep import config as C
    from sweep.jobs import run_job
    cfg = C.resolve(C.load(args.config), ri)
    run_dir = C.run_dir(args.out, cfg) if args.out else Path(args.config).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    C.save(cfg, run_dir / "config.json")
    if (run_dir / "DONE").exists() and not args.force:
        print(f"skip: DONE exists in {run_dir}")
        return 0
    for m in ("DONE", "FAILED"):
        (run_dir / m).unlink(missing_ok=True)
    log = open(run_dir / "log.txt", "a")
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    prog = run_dir / "progress.json"

    def _mark(status):
        try:
            d = json.loads(prog.read_text()) if prog.exists() else {}
        except Exception:
            d = {}
        d.update(status=status, pid=os.getpid(), last_update=time.strftime("%Y-%m-%d %H:%M:%S"))
        prog.write_text(json.dumps(d, indent=1))

    def _on_term(signum, frame):
        _mark("killed")
        print(f"\n[sweep] received signal {signum}: stopping")
        sys.exit(143)

    signal.signal(signal.SIGTERM, _on_term)
    _mark("running")
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {cfg['kind']} {cfg['group']}/{cfg['name']}  id={cfg['run_id']}  pid={os.getpid()}")
    print(json.dumps({k: cfg["params"][k] for k in ("N_REF", "N_FOURIER", "N_AVG", "N_AVG_MAX", "SEED0", "GRAD_INIT_FRAC", "N_ITER", "TOBS", "T_WARM", "LX", "LY")}))
    t0 = time.time()
    try:
        run_job(cfg, run_dir, ri)
    except SystemExit:
        raise
    except BaseException:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        _mark("failed")
        traceback.print_exc()
        return 1
    _mark("done")
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"=== done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
