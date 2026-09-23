"""Queue + detached scheduler for sweep jobs, and a status table for notebooks.

  make_queue(configs, root, ri)            -> writes <root>/queue_<stamp>.json (run dirs with config.json)
  start_detached(queue_file, max_cores, mem_budget_gb) -> pid of a scheduler that survives the notebook kernel
  python -m sweep.launcher --queue Q --max-cores 16 --mem-budget-gb 60   (the scheduler itself)
  status(root) -> pandas.DataFrame ;  tail(run_dir, n)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SIMPLE_DIR = Path(__file__).resolve().parents[1]


def _env():
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    try:                                        # macOS
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
        return int(out) / 1e9 * 0.6
    except Exception:
        return float("inf")


def make_queue(configs, root, ri, queue_name=None):
    from . import config as C
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for cfg in configs:
        cfg = C.resolve(cfg, ri)
        seeds = C.seeds_of(cfg) if cfg["kind"] == "optimise" else cfg.get(cfg["kind"], {}).get("seeds", [])
        nw = C.n_workers_of(cfg, seeds)
        rd = C.run_dir(root, cfg)
        C.save(cfg, rd / "config.json")
        entries.append(dict(run_dir=str(rd), kind=cfg["kind"], group=cfg["group"], name=cfg["name"], run_id=cfg["run_id"],
                            cores=nw, mem_gb=round(C.mem_estimate_gb(cfg, nw), 2)))
    qf = root / (queue_name or f"queue_{time.strftime('%Y%m%d_%H%M%S')}.json")
    qf.write_text(json.dumps(entries, indent=1))
    return qf


def _start_job(entry):
    rd = Path(entry["run_dir"])
    log = open(rd / "log.txt", "a")
    return subprocess.Popen([sys.executable, "-m", "sweep.run", "--config", str(rd / "config.json")],
                            cwd=str(SIMPLE_DIR), stdout=log, stderr=subprocess.STDOUT, env=_env(), start_new_session=True)


def schedule(queue_file, max_cores, mem_budget_gb, poll=20, respect_free_mem=True):
    queue_file = Path(queue_file)
    queue = json.loads(queue_file.read_text())
    state_file = queue_file.with_suffix(".state.json")
    pending = [e for e in queue if not (Path(e["run_dir"]) / "DONE").exists()]
    running = {}
    print(f"[launcher] {len(pending)} pending of {len(queue)} ({len(queue) - len(pending)} already DONE); max_cores={max_cores} mem_budget={mem_budget_gb} GB")

    def _save():
        state_file.write_text(json.dumps(dict(pid=os.getpid(), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                              running=[dict(run_dir=k, pid=v[0].pid) for k, v in running.items()],
                                              pending=[e["run_dir"] for e in pending]), indent=1))
    while pending or running:
        for rd in list(running):
            p, e = running[rd]
            if p.poll() is not None:
                print(f"[launcher] finished ({p.returncode}): {e['group']}/{e['name']}")
                del running[rd]
        cores_used = sum(e["cores"] for _, e in running.values())
        mem_used = sum(e["mem_gb"] for _, e in running.values())
        started = False
        for e in list(pending):
            if (Path(e["run_dir"]) / "DONE").exists():
                pending.remove(e)
                continue
            fits_cores = cores_used + e["cores"] <= max_cores or not running
            fits_mem = mem_used + e["mem_gb"] <= mem_budget_gb or not running
            fits_free = (not respect_free_mem) or (mem_available_gb() >= e["mem_gb"] + 2) or not running
            if fits_cores and fits_mem and fits_free:
                running[e["run_dir"]] = (_start_job(e), e)
                pending.remove(e)
                cores_used += e["cores"]
                mem_used += e["mem_gb"]
                started = True
                print(f"[launcher] started {e['group']}/{e['name']} (cores={e['cores']}, ~{e['mem_gb']} GB); running={len(running)} pending={len(pending)}")
            else:
                break                               # keep the queue order (cheap jobs first)
        _save()
        if not started:
            time.sleep(poll)
    print("[launcher] queue finished")
    _save()


def start_detached(queue_file, max_cores, mem_budget_gb, poll=20):
    """Start the scheduler as its own session so it survives the notebook kernel; returns the pid."""
    queue_file = Path(queue_file)
    log = open(queue_file.with_suffix(".launcher.log"), "a")
    p = subprocess.Popen([sys.executable, "-m", "sweep.launcher", "--queue", str(queue_file), "--max-cores", str(max_cores),
                          "--mem-budget-gb", str(mem_budget_gb), "--poll", str(poll)],
                         cwd=str(SIMPLE_DIR), stdout=log, stderr=subprocess.STDOUT, env=_env(), start_new_session=True)
    return p.pid


def status(root):
    """One row per run dir under root (has config.json): kind, group, name, state, progress, timing."""
    import pandas as pd
    rows = []
    for cfgp in sorted(Path(root).rglob("config.json")):
        rd = cfgp.parent
        try:
            cfg = json.loads(cfgp.read_text())
        except Exception:
            continue
        prog = {}
        if (rd / "progress.json").exists():
            try:
                prog = json.loads((rd / "progress.json").read_text())
            except Exception:
                prog = {}
        if (rd / "DONE").exists():
            state = "done"
        elif (rd / "FAILED").exists():
            state = "failed"
        elif prog.get("status") in ("running", "killed"):
            state = prog["status"]
            if state == "running" and prog.get("pid"):
                try:
                    os.kill(int(prog["pid"]), 0)
                except Exception:
                    state = "dead?"
        else:
            state = "queued"
        p = cfg.get("params", {})
        rows.append(dict(group=cfg.get("group"), name=cfg.get("name"), kind=cfg.get("kind"), state=state,
                         N_REF=p.get("N_REF"), n_seeds=p.get("N_AVG"), N_FOURIER=p.get("N_FOURIER"), frac=p.get("GRAD_INIT_FRAC"),
                         iter=prog.get("iter"), n_iter=prog.get("n_iter"), loss=prog.get("loss"), gnorm=prog.get("gnorm"),
                         elapsed_min=(round(prog["elapsed_s"] / 60, 1) if prog.get("elapsed_s") else None),
                         eta_min=(round(prog["eta_s"] / 60, 1) if prog.get("eta_s") else None),
                         updated=prog.get("last_update"), run_dir=str(rd)))
    return pd.DataFrame(rows)


def tail(run_dir, n=20):
    p = Path(run_dir) / "log.txt"
    if not p.exists():
        return "(no log yet)"
    lines = p.read_text().splitlines()
    return "\n".join(lines[-n:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--max-cores", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--mem-budget-gb", type=float, default=mem_available_gb() * 0.8)
    ap.add_argument("--poll", type=float, default=20)
    args = ap.parse_args(argv)
    schedule(args.queue, args.max_cores, args.mem_budget_gb, poll=args.poll)


if __name__ == "__main__":
    main()
