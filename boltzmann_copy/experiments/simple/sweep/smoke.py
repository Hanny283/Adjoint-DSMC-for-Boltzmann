"""End-to-end smoke test (< 5 min):  python -m sweep.smoke [--root DIR]

Runs a tiny optimisation serially and with a 2-worker pool (must be bitwise identical), checks skip-on-DONE,
runs a tiny gradient-statistics job, then builds the figures and the HTML report from these results.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SIMPLE_DIR = Path(__file__).resolve().parents[1]
if str(SIMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(SIMPLE_DIR))


def _run(cfg_path, force=False):
    cmd = [sys.executable, "-m", "sweep.run", "--config", str(cfg_path)] + (["--force"] if force else [])
    t = time.time()
    r = subprocess.run(cmd, cwd=str(SIMPLE_DIR), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"run failed: {cfg_path}")
    return time.time() - t, r.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SIMPLE_DIR / "sweep_results" / "smoke"))
    args = ap.parse_args(argv)
    import runInflow as ri
    from sweep import config as C, grids
    from sweep.analysis import load_history, make_figures, write_report
    root = Path(args.root)
    dirs = {}
    for name, nw in (("serial", 1), ("pool", 2)):
        cfg = C.resolve(grids.smoke_optimise(name=f"smoke_{name}", n_workers=nw), ri)
        rd = C.run_dir(root, cfg); C.save(cfg, rd / "config.json"); dirs[name] = rd
        dt, out = _run(rd / "config.json", force=True)
        print(f"{name}: {dt:.0f}s  ->  {rd.name}")
    hs, hp = load_history(dirs["serial"]), load_history(dirs["pool"])
    for k in ("loss", "gnorm", "C_raw", "G", "losses_seed", "trial_loss", "force"):
        assert np.array_equal(np.nan_to_num(hs[k]), np.nan_to_num(hp[k])), f"serial vs pool differ in {k}"
    n, P = hs["C_raw"].shape
    assert P == 7 and n >= 2 and hs["G"].shape == (n, 2, P) and hs["losses_seed"].shape == (n, 2), (n, P, hs["G"].shape)
    assert np.allclose(np.nanmean(hs["G"], axis=1), hs["grad"]), "G.mean must equal grad"
    assert (dirs["serial"] / "DONE").exists() and (dirs["serial"] / "summary.json").exists() and (dirs["serial"] / "quicklook.png").exists()
    # skip on DONE
    dt, out = _run(dirs["serial"] / "config.json")
    assert "skip: DONE" in out, out[-500:]
    # determinism: a fresh run of the same config in another dir
    cfg = C.resolve(grids.smoke_optimise(name="smoke_serial_again", n_workers=1), ri)
    rd = C.run_dir(root, cfg); C.save(cfg, rd / "config.json"); _run(rd / "config.json", force=True)
    ha = load_history(rd)
    assert np.array_equal(hs["loss"], ha["loss"]) and np.array_equal(np.nan_to_num(hs["G"]), np.nan_to_num(ha["G"])), "CRN determinism broken"
    # gradstat
    cfg = C.resolve(grids.smoke_gradstat(n_workers=2), ri)
    rd = C.run_dir(root, cfg); C.save(cfg, rd / "config.json"); dt, _ = _run(rd / "config.json", force=True)
    with np.load(rd / "gradstat.npz") as z:
        G, gm = z["G"], z["g_mean"]
    assert G.shape == (3, 7) and np.allclose(G.mean(0), gm), G.shape
    print(f"gradstat: {dt:.0f}s  G.shape={G.shape}")
    s = json.loads((dirs["serial"] / "summary.json").read_text())
    print("classification:", {k: s["classification"][k] for k in ("verdict", "decayed", "plateau", "at_noise_floor", "n_iter_done")})
    made = make_figures(root, baseline=dict(grids.SMOKE, GRAD_INIT_FRAC=0.05))
    rep = write_report(root, title="smoke test")
    print(f"figures: {len(made)} -> {root / 'figures'};  report: {rep}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
