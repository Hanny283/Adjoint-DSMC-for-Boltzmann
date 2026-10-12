"""Build and start the final pass-3 block once the main pass-3 queue has finished (run from experiments/simple).
Reads the finished growth runs, forms their consensus C3, and queues probes, the C3 high-mode test and the
held-out evaluations (sweep.grids.pass3_final)."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
import runInflow as ri
from sweep import grids, launcher
T_EVAL = 20.0
root = Path("sweep_results_pass3_diag")
sh = json.loads((root / "shapes.json").read_text())
finals = {}
for p in sorted(glob.glob("sweep_results_pass3/adaptive/*/summary.json")):
    s = json.load(open(p))
    if (Path(p).parent / "DONE").exists():
        finals[f"p3_{s['name']}"] = s["C_opt_eff"]
if not finals:
    raise SystemExit("no finished pass-3 growth runs")
C3 = np.mean([np.asarray(c) for c in finals.values()], axis=0).tolist()
sh["C3"] = C3
sh["C3_note"] = f"mean final C_eff of the finished pass-3 growth runs: {sorted(finals)}"
(root / "shapes.json").write_text(json.dumps(sh))
q = grids.pass3_final(finals, C3, sh["probes"], rewarm_t=T_EVAL)
qf = launcher.make_queue(q, str(root), ri, queue_name="queue_final.json")
print(f"{len(q)} jobs; finals: {sorted(finals)}", flush=True)
print("final-block scheduler pid", launcher.start_detached(qf, max_cores=16, mem_budget_gb=60), flush=True)
