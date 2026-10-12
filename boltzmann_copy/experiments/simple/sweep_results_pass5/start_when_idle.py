"""Start the pass-5 queue once the machine is mostly idle: fewer than 3 busy cores (from /proc/stat, all users)
in each of 10 consecutive one-minute samples. Run from experiments/simple."""
import sys, time
sys.path.insert(0, ".")
from sweep import launcher

def busy_cores(dt=60.0):
    def read():
        f = [float(x) for x in open("/proc/stat").readline().split()[1:]]
        idle = f[3] + f[4]
        return sum(f) - idle, sum(f)
    b0, t0 = read(); time.sleep(dt); b1, t1 = read()
    import os
    return (b1 - b0) / (t1 - t0) * os.cpu_count()

quiet = 0
while quiet < 10:
    b = busy_cores()
    quiet = quiet + 1 if b < 3.0 else 0
    print(time.strftime("%H:%M:%S"), f"busy cores {b:.1f}  quiet minutes {quiet}", flush=True)
print("pass-5 scheduler pid", launcher.start_detached("sweep_results_pass5/queue_pass5.json", max_cores=16, mem_budget_gb=60), flush=True)
