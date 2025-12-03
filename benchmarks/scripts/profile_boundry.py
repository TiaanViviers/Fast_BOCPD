"""
Tool to profile the performance at the boundary between Python and C code
It measures the overhead of the Python loop vs the actual C work done.
"""

import time
import numpy as np
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard, OnlineChangeDetector

data = np.random.randn(10000)

model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
hazard = ConstantHazard(lambda_=150)
bocpd = BOCPD(model, hazard, max_run_length=450)
detector = OnlineChangeDetector(bocpd, min_confidence=0.3)

# Measure just the Python loop overhead
start = time.perf_counter()
for x in data:
    pass  # Empty loop
empty_loop_time = time.perf_counter() - start

# Measure the actual update calls
start = time.perf_counter()
for x in data:
    detector.update(x)
total_time = time.perf_counter() - start

actual_work = total_time - empty_loop_time

print(f"Empty Python loop:     {empty_loop_time:.4f}s ({empty_loop_time/total_time*100:.1f}%)")
print(f"Actual C work:         {actual_work:.4f}s ({actual_work/total_time*100:.1f}%)")
print(f"Total time:            {total_time:.4f}s")
print(f"\nPer-call overhead:     {empty_loop_time/len(data)*1e6:.2f} µs")
print(f"Per-call C work:       {actual_work/len(data)*1e6:.2f} µs")