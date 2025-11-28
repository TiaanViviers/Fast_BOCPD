"""
Example: Using OnlineChangeDetector for streaming data
"""
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard, OnlineChangeDetector
import numpy as np

# Generate synthetic data with changepoints
np.random.seed(42)
data = np.concatenate([
    np.random.randn(50) * 0.5 + 0.0,   # Segment 1
    np.random.randn(50) * 0.5 + 5.0,   # Segment 2
    np.random.randn(50) * 0.5 + -3.0,  # Segment 3
])

# Setup detector
bocpd = BOCPD(
    obs_model=GaussianNIG(mu0=0, kappa0=1, alpha0=1, beta0=1),
    hazard=ConstantHazard(lambda_=50),
    max_run_length=100
)
detector = OnlineChangeDetector(bocpd, min_confidence=0.3)

# Process stream
print("Processing data stream...")
print("=" * 60)

for t, x in enumerate(data):
    cp = detector.update(x, metadata=f"sample_{t}")
    
    if cp:
        print(f"\n🔔 {cp}")
    
    # Show current run length every 10 steps
    if t % 10 == 0:
        run_length = detector.get_current_run_length()
        print(f"t={t:3d}: run_length={run_length:2d}, x={x:6.2f}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

changepoints = detector.get_changepoints()
print(f"\nDetected {len(changepoints)} changepoints:")
for cp in changepoints:
    print(f"  {cp}")

segments = detector.get_segments()
print(f"\nSegments:")
for i, (start, end) in enumerate(segments):
    print(f"  Segment {i+1}: [{start:3d}, {end:3d}) - length: {end-start}")
