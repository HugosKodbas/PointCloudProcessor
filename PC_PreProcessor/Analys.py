import numpy as np
from pathlib import Path

# ---- change this ----
npy_path = Path("Area_5_bathroom.npy")
# ---------------------

data = np.load(npy_path)
coords = data[:, :3].astype(np.float64)

mins = coords.min(axis=0)
maxs = coords.max(axis=0)
mean = coords.mean(axis=0)
median = np.median(coords, axis=0)
std = coords.std(axis=0)
extent = maxs - mins

spans_origin = (mins < 0) & (maxs > 0)
mean_near_origin = np.all(np.abs(mean) < 1e-2)

print(f"File: {npy_path}  shape={data.shape}")
print(f"Min (x,y,z):     {mins}")
print(f"Max (x,y,z):     {maxs}")
print(f"Extent (dx,dy,dz): {extent}")
print(f"Mean (centroid): {mean}")
print(f"Median:          {median}")
print(f"Std dev:         {std}")
print(f"Spans origin?    {spans_origin}   (per axis)")
print(f"Mean near origin? {mean_near_origin}")

# Also show a "centered" version's centroid (after subtracting min or mean)
coords_min_shifted = coords - mins
coords_mean_centered = coords - mean
print("\nIf you subtract min (like your loader does):")
print("  new min:", coords_min_shifted.min(axis=0), "new mean:", coords_min_shifted.mean(axis=0))
print("If you center by mean:")
print("  new mean:", coords_mean_centered.mean(axis=0), "new min:", coords_mean_centered.min(axis=0), "new max:", coords_mean_centered.max(axis=0))
