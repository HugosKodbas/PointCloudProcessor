# Floorplan generator for 2D floorplan generation from point clouds
# Requires processed point clouds with predicted semantic labels (e.g. from PointNeXt)
# Author: Hugo Nilsson
# Date: 2025-02-09

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation, binary_closing
from skimage.morphology import remove_small_objects

# Floorplan generator for 2D floorplan generation from point clouds
# Requires processed point clouds with predicted semantic labels (e.g. from PointNeXt)
# Author: Hugo Nilsson
# Date: 2025-02-09

wall_csv   = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_wall_coords_cloud0.csv"
door_csv   = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_door_coords_cloud0.csv"
window_csv = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_window_coords_cloud0.csv"

def load_xy(csv_path: str):
    """Load a CSV and return (x, y) as numpy arrays. Expects columns named x,y (case-insensitive)."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    x_col = cols.get("x")
    y_col = cols.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"{csv_path} must contain 'x' and 'y' columns. Found: {list(df.columns)}")
    return df[x_col].to_numpy(), df[y_col].to_numpy()

def compute_bounds(*xy_arrays, pad=0.25):
    """Compute common XY bounds for all point sets. pad in same units as coordinates."""
    xs = np.concatenate([a[0] for a in xy_arrays if a[0].size > 0])
    ys = np.concatenate([a[1] for a in xy_arrays if a[1].size > 0])
    xmin, xmax = xs.min() - pad, xs.max() + pad
    ymin, ymax = ys.min() - pad, ys.max() + pad
    print(f"Bounds: X [{xmin:.2f}, {xmax:.2f}], Y [{ymin:.2f}, {ymax:.2f}] with padding {pad}")
    return xmin, xmax, ymin, ymax

def rasterize_points(x, y, xmin, ymin, cell_size, H, W):
    """Convert points to a boolean grid mask."""
    if x.size == 0:
        return np.zeros((H, W), dtype=bool)

    ix = np.floor((x - xmin) / cell_size).astype(int)
    iy = np.floor((y - ymin) / cell_size).astype(int)

    # clip to image bounds
    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy = ix[valid], iy[valid]

    mask = np.zeros((H, W), dtype=bool)
    mask[iy, ix] = True
    return mask

def main():
    # --- Load data ---
    wx, wy = load_xy(wall_csv)
    dx, dy = load_xy(door_csv)
    vx, vy = load_xy(window_csv)

    # --- Settings ---
    cell_size_m = 0.05  # 5 cm per pixel (use 0.10 if sparse)
    pad_m = 0.25        # padding around the scene

    # wall cleanup
    min_wall_blob_px = 30    # remove tiny speckles
    wall_dilate_r = 1        # pixels
    wall_close_r = 3         # pixels

    # openings carving
    door_open_r = 6          # pixels  (~30cm at 5cm/px)
    window_open_r = 4        # pixels  (~20cm at 5cm/px)

    # --- Create common grid extents ---
    xmin, xmax, ymin, ymax = compute_bounds((wx, wy), (dx, dy), (vx, vy), pad=pad_m)
    W = int(np.ceil((xmax - xmin) / cell_size_m)) + 1
    H = int(np.ceil((ymax - ymin) / cell_size_m)) + 1

    # --- Rasterize ---
    Wmask = rasterize_points(wx, wy, xmin, ymin, cell_size_m, H, W)
    Dmask = rasterize_points(dx, dy, xmin, ymin, cell_size_m, H, W)
    Vmask = rasterize_points(vx, vy, xmin, ymin, cell_size_m, H, W)

    # --- Clean walls ---
    # remove tiny components first (helps a lot with noisy predictions)
    Wclean = remove_small_objects(Wmask, min_size=min_wall_blob_px)

    # thicken and connect gaps
    if wall_dilate_r > 0:
        Wclean = binary_dilation(Wclean, iterations=wall_dilate_r)
    if wall_close_r > 0:
        Wclean = binary_closing(Wclean, iterations=wall_close_r)

    # --- Carve openings ---
    Dopen = binary_dilation(Dmask, iterations=door_open_r) if door_open_r > 0 else Dmask
    Vopen = binary_dilation(Vmask, iterations=window_open_r) if window_open_r > 0 else Vmask
    openings = Dopen | Vopen

    Wopened = Wclean & (~openings)

    # --- Save + show a simple floor plan ---
    # Create an RGB image for visualization:
    # walls = black, background = white, doors = red overlay, windows = blue overlay (purely for debugging)
    img = np.ones((H, W, 3), dtype=np.float32)
    img[Wopened] = 0.0

    # overlays (debug)
    img[Dmask] = [1.0, 0.2, 0.2]
    img[Vmask] = [0.2, 0.4, 1.0]

    # flip vertically for nicer display (so Y increases upward visually)
    img_vis = np.flipud(img)

    out_png = "floorplan_debug.png"
    plt.figure(figsize=(10, 10))
    plt.imshow(img_vis)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.show()

    print(f"Saved: {out_png}")
    print(f"Grid size: {W} x {H} at {cell_size_m} m/px")

if __name__ == "__main__":
    main()
