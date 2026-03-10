import argparse
from pathlib import Path
import numpy as np
import scipy.spatial as spatial

DESTINATION_FOLDER = Path("./ply_files")
NPY_OUTPUT_PATH = Path("./npy_files")

RADIUS_THRESHOLD = 0.05  # meters
MIN_NEIGHBORS = 20


def extents_print(xyz, tag=""):
    pts = np.asarray(xyz)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ext = maxs - mins
    print(f"{tag} mins:", mins)
    print(f"{tag} maxs:", maxs)
    print(f"{tag} extents (x,y,z):", ext)


def rot_x_deg(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float64)  # Open3D likes float64


def rot_z_rad(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ], dtype=np.float64)


def estimate_yaw_manhattan(xy, k=15, linearity_threshold=0.7, n_subsample=80000, seed=42):
    """
    Estimate the yaw correction angle by computing KNN local normals
    and finding the dominant wall direction via a Manhattan-folded histogram.

    Why this works better than PCA:
      - PCA finds the direction of maximum *variance* across the entire cloud.
        If point density is uneven (furniture, floors, one long wall), PCA is
        biased toward the dense cluster, not the wall directions.
      - This method computes a *local* normal at each point, keeps only
        points with high linearity (= wall-like), and histograms their
        angles.  The peak of the histogram is the true wall orientation.

    Steps:
      1. Subsample for speed, build KD-tree.
      2. For each point, PCA on its k nearest neighbours to get a local
         2D normal direction and a linearity score.
      3. Keep only points whose linearity exceeds a threshold (wall-like).
      4. Fold normal angles into [0°, 90°) (Manhattan assumption: walls
         are at 0° or 90°).
      5. Smooth the histogram and return the peak as the correction angle.

    Returns
    -------
    correction_rad : float
        Angle in radians to rotate the cloud so walls align with X/Y.
    peak_deg : float
        The detected wall-normal offset in degrees (informational).
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(seed)
    n = min(n_subsample, len(xy))
    pts = xy[rng.choice(len(xy), n, replace=False)].astype(np.float64)

    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=k)

    angles = np.empty(n, dtype=np.float64)
    linearity = np.empty(n, dtype=np.float64)

    for i in range(n):
        neighbors = pts[nn_idx[i]]
        nb_c = neighbors - neighbors.mean(axis=0)
        cov = nb_c.T @ nb_c
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Normal = minor eigenvector (smallest eigenvalue)
        angles[i] = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        # Linearity: how "line-like" the neighbourhood is
        ev = np.sort(eigvals)[::-1]
        linearity[i] = (ev[0] - ev[1]) / (ev[0] + 1e-12)

    # Keep only wall-like points
    mask = linearity > linearity_threshold
    if mask.sum() < 50:
        mask = linearity > np.percentile(linearity, 80)

    wall_angles = angles[mask] % np.pi           # fold into [0, pi)
    wall_angles_folded = np.degrees(wall_angles) % 90  # fold into [0, 90°)

    hist, edges = np.histogram(wall_angles_folded, bins=180, range=(0, 90))
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2.5)

    peak_deg = float(edges[np.argmax(hist_smooth)] + 0.25)
    correction_rad = np.radians(-peak_deg)

    return correction_rad, peak_deg


def dbscan_keep_largest_clusters(xyz: np.ndarray, eps: float, min_points: int, topk: int = 1):
    """
    Run DBSCAN and keep the largest cluster(s).
    Returns: keep_mask (N,), labels (N,), kept_cluster_ids (list)
    """
    import open3d as o3d

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    labels = np.array(
        p.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=True),
        dtype=np.int32
    )

    valid = labels >= 0
    if not np.any(valid):
        # No clusters found; keep everything (safer than deleting all)
        keep = np.ones(len(xyz), dtype=bool)
        return keep, labels, []

    cluster_ids, counts = np.unique(labels[valid], return_counts=True)
    order = np.argsort(counts)[::-1]
    kept = cluster_ids[order[:max(1, int(topk))]]

    keep_mask = np.isin(labels, kept)
    return keep_mask, labels, kept.tolist()


def plot_alignment(xyz_before, xyz_after, colors=None, max_points=200_000):
    """
    Side-by-side top-down XY scatter plot of the point cloud
    before and after alignment + shift.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = min(max_points, len(xyz_before))
    idx = rng.choice(len(xyz_before), n, replace=False)

    # Build color array
    if colors is not None:
        c = np.asarray(colors[idx], dtype=np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        c = c.clip(0, 1)
    else:
        c = "gray"

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Before ----
    ax_l.scatter(xyz_before[idx, 0], xyz_before[idx, 1], c=c, s=0.15, alpha=0.4)
    ax_l.axhline(0, color="red", lw=0.8, ls="--", alpha=0.6)
    ax_l.axvline(0, color="red", lw=0.8, ls="--", alpha=0.6)
    ax_l.set_aspect("equal")
    ax_l.set_title("Before alignment (filtered)")
    ax_l.set_xlabel("X (m)")
    ax_l.set_ylabel("Y (m)")

    # ---- After ----
    ax_r.scatter(xyz_after[idx, 0], xyz_after[idx, 1], c=c, s=0.15, alpha=0.4)
    ax_r.axhline(0, color="red", lw=0.8, ls="-", alpha=0.7, label="X = 0 / Y = 0")
    ax_r.axvline(0, color="red", lw=0.8, ls="-", alpha=0.7)
    ax_r.set_aspect("equal")
    ax_r.set_title("After alignment + XY shift")
    ax_r.set_xlabel("X (m)")
    ax_r.set_ylabel("Y (m)")
    ax_r.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("google_drive_link", help="Link to Google Drive location for .ply-file.")
    parser.add_argument("output_filename", help="Desired base filename (no extension needed).")

    parser.add_argument("--label", type=int, default=0, help="Dummy label to fill (default: 0)")
    parser.add_argument("--downsample_voxel", type=float, default=0.01, help="Voxel size in meters. 0 disables.")
    parser.add_argument("--make_z_up", action="store_true", help="Rotate +90° around X (Y-up -> Z-up)")
    parser.add_argument("--align_xy", action="store_true", help="Rotate around Z to align dominant direction with X-axis")

    # ROR filter
    parser.add_argument("--filter_ror", action="store_true", help="Run radius outlier removal (KDTree-based)")

    # DBSCAN blob removal
    parser.add_argument("--dbscan", action="store_true", help="Run DBSCAN and keep largest cluster(s)")
    parser.add_argument("--dbscan_eps", type=float, default=0.25, help="DBSCAN eps in meters (cluster radius)")
    parser.add_argument("--dbscan_min_points", type=int, default=30, help="DBSCAN min_points")
    parser.add_argument("--dbscan_keep_topk", type=int, default=1, help="Keep top-K largest clusters (1 keeps main blob)")

    # Plotting
    parser.add_argument("--plot", action="store_true", help="Show before/after alignment plot")

    args = parser.parse_args()

    DESTINATION_FOLDER.mkdir(parents=True, exist_ok=True)
    NPY_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit("Missing dependency: open3d. Install with: pip install open3d") from e

    try:
        import gdown
    except ImportError as e:
        raise SystemExit("Missing dependency: gdown. Install with: pip install gdown") from e

    # ---- Download ----
    name = args.output_filename
    if name.lower().endswith(".npy"):
        name = name[:-4]
    if name.lower().endswith(".ply"):
        name = name[:-4]

    ply_path = DESTINATION_FOLDER / f"Area_5_{name}.ply"

    print(f"Trying to download file from link: {args.google_drive_link}")
    out = gdown.download(url=args.google_drive_link, output=str(ply_path), quiet=False, fuzzy=True)
    if out is None:
        raise SystemExit("gdown download failed (returned None). Check sharing permissions / link type.")
    print(f"Downloaded file to: {ply_path}")

    # Quick HTML check (common Drive issue)
    with open(ply_path, "rb") as f:
        head = f.read(200).lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise SystemExit("Downloaded HTML instead of a PLY. Make the Drive file public (Anyone with link).")

    # ---- Load ----
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise SystemExit("Loaded 0 points from PLY. The downloaded file may be invalid.")

    # Print extents pre-processing
    extents_print(np.asarray(pcd.points), tag="Pre-Processing")

    # ---- Rotate Z-up ----
    if args.make_z_up:
        R = rot_x_deg(+90.0)
        center = pcd.get_axis_aligned_bounding_box().get_center()
        pcd.rotate(R, center=center)

    # ---- Downsample ----
    if args.downsample_voxel and args.downsample_voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.downsample_voxel))

    # ---- Extract XYZ and Colors ----
    xyz = np.asarray(pcd.points, dtype=np.float32)

    if pcd.has_colors():
        # Open3D stores colors as 0..1
        colors = np.asarray(pcd.colors, dtype=np.float32)  # (N,3) in 0..1
    else:
        colors = None

    # ---- Radius Outlier Removal (your KDTree step) ----
    xyz_filtered = xyz
    colors_filtered = colors

    if args.filter_ror:
        print("Applying radius outlier removal...")
        KDtree = spatial.KDTree(xyz, leafsize=MIN_NEIGHBORS)
        to_keep = np.zeros(xyz.shape[0], dtype=bool)

        for idx, point in enumerate(xyz):
            indices = KDtree.query_ball_point(point, r=RADIUS_THRESHOLD)
            if len(indices) >= MIN_NEIGHBORS:
                to_keep[idx] = True

        xyz_filtered = xyz[to_keep]
        colors_filtered = colors[to_keep] if colors is not None else None

        print(f"Radius filter kept {xyz_filtered.shape[0]}/{xyz.shape[0]} points ({100*xyz_filtered.shape[0]/xyz.shape[0]:.2f}%)")

    # ---- DBSCAN Keep Largest Blob(s) ----
    if args.dbscan:
        print("Running DBSCAN to remove big blobs...")
        keep2, labels2, kept_ids = dbscan_keep_largest_clusters(
            xyz_filtered,
            eps=args.dbscan_eps,
            min_points=args.dbscan_min_points,
            topk=args.dbscan_keep_topk
        )
        before = xyz_filtered.shape[0]
        xyz_filtered = xyz_filtered[keep2]
        if colors_filtered is not None:
            colors_filtered = colors_filtered[keep2]

        print(f"DBSCAN kept clusters {kept_ids} -> {xyz_filtered.shape[0]}/{before} points ({100*xyz_filtered.shape[0]/before:.2f}%)")

    # ---- Snapshot for plotting (before alignment) ----
    xyz_before_align = xyz_filtered.copy()

    # ---- Align XY (yaw) on filtered points ----
    if args.align_xy and xyz_filtered.shape[0] > 10:
        print("Aligning point cloud to dominant wall direction (Manhattan histogram)...")
        xyz_tmp = xyz_filtered.astype(np.float64)

        bbox = np.column_stack([xyz_tmp.min(axis=0), xyz_tmp.max(axis=0)])
        center = bbox.mean(axis=1)

        print(f"Rotation center: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")

        # Build a yaw-estimation subset (focus on "wall band")
        z = xyz_tmp[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        low_z  = zmin + 0.2 * (zmax - zmin)
        high_z = zmax - 0.2 * (zmax - zmin)
        mask = (z >= low_z) & (z <= high_z)

        xy_for_yaw = xyz_tmp[mask, :2] if mask.sum() > 1000 else xyz_tmp[:, :2]

        correction_rad, peak_deg = estimate_yaw_manhattan(xy_for_yaw)
        print(f"Detected wall offset: {peak_deg:.1f}° from axis")
        print(f"Applying correction rotation: {np.degrees(correction_rad):.1f}°")

        Rz = rot_z_rad(correction_rad)
        xyz_rot = (xyz_tmp - center) @ Rz.T + center

        xyz_filtered = xyz_rot.astype(np.float32)
        print("Alignment complete.")

    # ---- Shift filtered cloud so all XY coordinates are non-negative ----
    xy_min = xyz_filtered[:, :2].min(axis=0)
    xyz_filtered[:, 0] -= xy_min[0]
    xyz_filtered[:, 1] -= xy_min[1]
    print(f"Shifted filtered cloud so min XY = (0, 0). Translation: ({-xy_min[0]:.3f}, {-xy_min[1]:.3f})m")

    # ---- Plot before/after alignment ----
    if args.plot:
        plot_alignment(xyz_before_align, xyz_filtered, colors_filtered)


    # ---- Convert colors to 0..255 ----
    if colors_filtered is not None:
        rgb255 = (colors_filtered * 255.0).clip(0, 255).astype(np.float32)
    else:
        rgb255 = np.zeros((xyz_filtered.shape[0], 3), dtype=np.float32)

    # ---- Labels and Save ----
    label = np.full((xyz_filtered.shape[0], 1), float(args.label), dtype=np.float32)
    out_arr = np.concatenate([xyz_filtered, rgb255, label], axis=1)  # (N,7)

    # Print extents post-processing
    extents_print(xyz_filtered, tag="Post-Processing")

    npy_path = NPY_OUTPUT_PATH / f"Area_5_{name}.npy"
    np.save(str(npy_path), out_arr)
    print(f"Saved {npy_path} with shape {out_arr.shape}")


if __name__ == "__main__":
    main()