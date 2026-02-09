import argparse
from pathlib import Path
import scipy.spatial as spatial  
import numpy as np

DESTINATION_FOLDER = Path("./ply_files")
NPY_OUTPUT_PATH = Path("./npy_files")

RADIUS_THRESHOLD = 0.05  # Meters
MIN_NEIGHBORS = 20

""" def extents_print(xyz, tag=""):
    pts = np.asarray(xyz)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ext = maxs - mins
    print(f"{tag} mins:", mins)
    print(f"{tag} maxs:", maxs)
    print(f"{tag} extents (x,y,z):", ext) """

# Rotation matrices

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

# REQUIRES TESTING! I DONT KNOW HOW THIS BEHAVES WITH LARGE SCANS CONTAINING MULTIPLE ROOMS AND HALLWAYS

def estimate_yaw_pca_xy(xyz: np.ndarray) -> float:
    """
    Estimate the dominant yaw angle (radians) in the XY plane using PCA.
    Returns angle in radians of the first principal direction
    """
    xy = xyz[:, :2].astype(np.float64)
    xy -= xy.mean(axis=0, keepdims=True)

    # Covariance and eigenvectors
    cov = (xy.T @ xy) / max(len(xy) - 1, 1) # ([var_xx, cov_xy], [cov_yx, var_yy]) Which tells us the directions of variance
    # eigenvectors = orthogonal directions in XY
    # eigenvalues  = variance (spread) along each eigenvector direction
    # For reference: https://en.wikipedia.org/wiki/Principal_component_analysis#Computation
    # Largest eigenvalue typically corresponds to the largest direction: giving us the correct eigenvector.
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]  # Find the largest eigenvectors eigenvalue, that we know which wall is longest.
    angle = np.arctan2(v[1], v[0])  # v[1] = y, v[0] = x, so this gives angle in XY plane
    return angle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("google_drive_link", help="Link to Google Drive location for .ply-file.")
    parser.add_argument("output_filename", help="Desired base filename (no extension needed).")
    parser.add_argument("--label", type=int, default=0, help="Dummy label to fill (default: 0)")
    parser.add_argument("--downsample_voxel", type=float, default=0.01, help="Voxel size in meters. 0 disables.")
    parser.add_argument("--make_z_up", action="store_true", help="Rotate +90° around X (Y-up -> Z-up)")
    parser.add_argument("--align_xy", action="store_true", help="Rotate around Z to align dominant direction with X-axis") 
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

    # ---- Load ----
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    # --- Print extents before any processing, to help with tuning ---
    #extents_print(pcd, tag="Pre-Processing")

    if len(pcd.points) == 0:
        raise SystemExit("Loaded 0 points from PLY. The downloaded file may be HTML or invalid PLY.")

    # Checks the format of the downloaded file to ensure it's a PLY. Had some problems with HTML files.
    with open(ply_path, "rb") as f:
        head = f.read(200).lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise SystemExit("Downloaded HTML instead of a PLY. Make the Drive file public (Anyone with link).")

    # ---- Rotate (apply to pcd) ----
    if args.make_z_up:
        R = rot_x_deg(+90.0)
        # Move center of rotation to center of point cloud, with center being the Z-axis
        center = pcd.get_axis_aligned_bounding_box().get_center()
        pcd.rotate(R, center=center)

    # ---- Downsample ----
    if args.downsample_voxel and args.downsample_voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.downsample_voxel))

    ### Filtration logic
    xyz = np.asarray(pcd.points, dtype=np.float32)

    # Step 1: Radius outlier removal

    KDtree = spatial.KDTree(xyz, leafsize=MIN_NEIGHBORS)

    to_keep = np.zeros(xyz.shape[0], dtype=bool)

    # Remove all points that have fewer than MIN_NEIGHBORS within RADIUS_THRESHOLD
    for idx, point in enumerate(xyz):
        indices = KDtree.query_ball_point(point, r=RADIUS_THRESHOLD)
        if len(indices) >= MIN_NEIGHBORS:
            to_keep[idx] = True

    xyz_filtered = xyz[to_keep]

    # --- Rotate again to align with dominant direction ----

    if args.align_xy:
        print("Aligning point cloud to dominant XY direction...")
        xyz_tmp = xyz_filtered.astype(np.float64)

        # Ignore ceilings and floors, walls determine direction.
        z = xyz_tmp[:, 2]
        zmin, zmax = float(np.min(z)), float(np.max(z))
        
        # Filter out a band in the middle 60% of Z values to focus on walls, which typically determine the dominant direction. This helps avoid ceilings/floors dominating        
        low_z = zmin + 0.2 * (zmax - zmin)
        high_z = zmax - 0.2 * (zmax - zmin)
        mask = (z >= low_z) & (z <= high_z)
        xyz_for_yaw = xyz_tmp[mask] if mask.sum() > 1000 else xyz_tmp

        yaw = estimate_yaw_pca_xy(xyz_for_yaw)
        # Don't orbit global Origin, but rather the center of the point cloud.
        center = xyz_filtered.mean(axis=0)

        Rz = rot_z_rad(-yaw)  # Rotate by negative yaw to align with
        xyz_filtered = (xyz_filtered - center) @ Rz.T + center

        print("Successful rotation!")

    # ---- Colors ----
    if pcd.has_colors():
        # Filter colors to match filtered points
        colors = np.asarray(pcd.colors, dtype=np.float32)
        colors_filtered = colors[to_keep]
        rgb01 = colors_filtered
        rgb255 = (rgb01 * 255.0).clip(0, 255).astype(np.float32)
    else:
        rgb255 = np.zeros((xyz_filtered.shape[0], 3), dtype=np.float32)
    label = np.full((xyz_filtered.shape[0], 1), float(args.label), dtype=np.float32)

    out_arr = np.concatenate([xyz_filtered, rgb255, label], axis=1)  # (N,7)
    print("Successfully converted and filtered point cloud.")

    # Prints for tuning
    #extents_print(xyz_filtered, tag="Post-Processing")
    print("Before filter:", xyz.shape[0])
    print("After filter :", xyz_filtered.shape[0])
    print("Kept %       :", 100.0 * xyz_filtered.shape[0] / xyz.shape[0])

    npy_path = NPY_OUTPUT_PATH / f"Area_5_{name}.npy"
    np.save(str(npy_path), out_arr)
    print(f"Saved {npy_path} with shape {out_arr.shape}")

if __name__ == "__main__":
    main()
