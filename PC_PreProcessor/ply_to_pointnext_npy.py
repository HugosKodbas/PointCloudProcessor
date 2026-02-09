# Author: Hugo Nilsson
# Date: 2025-02-01
# Description: Convert .ply point cloud to .npy format compatible with PointNeXt.
# Is able to rotate the point cloud around the X axis to convert from Y-up to Z-up coordinate frame, which is
# common in 3D vision datasets. (atleast for usage with PointNeXt)
# Usage: python ply_to_pointnext_npy.py <INPUT_SCAN.ply> <OUTPUT_SCAN> --label(optional) --downsample_voxel(optional) --y_up_to_z_up(optional)
# Use --scale 0.001 if your scan is in mm and you want meters.
# Use downsample_voxel to downsample the point cloud by a voxel grid filter before saving.
# Use --y_up_to_z_up to rotate the point cloud +90° around X axis, if Y is up in the input.
# Recommended: RUN FROM TERMINAL!

import argparse
import numpy as np

def rot_x_deg(deg: float) -> np.ndarray:
    """Rotation matrix for rotation about X axis by deg degrees."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", help="Input .ply point cloud")
    parser.add_argument("npy_path", help="Output .npy (N x 7) [x y z r g b label]")
    parser.add_argument("--label", type=int, default=0, help="Dummy label to fill (default: 0)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Multiply XYZ by this (use 0.001 if your scan is in mm and you want meters)")
    parser.add_argument("--downsample_voxel", type=float, default=0.01, # THIS NUMBER SEEMS REASONABLE AS DEFAULT
                        help="Optional voxel size for downsampling (meters). 0 disables.")
    parser.add_argument("--y_up_to_z_up", action="store_true",
                        help="Rotate +90° around X so Y-up becomes Z-up (x'=x, y'=-z, z'=y)")
    args = parser.parse_args()
    
    # Check if user has open3d
    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit("Missing dependency: open3d. Install with: pip install open3d") from e

    pcd = o3d.io.read_point_cloud(args.ply_path)
    if len(pcd.points) == 0:
        raise SystemExit("Loaded 0 points from PLY. Check the file.")

    # Optional downsample first (keeps points/colors aligned)
    if args.downsample_voxel and args.downsample_voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.downsample_voxel))

    xyz = np.asarray(pcd.points, dtype=np.float32)

    # Rotate coordinate frame if needed: Y-up -> Z-up
    if args.y_up_to_z_up:
        R = rot_x_deg(+90.0)         # +90° about X
        xyz = xyz @ R.T              # apply rotation

    # Scale units (e.g., mm->m)
    xyz *= float(args.scale)

    # Colors: Open3D gives 0..1 if present
    if pcd.has_colors():
        rgb01 = np.asarray(pcd.colors, dtype=np.float32)
        rgb255 = (rgb01 * 255.0).clip(0, 255).astype(np.float32)
    else:
        rgb255 = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    label = np.full((xyz.shape[0], 1), float(args.label), dtype=np.float32)

    out = np.concatenate([xyz, rgb255, label], axis=1)  # (N, 7)
    np.save(args.npy_path, out)
    print(f"Saved {args.npy_path} with shape {out.shape}")

if __name__ == "__main__":
    main()
