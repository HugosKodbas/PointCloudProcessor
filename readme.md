# Pre-processor

Preprocesses a raw LiDAR point cloud (.ply) for PointNeXt semantic segmentation. Downloads from Google Drive, applies filtering and geometric alignment, and outputs a .npy file in the format expected by PointNeXt (N×7: x, y, z, r, g, b, label).

## Pipeline

1. **Download** — Fetches .ply from a Google Drive link via `gdown`.
2. **Z-up rotation** (`--make_z_up`) — Rotates +90° around the X-axis so that the Z-axis becomes the gravity dimension, as required by the S3DIS-trained model.
3. **Voxel downsampling** (`--downsample_voxel`) — Reduces point density to a uniform grid (default 0.01m).
4. **Radius Outlier Removal** (`--filter_ror`) — Removes isolated outlier points using a KD-tree radius query (r=0.05m, min 20 neighbours).
5. **DBSCAN cluster filtering** (`--dbscan`) — Removes disconnected blobs by keeping only the largest connected component.
6. **Manhattan-world XY alignment** (`--align_xy`) — Rotates around the Z-axis to align walls with the X/Y axes. Uses KNN local normals with a linearity filter and a Manhattan-folded angle histogram (replaces the previous PCA-based method, which was unreliable due to non-uniform point density biasing the eigenvectors).
7. **XY origin shift** — Translates the filtered cloud so that all X and Y coordinates are non-negative (min X = 0, min Y = 0).
8. **Export** — Saves as .npy with shape (N, 7): [x, y, z, r, g, b, label].

## Usage

```
python ply_to_npy_db.py <GOOGLE_DRIVE_LINK> <OUTPUT_NAME> [OPTIONS]
```

`<>` = required, `[]` = optional

| Flag | Default | Description |
|------|---------|-------------|
| `--make_z_up` | off | Rotate +90° around X (Y-up → Z-up) |
| `--downsample_voxel` | 0.01 | Voxel size in meters (0 to disable) |
| `--filter_ror` | off | Enable radius outlier removal |
| `--dbscan` | off | Enable DBSCAN blob removal |
| `--dbscan_eps` | 0.25 | DBSCAN neighbourhood radius (meters) |
| `--dbscan_min_points` | 30 | DBSCAN minimum cluster size |
| `--dbscan_keep_topk` | 1 | Number of largest clusters to keep |
| `--align_xy` | off | Align walls with X/Y axes |
| `--label` | 0 | Dummy label value for the label column |

## Recommended command

```
python ply_to_npy_db.py <GOOGLE_DRIVE_LINK> Office --make_z_up --downsample_voxel 0.02 --filter_ror --dbscan --dbscan_eps 0.25 --dbscan_min_points 30 --align_xy
```

**Note:** The Google Drive file must be shared (Anyone with the link).

## Dependencies

```
pip install numpy scipy open3d gdown
```

## Output

- `./ply_files/Area_5_<name>.ply` — downloaded raw scan
- `./npy_files/Area_5_<name>.npy` — processed output (N×7)

# Post-processor

Responsible for generating floor plans from model-processed point clouds.

*WIP*