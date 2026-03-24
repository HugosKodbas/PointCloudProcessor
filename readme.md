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
| `--plot` | off | Show before/after alignment plot |

## Recommended command

```
python ply_to_npy_final.py <GOOGLE_DRIVE_LINK> <ROOM_NAME> --make_z_up --downsample_voxel 0.02 --filter_ror --dbscan --dbscan_eps 0.25 --dbscan_min_points 30 --align_xy --plot
```

## For scans acquired with PolyCam
```
python ply_to_npy_final.py <GOOGLE_DRIVE_LINK> <ROOM_NAME> --downsample_voxel 0.02 --filter_ror --dbscan --dbscan_eps 0.25 --dbscan_min_points 30 --align_xy --plot
```

**Note:** The Google Drive file must be shared (Anyone with the link).

## Dependencies

```
pip install numpy scipy open3d gdown matplotlib
```

## Output

- `./ply_files/Area_5_<name>.ply` — downloaded raw scan
- `./npy_files/Area_5_<name>.npy` — processed output (N×7)

# Post-processor

Generates structured 2D floor plans (JSON) from semantically segmented point cloud data. Operates on per-class CSV files containing wall, door, and window coordinates produced either by PointNeXt predictions or extracted from S3DIS ground-truth labels.

## Pipeline

1. **RANSAC line fitting** — Iteratively extracts dominant line segments from wall/door/window point clouds under a Manhattan-world angle constraint.
2. **Parallel line detection & removal** — Identifies near-parallel duplicate segments using a Disjoint-Set-Union structure and collapses each group to the strongest segment.
3. **Gap detection & wall splitting** — Projects inliers onto the line direction, detects gaps exceeding a threshold, and splits into separate segments.
4. **Corner extension** — Snaps nearby wall endpoints to pairwise line intersections to form clean corners.
5. **Collinear wall bridging** — Inserts short bridging segments between nearby collinear wall fragments.
6. **Door & window filtering** — Matches door/window segments to walls by midpoint distance and angle, discards unattached detections.
7. **Wall segment partitioning** — Projects door/window segments onto matched walls and removes the corresponding intervals, leaving wall pieces on either side of each opening.
8. **JSON export** — Writes wall, door, and window endpoints and lengths to a structured JSON file.

## Usage

### Batch mode (Experiment 1 workflow)

Process all rooms in a directory at once. Requires `--gt` or `--pred` to indicate the data source.

```
# Ground-truth floor plans
python floorplan_generator_V3.py ./gt_coords --gt --batch

# Predicted floor plans
python floorplan_generator_V3.py ./pred_coords --pred --batch
```

Output:
```
./output/Area_5_hallway_8_floorplan_gt.json
./output/Area_5_hallway_8_floorplan_pred.json
...
```

### Single room

```
# By room name (auto-discovers the 3 CSVs in the directory)
python floorplan_generator_V3.py ./gt_coords --gt --room_name Area_5_hallway_8

# For custom scans
python floorplan_generator_V4.py ./Custom_Scans --pred --room_name Area_5_<ROOM_NAME> [--plot]

# By explicit CSV paths
python floorplan_generator_V3.py --pred --wall_csv wall.csv --door_csv door.csv --window_csv window.csv --room_name my_room
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gt` | — | Input CSVs are ground-truth (mutually exclusive with `--pred`) |
| `--pred` | — | Input CSVs are predictions (mutually exclusive with `--gt`) |
| `--batch` | off | Process all rooms found in the input directory |
| `--output_dir` | `./output` | Directory for output JSON files |
| `--plot` | off | Show debug plot for each room |
| `--wall_csv` | — | Path to wall CSV (single-file mode) |
| `--door_csv` | — | Path to door CSV (single-file mode) |
| `--window_csv` | — | Path to window CSV (single-file mode) |
| `--room_name` | — | Room name for output file (single-file mode) |

### Expected CSV naming conventions

| Source | Wall | Door | Window |
|--------|------|------|--------|
| `--gt` | `<room>_gt_wall_coords.csv` | `<room>_gt_door_coords.csv` | `<room>_gt_window_coords.csv` |
| `--pred` | `<room>_wall_coords.csv` | `<room>_door_coords.csv` | `<room>_window_coords.csv` |

## Dependencies

```
pip install numpy pandas matplotlib
```

# Ground-Truth Extraction Utility

Extracts ground-truth wall, door, and window coordinate CSVs from S3DIS `.npy` files. Produces the same CSV format as the PointNeXt inference script (`main.py`), so the output can be fed directly into the post-processor for comparison.

## Usage

```
# Single room
python extract_gt_coords.py Area_5_hallway_8.npy

# All Area 5 rooms in a directory
python extract_gt_coords.py /path/to/s3dis/raw/ --batch
```

| Flag | Default | Description |
|------|---------|-------------|
| `--batch` | off | Process all `Area_5_*.npy` files in the input directory |
| `--output_dir` | `./gt_coords` | Directory for output CSVs |

## Output

```
./gt_coords/Area_5_hallway_8_gt_wall_coords.csv
./gt_coords/Area_5_hallway_8_gt_door_coords.csv
./gt_coords/Area_5_hallway_8_gt_window_coords.csv
```

## Dependencies

```
pip install numpy
```