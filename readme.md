# Usage
<> - required argument
[] - optional, but highly recommended arguments
```
python ply_to_pointnext_npy.py <LINK_TO_GOOGLE_DRIVE_FILE> [Output_File_Name] [--downsample_voxel] [--dbscan] [--dbscan_eps] [--dbscan_min_points] [--align_xy] [--make_z_up]
```

The above command downloads raw scan from google drive, rotates it around X-axis so that Z axis is the gravity dimension (Z points up), filters outliers and removes disconnected blobs, and also tries to align the dominant direction along the XY-axis. 

*REMEMBER THAT FILES SHOULD RESIDE IN A SHARED FOLDER ON GOOGLE DRIVE*

## Copy paste command
```
python ply_to_npy_db.py <GOOGLE_DRIVE_LINK>>  Office_db_noZ  --downsample_voxel 0.02 --dbscan --dbscan_eps 0.25 --dbscan_min_points 30 --align_xy
```

# Pre-processor

Preprocesses point cloud. Does the following (currently):
- Converts .ply to .npy which is required for PointNeXt
- Rotates point cloud around X-axis in order to make Z-axis point up. Also required for the model, since that is the "gravity dimension"
- Filters the cloud to remove outliers and clusters:
    - Radius-Outlier-Removal (for isolated outliers)
    - DBSCAN (for big clusters or blobs)
- Rotates around Z-axis in order to align walls with XY. Doesn't work very well, since custom scans can still have outliers which affect the eigenvectors.

# Post-processor

Responsible for generating the actual floor plans from model-processed point clouds. 

*WIP*