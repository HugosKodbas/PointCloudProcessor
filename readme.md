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