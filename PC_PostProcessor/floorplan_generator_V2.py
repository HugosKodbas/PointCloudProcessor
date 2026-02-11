import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

wall_csv   = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_wall_coords_cloud0.csv"
door_csv   = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_door_coords_cloud0.csv"
window_csv = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_window_coords_cloud0.csv"

def line_to_line_distance(model1, model2):
    """
    Geometric distance from line1 to line2 for normalized lines ax+by+c=0.
    Works even when (a,b) differ slightly (nearly-parallel case).

    Assumes both lines are normalized: sqrt(a^2+b^2)=1.
    """
    a1, b1, c1 = model1
    a2, b2, c2 = model2

    # closest point on line1 to the origin is p0 = (-a1*c1, -b1*c1)
    x0 = -a1 * c1
    y0 = -b1 * c1

    # distance from that point to line2 is |a2*x0 + b2*y0 + c2|
    return abs(a2 * x0 + b2 * y0 + c2)


def segment_length(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def load_xy(csv_path: str):
    """Load a CSV and return (x, y) as numpy arrays. Expects columns named x,y (case-insensitive)."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    x_col = cols.get("x")
    y_col = cols.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"{csv_path} must contain 'x' and 'y' columns. Found: {list(df.columns)}")
    return df[x_col].to_numpy(), df[y_col].to_numpy()


def fit_line_pca_xy(xy: np.ndarray):
    """
    Fit a 2D line to points (Total Least Squares) via PCA.
    Returns normalized (a, b, c) for ax + by + c = 0, with sqrt(a^2+b^2)=1.
    """
    if xy.shape[0] < 2:
        return None

    mu = xy.mean(axis=0)
    centered = xy - mu

    cov = (centered.T @ centered) / max(xy.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = eigvecs[:, np.argmax(eigvals)]  # direction along line (dx, dy)

    # normal vector is perpendicular to direction
    a = -d[1]
    b =  d[0]
    norm = math.hypot(a, b)
    if norm < 1e-12:
        return None
    a /= norm
    b /= norm
    c = -(a * mu[0] + b * mu[1])
    return (float(a), float(b), float(c))


def line_angle(a, b):
    """
    Orientation of the line direction in radians (0..pi).
    Line direction is (b, -a) because normal is (a,b).
    """
    theta = math.atan2(-a, b)
    while theta < 0:
        theta += math.pi
    while theta >= math.pi:
        theta -= math.pi
    return theta


def is_manhattan_line(model, angle_thresh_deg=10.0):
    """
    Reject diagonals by only allowing lines near horizontal or vertical.
    Uses the direction angle theta in [0, pi).
      - horizontal: theta ~ 0 or pi
      - vertical:   theta ~ pi/2
    """
    a, b, _ = model
    theta = line_angle(a, b)
    deg = abs(math.degrees(theta))
    # fold to [0, 90]
    if deg > 90:
        deg = 180 - deg

    return (deg <= angle_thresh_deg) or (abs(deg - 90) <= angle_thresh_deg)


def point_line_distances(model, xy: np.ndarray):
    """Distances for normalized ax+by+c=0 line: |ax+by+c|."""
    a, b, c = model
    return np.abs(xy[:, 0] * a + xy[:, 1] * b + c)


def line_to_segment(model, inliers_xy: np.ndarray):
    """Convert an infinite line to a segment using min/max projection of inliers."""
    a, b, _ = model
    dx, dy = b, -a  # direction along line
    anchor = inliers_xy.mean(axis=0)

    diffs = inliers_xy - anchor
    t = diffs[:, 0] * dx + diffs[:, 1] * dy
    tmin, tmax = float(t.min()), float(t.max())

    p1 = (float(anchor[0] + tmin * dx), float(anchor[1] + tmin * dy))
    p2 = (float(anchor[0] + tmax * dx), float(anchor[1] + tmax * dy))
    return p1, p2


def boundary_filter_xy(wx, wy, margin=0.20):
    """
    Keep only points within 'margin' of the axis-aligned bounding box.
    This strongly reduces interior clutter and diagonal hallucinations.
    """
    x = np.asarray(wx, dtype=np.float64)
    y = np.asarray(wy, dtype=np.float64)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    keep = (
        (x <= xmin + margin) |
        (x >= xmax - margin) |
        (y <= ymin + margin) |
        (y >= ymax - margin)
    )
    return x[keep], y[keep]


def ransac_walls_xy(wx, wy,
                    distance_threshold=0.03,
                    min_inliers=2000,
                    max_iterations=500,
                    max_walls=20,
                    refine_with_pca=True,
                    manhattan_only=True,
                    manhattan_angle_thresh_deg=10.0,
                    min_wall_length=0.5):
    """
    Detect multiple wall lines using RANSAC. Returns list of dicts:
      { 'model': (a,b,c), 'inliers_xy': ndarray Nx2, 'p1':(x,y), 'p2':(x,y) }
    """
    xy_all = np.column_stack([wx, wy]).astype(np.float64)
    remaining = xy_all.copy()

    walls = []
    print("Coordinates:", remaining.shape[0])

    for w in range(max_walls):
        if remaining.shape[0] < min_inliers:
            break

        best_inlier_idx = None
        best_model = None
        best_count = 0

        n = remaining.shape[0]

        for _ in range(max_iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = remaining[i1]
            p2 = remaining[i2]
            if np.allclose(p1, p2):
                continue

            a = float(p2[1] - p1[1])
            b = float(p1[0] - p2[0])
            c = -(a * p1[0] + b * p1[1])

            denom = math.hypot(a, b)
            if denom < 1e-12:
                continue

            # normalize so distances are |ax+by+c|
            a /= denom
            b /= denom
            c /= denom
            model = (a, b, c)

            # Optional: reject diagonals early
            if manhattan_only and (not is_manhattan_line(model, manhattan_angle_thresh_deg)):
                continue

            d = point_line_distances(model, remaining)
            inlier_mask = d < distance_threshold
            count = int(inlier_mask.sum())

            if count > best_count:
                best_count = count
                best_inlier_idx = np.where(inlier_mask)[0]
                best_model = model

        if best_model is None or best_count < min_inliers:
            break

        inliers_xy = remaining[best_inlier_idx]

        # refine line with PCA on inliers (more stable)
        if refine_with_pca:
            refined = fit_line_pca_xy(inliers_xy)
            if refined is not None:
                # enforce manhattan after refit too
                if (not manhattan_only) or is_manhattan_line(refined, manhattan_angle_thresh_deg):
                    best_model = refined

        p1_seg, p2_seg = line_to_segment(best_model, inliers_xy)
        if segment_length(p1_seg, p2_seg) < min_wall_length:   # <-- choose your minimum, e.g. 0.5m
            continue


        walls.append({
            "model": best_model,
            "inliers_xy": inliers_xy,
            "p1": p1_seg,
            "p2": p2_seg,
            "n_inliers": int(inliers_xy.shape[0]),
        })
        print(f"Wall {len(walls)}: inliers={inliers_xy.shape[0]} model={best_model}")

        # remove inliers for next iteration
        keep_mask = np.ones(remaining.shape[0], dtype=bool)
        keep_mask[best_inlier_idx] = False
        remaining = remaining[keep_mask]

    return walls


def merge_collinear_walls(walls,
                          angle_thresh_deg=5.0,
                          offset_thresh=0.08,
                          min_merged_inliers=2000,
                          # NEW: reject merged walls that don't actually fit
                          fit_median_thresh=0.03,
                          # NEW: keep only horizontal/vertical
                          manhattan_only=True,
                          manhattan_angle_thresh_deg=10.0):
    """
    Merge near-parallel, nearby lines.
    Also rejects merged candidates if the refit residual is too large (prevents diagonals).
    """
    if not walls:
        return []

    angle_thresh = math.radians(angle_thresh_deg)

    def canonicalize(model):
        a, b, c = model
        # make representation consistent (flip sign)
        if c < 0 or (abs(c) < 1e-12 and a < 0):
            return (-a, -b, -c)
        return (a, b, c)

    items = []
    for w in walls:
        a, b, c = canonicalize(w["model"])
        theta = line_angle(a, b)
        items.append({
            "model": (a, b, c),
            "theta": theta,
            "inliers_xy": w["inliers_xy"],
        })

    used = [False] * len(items)
    merged = []

    for i in range(len(items)):
        if used[i]:
            continue
        used[i] = True

        group_inliers = [items[i]["inliers_xy"]]
        #a0, b0, c0 = items[i]["model"]
        t0 = items[i]["theta"]

        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            #aj, bj, cj = items[j]["model"]
            tj = items[j]["theta"]

            dtheta = abs(tj - t0)
            dtheta = min(dtheta, math.pi - dtheta)

            # near-parallel AND close offset
            # (note: parallel lines can have different c, but if they represent the same wall they should be close)
            dist_lines = line_to_line_distance((items[i]["model"]), (items[j]["model"]))
            if dtheta <= angle_thresh and dist_lines <= offset_thresh:
                used[j] = True
                group_inliers.append(items[j]["inliers_xy"])

        all_inliers = np.vstack(group_inliers)
        if all_inliers.shape[0] < min_merged_inliers:
            continue

        refit = fit_line_pca_xy(all_inliers)
        if refit is None:
            continue

        if manhattan_only and (not is_manhattan_line(refit, manhattan_angle_thresh_deg)):
            continue

        d = point_line_distances(refit, all_inliers)
        if float(np.median(d)) > fit_median_thresh:
            continue

        p1_seg, p2_seg = line_to_segment(refit, all_inliers)
        merged.append({
            "model": refit,
            "inliers_xy": all_inliers,
            "p1": p1_seg,
            "p2": p2_seg,
            "n_inliers": int(all_inliers.shape[0]),
            "median_residual": float(np.median(d)),
        })

    return merged


def plot_walls(wx, wy, walls, title="Walls"):
    plt.figure(figsize=(10, 6))
    plt.scatter(wx, wy, s=1)

    for w in walls:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        plt.plot([x1, x2], [y1, y2], linewidth=2)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def create_intersections(walls):
    """
    Create intersection points between all pairs of walls.
    Returns list of (x,y) tuples.
    """
    print(f"Creating intersections for {len(walls)} walls")

def main():
    wx, wy = load_xy(wall_csv)

    wx_f, wy_f = boundary_filter_xy(wx, wy, margin=0.20)
    print(f"Boundary-filtered points: {len(wx_f)}/{len(wx)}")

    # 1) RANSAC: extract candidate lines
    print("Right-wall candidate count:",
    np.sum(wx >= wx.max() - 0.20))
    walls = ransac_walls_xy(
        wx_f, wy_f,
        distance_threshold=0.03,     # tighter than 0.05
        min_inliers=1000,            # increase for stability
        max_iterations=500,          # more robust
        max_walls=20,
        refine_with_pca=True,
        manhattan_only=True,
        manhattan_angle_thresh_deg=10.0,
        min_wall_length=0.1           # <-- new filter to remove tiny segments
    )

    # 2) Merge: collapse duplicates / fragments
    merged = merge_collinear_walls(
        walls,
        angle_thresh_deg=5.0,
        offset_thresh=0.14,
        min_merged_inliers=2000,
        fit_median_thresh=0.03,      # must match distance_threshold scale
        manhattan_only=True,
        manhattan_angle_thresh_deg=10.0
    )

    create_intersections(merged)

    print(f"RANSAC walls: {len(walls)}")
    print(f"Merged walls: {len(merged)}")
    for i, w in enumerate(merged, 1):
        print(f"Merged {i}: model={w['model']} inliers={w['n_inliers']} median_residual={w['median_residual']:.4f}")

    plot_walls(wx_f, wy_f, merged, title="Merged wall segments (Manhattan + residual check)")


if __name__ == "__main__":
    main()
