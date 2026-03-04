import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Any, Optional, Tuple

def closest_endpoint_pair_between_segments(wall1, wall2):
    """
    Find the closest endpoint-to-endpoint pair between two segments.

    Parameters
    ----------
    endpoints_a : tuple/list of 2 points
        (p1, p2) where each point is (x, y) or array-like.
    endpoints_b : tuple/list of 2 points
        (q1, q2)

    Returns
    -------
    best_pair : dict
        {
          "a_index": 0 or 1,
          "b_index": 0 or 1,
          "p": np.ndarray shape (2,),
          "q": np.ndarray shape (2,),
          "distance": float
        }
    """

    p1 = np.asarray(wall1['endpoints'][0], dtype=np.float64)
    p2 = np.asarray(wall1['endpoints'][1], dtype=np.float64)
    q1 = np.asarray(wall2['endpoints'][0], dtype=np.float64)
    q2 = np.asarray(wall2['endpoints'][1], dtype=np.float64)

    PAIR_P = [p1, p2]
    PAIR_Q = [q1, q2]

    closest_pair = None
    for x1, y1 in enumerate(PAIR_P):
        for x2, y2 in enumerate(PAIR_Q):
            dist = np.linalg.norm(y1 - y2)
            if closest_pair is None or dist < closest_pair["distance"]:
                closest_pair = {
                    "model1": wall1['original_id'],
                    "model2": wall2['original_id'],
                    "a_index": x1,
                    "b_index": x2,
                    "p": y1,
                    "q": y2,
                    "distance": dist
                }
    return closest_pair

def is_parallel(model1, model2, angle_threshold=0.05):
    '''Helper function for detect_parallel_lines, checks if two lines are parallel within an angle threshold.'''
    a1, b1, _ = model1
    a2, b2, _ = model2

    theta1 = normalize_angle(a1, b1)
    theta2 = normalize_angle(a2, b2)

    dtheta = angle_between_lines(theta1, theta2)

    return dtheta <= angle_threshold

def line_from_points(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    x1, y1 = float(p[0]), float(p[1])
    x2, y2 = float(q[0]), float(q[1])

    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1

    n = np.hypot(a, b)
    if n < eps:
        return None
    a, b, c = a/n, b/n, c/n

    # optional: consistent sign
    if a < 0 or (abs(a) < eps and b < 0):
        a, b, c = -a, -b, -c

    return (a, b, c)

# Co-linear Wall Bridging
""" def bridge_collinear_walls(walls, separation_threshold):
    n = len(walls)

    new_models = [] # Start with original walls, we will add new bridged walls to this list.
    for w1 in range(n):
        for w2 in range(w1 + 1, n):
            if is_parallel(walls[w1]['model'], walls[w2]['model']):
                pair = closest_endpoint_pair_between_segments(walls[w1], walls[w2])
                if pair['distance'] <= separation_threshold:
                    a, b, c = line_from_points(pair['p'], pair['q'])
                    model = (a, b, c)
                    new_models.append({
                        'original_id': f"{len(walls)+1}",
                        'wall_id': None,
                        'model': model,
                        'inliers': [],
                        'endpoints': (pair['p'], pair['q']),
                        'num_inliers': 0,
                        'is_parallel': False
                    })

    walls = walls + new_models
    return walls
    """
# Patched version of above function
def bridge_collinear_walls(walls, separation_threshold):
    n = len(walls)
    new_models = []

    for w1 in range(n):
        for w2 in range(w1 + 1, n):
            if is_parallel(walls[w1]['model'], walls[w2]['model']):
                pair = closest_endpoint_pair_between_segments(walls[w1], walls[w2])

                if pair['distance'] <= separation_threshold:
                    coeffs = line_from_points(pair['p'], pair['q']) # Guard for degenerate case where endpoints are the same, which causes line_from_points to fail. This can happen when two walls are very close and their endpoints are snapped together in the intersection extension step.
                    if coeffs is None:
                        continue
                    a, b, c = coeffs

                    model = (a, b, c)
                    new_models.append({
                        'original_id': f"{len(walls)+1}",
                        'wall_id': None,
                        'model': model,
                        'inliers': [],
                        'endpoints': (pair['p'], pair['q']),
                        'num_inliers': 0,
                        'is_parallel': False
                    })

    return walls + new_models

# ### 
# GAP DETECTION
# #################################################################################################################
def split_walls_by_gaps(models, xy_matrix, gap_treshold):
    # Walk through each wall model and check for gaps in the inlier points. If a gap is found, split the wall into two segments.
    '''
    Split a wall whose inlier points are clustered in separate patches, i.e contains a gap somewhere.

    Parameters
    ---
    models    : dict with keys 'model', 'inliers', 'wall_id', 'num_inliers', 'endpoints', 'is_parallel'
    xy_matrix : (N, 2) numpy array of all wall points (x,y)
    gap_threshold : float, distancce in meters that defines a gap. If the detected gap is greater than this threshold, we split the wall into two segments.
                    this should remove most floating wall segments.
    
    
    Returns
    ---
    A list of wall dicts. If no gap is found the list contains the original wall, unchanged.
    
    
    '''

    inlier_idx = models['inliers']
    inlier_pts = xy_matrix[inlier_idx]

    a, b, _ = models['model']

    direction = np.array([-b, a], dtype=np.float64) # Direction vector of the line, perpendicular to the normal vector (a,b)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    # Scalar projection of eacch inlier onto the line direction
    proj = inlier_pts @ direction
    order = np.argsort(proj)
    sorted_proj = proj[order]

    gaps = np.diff(sorted_proj)
    gap_mask = gaps > gap_treshold

    if not np.any(gap_mask):
        return [models]

    # Build segments
    split_after = np.where(gap_mask)[0]
    starts = np.concatenate(([0], split_after + 1))
    ends = np.concatenate((split_after + 1, [len(order)]))

    segments = []

    for s, e in zip(starts, ends):
        seg_order = order[s:e]
        # Segments with less than 800 inliers are considered junk.
        if len(seg_order) < 120:
            continue
        seg_global = [inlier_idx[i] for i in seg_order]
        p1, p2 = calculate_endpoints(xy_matrix[seg_global])
        print(f"Splitting wall {models['original_id']} into segment with {len(seg_global)} inliers, endpoints {p1}, {p2}")
        L = line_length(p1, p2)
        print(f"Segment length: {L:.2f}m")
        if L < 0.45:
            print(f"Discarding segment {models['original_id']} with length {L:.2f}m, which is below the {L*100:.2f}cm threshold.")
            continue
        segments.append({
            'original_id': models['original_id'], # We can keep the same original_id for all segments, since they come from the same original wall.
            'wall_id': models['wall_id'], # We can keep the same wall_id for
            'model': models['model'], # The line model is the same for both segments, since they lie on the same line.
            'inliers': seg_global,
            'endpoints': (p1, p2),
            'num_inliers': len(seg_global),
            'is_parallel': models['is_parallel']
        })

    return segments

def apply_gap_detection(models, xy_matrix, gap_treshold):
    new_models = []
    for wall in models:
        new_models.extend(split_walls_by_gaps(wall, xy_matrix, gap_treshold))

    # Re-assign wall_ids so they are contiguous from 0
    for new_id, w in enumerate(new_models):
        w['wall_id'] = new_id

    return new_models


# ###
# INTERSECTION + CORNER EXTENSION
# ###

def line_intersection_point(model1, model2):
    '''
    Intersection of lines given in ax + by + c = 0 form. Returns None if lines are parallel.
    '''

    a1, b1, c1 = model1
    a2, b2, c2 = model2

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    
    # Apply cramers rule
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det

    return np.array([x, y], dtype=np.float64)

def extend_wall_to_intersection(
    walls,
    max_extension,
    endpoint_guard=0.10,   # meters: if an endpoint is already this close to another endpoint, don't move it
    ignore_same_wall=True  # ignore comparing endpoint to other endpoint in same wall
):
    """
    Extend/snap wall endpoints to pairwise line intersections, but only if:
      1) endpoint is within max_extension of the intersection, and
      2) endpoint is NOT already very close to some other wall endpoint (endpoint_guard).

    Parameters
    ----------
    walls : list[dict]
        Each dict has keys: 'model' (line params), 'endpoints' ((x,y),(x,y)), etc.
    max_extension : float
        Maximum allowed distance from endpoint to intersection to snap.
    endpoint_guard : float
        If an endpoint is already within this distance of another wall endpoint, we do not move it.
    ignore_same_wall : bool
        If True, do not treat the other endpoint of the same wall as "another wall endpoint".
    """

    n = len(walls)

    # Original endpoints (used for distance tests)
    orig = [[np.array(wall['endpoints'][0], dtype=np.float64),
             np.array(wall['endpoints'][1], dtype=np.float64)]
            for wall in walls]
    
    orig_id = [w.get("original_id", w.get("wall_id", i)) for i, w in enumerate(walls)]

    # Mutable endpoints we will write into
    new_ep = [[orig[i][0].copy(), orig[i][1].copy()] for i in range(n)]

    # Flatten all endpoints for "already connected?" tests
    # entries: (wall_index, endpoint_index, point_array)
    all_endpoints = [(wi, ei, orig[wi][ei]) for wi in range(n) for ei in range(2)]

    def endpoint_is_already_near_other_endpoint(wi, ei, pt):
        """True if pt is within endpoint_guard of any OTHER wall endpoint."""
        for wj, ej, q in all_endpoints:
            if wj == wi and ej == ei:
                continue
            if ignore_same_wall and wj == wi:
                continue
            if np.linalg.norm(pt - q) <= endpoint_guard:
                print(
                    f"GUARD: wall orig={orig_id[wi]} (idx {wi}) ep {ei} "
                    f"blocked by wall orig={orig_id[wj]} (idx {wj}) ep {ej}, "
                    f"dist={np.linalg.norm(pt-q):.3f}")
                return True
        return False

    for i in range(n):
        for j in range(i + 1, n):
            pt = line_intersection_point(walls[i]['model'], walls[j]['model'])
            if pt is None:
                continue
            pt = np.array(pt, dtype=np.float64)

            # Wall i: nearest endpoint to pt
            di = [np.linalg.norm(orig[i][ei] - pt) for ei in range(2)]
            best_ei = int(np.argmin(di))
            if di[best_ei] > max_extension:
                continue

            # Guard: don't move if already near some other endpoint
            if endpoint_is_already_near_other_endpoint(i, best_ei, orig[i][best_ei]):
                continue

            # Wall j: nearest endpoint to pt
            dj = [np.linalg.norm(orig[j][ej] - pt) for ej in range(2)]
            best_ej = int(np.argmin(dj))
            if dj[best_ej] > max_extension:
                continue

            if endpoint_is_already_near_other_endpoint(j, best_ej, orig[j][best_ej]):
                continue

            # Snap both endpoints
            print(f"Extending wall {walls[i]['original_id']} endpoint {best_ei} and wall {walls[j]['original_id']} endpoint {best_ej} to intersection point {pt}.")
            new_ep[i][best_ei] = pt.copy()
            new_ep[j][best_ej] = pt.copy()

    for i, w in enumerate(walls):
        w['endpoints'] = (new_ep[i][0], new_ep[i][1])

    return walls

""" def extend_wall_to_intersection(walls, max_extension):
    '''
    Extend a wall segment to a corner point, if the endpoint is within max_extension distance from the corner point.
    '''
    n = len(walls)
    
    orig = [[np.array(wall['endpoints'][0], dtype=np.float64),
            np.array(wall['endpoints'][1], dtype=np.float64)]
            for wall in walls]
    
    new_ep = [[orig[i][0].copy(), orig[i][1].copy()] for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            pt = line_intersection_point(walls[i]['model'], walls[j]['model'])
            if pt is None:
                continue

            # For wall i: pick the SINGLE nearest endpoint within max_extension
            di = [np.linalg.norm(orig[i][ei] - pt) for ei in range(2)]
            best_ei = int(np.argmin(di))
            if di[best_ei] > max_extension:
                continue

            # For wall j: same — pick SINGLE nearest endpoint
            dj = [np.linalg.norm(orig[j][ej] - pt) for ej in range(2)]
            best_ej = int(np.argmin(dj))
            if dj[best_ej] > max_extension:
                continue

            # Both walls have an endpoint close enough → snap them
            print(f"Extending wall {walls[i]['wall_id']} endpoint {best_ei} and wall {walls[j]['wall_id']} endpoint {best_ej} to intersection point {pt}.")
            new_ep[i][best_ei] = pt.copy()
            new_ep[j][best_ej] = pt.copy()

    for i, w in enumerate(walls):
        w['endpoints'] = (new_ep[i][0], new_ep[i][1])

    return walls """

def calculate_centroid(wx, wy):
    '''Function that calculates the median of all points, should result in a point that lies within the room, which will help us remove some lines'''
    x_mean, y_mean = np.mean(wx), np.mean(wy)

    return (x_mean, y_mean)
    
def distance_to_centroid(line, centroid):
    x_centroid = centroid[0]
    y_centroid = centroid[1]
    a, b, c = line

    # Distance is given by
    return abs(a*x_centroid + b*y_centroid + c) / (math.hypot(a, b))

# Transparent surfaces (windows, glass doors etc) and other architecture can cause duplicate surfaces that are interpreted as walls. 
# Option A: Choose the segment with the most amount of inliers, as it is likely this is the wall. More realisticly placed walls, but might cause trouble with
# shorter wall clusters not getting discovered.
# Option B: Always take the wall closest to the room centroid, might give a clearner look but will not be as dimensionally accurate.
def remove_parallel_walls(models, parallel_groups):

    if not parallel_groups:
        return models

    # Flatten all wall_ids that belong to parallel groups
    parallel_ids = {wid for group in parallel_groups for wid in group}

    cleaned_walls = [
        w for w in models
        if w['wall_id'] not in parallel_ids
    ]

    # For each group, keep only the walls with the most number of inliers (since model is very confident that this is the actual wall.)
    for group in parallel_groups:
        group_models = [
            w for w in models
            if w['wall_id'] in group
        ]
        if not group_models:
            continue
        
        strongest = max(group_models, key=lambda w: w['num_inliers'])

        cleaned_walls.append(strongest)

    return cleaned_walls

def normalize_coefficients(a, b, c):
    s = math.hypot(a, b)
    eps = 1e-8 # Some small numer near zero
    if s < eps:
        return None
    a = a / s
    b = b / s
    c = c / s

    # Flip the sign
    if c < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)

def distance_between_parallel_lines(L1, L2):
    ''' L1, L2 on form ax + bx + c = 0 '''
    # Distance between normalized lines from any point on L1, L2
    (a1, b1, c1) = L1
    (a2, b2, c2) = L2

    p0x = -a1 * c1
    p0y = -b1 * c1

    d = abs(a2*p0x + b2*p0y + c2)
    #print(d)
    return d

def angle_between_lines(theta1, theta2):
    '''Helper function for detect_parallel_lines, if angle between two lines is 0, they are parallel.'''
    d = abs(theta1 - theta2)
    return min(d, math.pi - d)


def detect_parallel_lines(walls, distance_threshold, angle_threshold):
    ''' Takes models (a, b, c) as input and investigates if they are parallel within a distance and angle threshold, 
    and collapses into one line in such cases. Lines within 15 cm (0.15m) should be merged.'''

    angle_threshold_radians = math.radians(angle_threshold) 

    model_angles = []
    for wall in range(len(walls)):
        angle = normalize_angle(walls[wall]['model'][0], walls[wall]['model'][1]) # Angle of the wall segment. Used for comparison later.
        model_angles.append({
            'original_id': walls[wall]['original_id'],
            'idx': wall, # We save the index in case of unexpected shuffling. 
            'wall_id': walls[wall]['wall_id'],
            'model': walls[wall]['model'],
            'angle': angle
        })
    
    # Disjoint-Set-Union. We want to organize parallel walls into groups.
    parent = {w['wall_id']: w['wall_id'] for w in model_angles}
    rank = {w['wall_id']: 0 for w in model_angles}
    
    # Simple helpers specific to this function... we can get away with placing them here.
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1


    parallel_pairs = []

    for i in range(len(model_angles)):
        for j in range(i+1, len(model_angles)):

            # Check for parallelness
            dtheta = angle_between_lines(model_angles[i]['angle'], model_angles[j]['angle'])
            if dtheta > angle_threshold_radians:
                continue

            # Distance check
            distance = distance_between_parallel_lines(model_angles[i]['model'], model_angles[j]['model'])
            if distance <= distance_threshold:
                wi = model_angles[i]['wall_id']
                wj = model_angles[j]['wall_id']

                parallel_pairs.append((wi, wj, distance, dtheta))

                union(wi, wj)

                walls[model_angles[i]['idx']]['is_parallel'] = True
                walls[model_angles[j]['idx']]['is_parallel'] = True

                #parallel_pairs.append((model_angles[i]['model'], model_angles[j]['model']))
    
    groups_dict = {}
    for w in model_angles:
        root = find(w['wall_id'])
        groups_dict.setdefault(root, set()).add(w['wall_id'])

    # Group size is minimum 2
    parallel_groups = [g for g in groups_dict.values() if len(g) >= 2]

    return walls, parallel_pairs, parallel_groups

# Calculate endpoints
def calculate_endpoints(inlier_points):
    x_min_idx = np.argmin(inlier_points[:, 0])
    x_max_idx = np.argmax(inlier_points[:, 0])
    y_min_idx = np.argmin(inlier_points[:, 1])
    y_max_idx = np.argmax(inlier_points[:, 1])
    
    x_length = inlier_points[x_max_idx, 0] - inlier_points[x_min_idx, 0] # X-axis extent
    y_length = inlier_points[y_max_idx, 1] - inlier_points[y_min_idx, 1] # Y-axis extent

    if x_length > y_length:
        p1 = inlier_points[x_min_idx]
        p2 = inlier_points[x_max_idx]
    else:
        p1 = inlier_points[y_min_idx]
        p2 = inlier_points[y_max_idx]

    return p1, p2

# Angle calculation utility
def normalize_angle(a, b):
    """Calculate angle of vector (a,b) in radians, normalized to [0, pi)."""
    theta = math.atan2(-a, b)  # atan2 returns [-pi, pi]
    while theta < 0:
        theta += math.pi
    while theta >= math.pi:
        theta -= math.pi
    return theta

# We only want to fit lines that are horizontal or vertical, manhattan world assumption.
def line_angle(p1, p2, max_angle=0.05):
    """Calculate angle of line p1p2 in radians."""
    a = p1[0] - p2[0] # Delta x
    b = p1[1] - p2[1] # Delta y

    theta = normalize_angle(a, b)

    rad_to_deg = abs(math.degrees(theta))
    
    # Check if the line is close to horizontal or vertical, and fold into [0, 90] range
    if rad_to_deg > 90:
        rad_to_deg = 180 - rad_to_deg
    
    return (rad_to_deg <= max_angle) or (abs(rad_to_deg - 90) <= max_angle)

def distance_to_line(a, b, c, x0, y0):
    """Calculate the distance from a point to a line ax+by+c=0"""
    return np.abs(a * x0 + b * y0 + c) / math.hypot(a, b)

def line_length(p1, p2):
    # Length of a line given two points is: sqrt ((X_2 - X_1)^2 + (Y_2 - Y_1)^2)
    return math.hypot((p2[0] - p1[0]), (p2[1] - p1[1]))

def _to_xy(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p[:2].copy()


def _seg_dir(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = b - a
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0], dtype=float)  # arbitrary fallback
    return v / n


def _angle_between_dirs(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    # Returns angle in radians between two direction vectors, in [0, pi/2] for undirected segments
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return 0.0
    u = u / nu
    v = v / nv
    # undirected: treat u and -u equivalent, so use abs(dot)
    dot = float(np.clip(abs(np.dot(u, v)), -1.0, 1.0))
    return float(np.arccos(dot))


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    # Distance from point p to segment ab in 2D
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < eps:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / ab2)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def filter_door_segments_keep_frame_opening(
    doors: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
    threshold: float,
    *,
    use_angle: bool = True,
    angle_threshold_deg: float = 10.0,
    angle_weight: float = 0.10,
    require_same_matched_wall_for_parallel_competitors: bool = True,
    wall_id_key: str = "wall_id",
    endpoints_key: str = "endpoints",
    inliers_key: str = "num_inliers",
    add_match_fields: bool = True,
    max_endpoint_wall_distance: float = None,
) -> List[Dict[str, Any]]:
    """
    Filters door segments by wall attachment and removes "open door leaf" segments,
    keeping the segment most consistent with being a door frame/opening on a wall.

    Logic:
      1) Match each door segment to its best wall (min midpoint-to-wall distance,
         optionally with angle penalty).
      2) Discard doors whose matched distance > threshold.
      3) Group doors by matched wall.
      4) For each group, keep only the door segment that "hugs" the wall best:
           - minimize max(endpoint distances to matched wall)
           - then minimize midpoint distance
           - then maximize num_inliers (tie-break)

    Parameters
    ----------
    doors, walls : list of dict
        Each entry must contain endpoints under `endpoints_key` as ((x1,y1),(x2,y2)).
        Walls should ideally have a stable id at `wall_id_key`. If missing, index is used.
    threshold : float
        Maximum allowed midpoint-to-wall-segment distance for a door to be considered attached.
    use_angle : bool
        If True, also penalize angle mismatch between door and wall directions.
    angle_threshold_deg : float
        If use_angle=True, walls with mismatch above this are still considered, but penalized
        (tighten by setting low or set to 180 to effectively disable thresholding).
    angle_weight : float
        Adds `angle_weight * angle_radians` to the matching score. Units: meters per radian.
        Keep small so distance dominates.
    require_same_matched_wall_for_parallel_competitors : bool
        If True, comparisons happen only within each matched-wall group (recommended).
    wall_id_key, endpoints_key, inliers_key : str
        Dict keys.
    add_match_fields : bool
        If True, annotate kept door dicts with:
          - 'matched_wall_id'
          - 'matched_wall_index'
          - 'matched_wall_distance'
          - 'matched_wall_score'

    Returns
    -------
    List[Dict[str, Any]]
        Filtered door dicts (references to original dicts, possibly with extra fields).
    """
    if not doors or not walls:
        return []

    # Precompute wall segments + ids
    wall_data: List[Tuple[int, Any, np.ndarray, np.ndarray, np.ndarray]] = []
    for wi, w in enumerate(walls):
        if endpoints_key not in w:
            continue
        w1, w2 = w[endpoints_key]
        a = _to_xy(w1)
        b = _to_xy(w2)
        wid = w.get(wall_id_key, wi)
        wdir = _seg_dir(a, b)
        wall_data.append((wi, wid, a, b, wdir))

    if not wall_data:
        return []

    # 1) Match each door to best wall and filter by threshold
    attached: List[Dict[str, Any]] = []
    for d in doors:
        if endpoints_key not in d:
            continue
        d1, d2 = d[endpoints_key]
        p1 = _to_xy(d1)
        p2 = _to_xy(d2)
        mid = 0.5 * (p1 + p2)
        ddir = _seg_dir(p1, p2)

        best = None  # (score, dist_mid, angle, wi, wid)
        for wi, wid, wa, wb, wdir in wall_data:
            dist_mid = _point_to_segment_distance(mid, wa, wb)
            if use_angle:
                ang = _angle_between_dirs(ddir, wdir)
                ang_deg = ang * 180.0 / np.pi

                # HARD REJECT: door opening should be roughly parallel to the wall it sits on
                if ang_deg > angle_threshold_deg:
                    continue

                score = dist_mid + angle_weight * ang
            else:
                ang = 0.0
                score = dist_mid

            if best is None or score < best[0]:
                best = (score, dist_mid, ang, wi, wid)

        if best is None:
            continue

        score, dist_mid, ang, wi, wid = best

        # Attachment criterion uses pure midpoint distance (not score)
        if dist_mid > float(threshold):
            continue

        if add_match_fields:
            d["matched_wall_id"] = wid
            d["matched_wall_index"] = wi
            d["matched_wall_distance"] = float(dist_mid)
            d["matched_wall_score"] = float(score)
            d["matched_wall_angle_deg"] = float(ang * 180.0 / np.pi)

        attached.append(d)

    if not attached:
        return []

    # 2) Group by matched wall
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    if require_same_matched_wall_for_parallel_competitors:
        for d in attached:
            wid = d.get("matched_wall_id", None)
            if wid is None:
                # fallback: treat as its own group
                wid = id(d)
            groups.setdefault(wid, []).append(d)
    else:
        # If user disables grouping requirement, treat all as one group
        groups["__all__"] = attached

    # Helper: get wall endpoints by matched_wall_index
    wall_by_index = {wi: (wa, wb, wdir, wid) for (wi, wid, wa, wb, wdir) in wall_data}

    # 3) For each group, keep the door that hugs the matched wall best
    kept: List[Dict[str, Any]] = []
    for wid, ds in groups.items():
        if not ds:
            continue
        if len(ds) == 1:
            kept.append(ds[0])
            continue

        best_d = None
        best_key = None  # tuple for lexicographic compare

        for d in ds:
            d1, d2 = d[endpoints_key]
            p1 = _to_xy(d1)
            p2 = _to_xy(d2)
            mid = 0.5 * (p1 + p2)

            wi = d.get("matched_wall_index", None)
            if wi is None or wi not in wall_by_index:
                # If for some reason match info missing, find nearest wall again cheaply:
                nearest = None
                for wi2, wid2, wa2, wb2, _wdir2 in wall_data:
                    dist_mid = _point_to_segment_distance(mid, wa2, wb2)
                    if nearest is None or dist_mid < nearest[0]:
                        nearest = (dist_mid, wi2, wid2, wa2, wb2)
                if nearest is None:
                    continue
                _, wi, _, wa, wb = nearest
            else:
                wa, wb, _wdir, _wid = wall_by_index[wi]

            d_mid = _point_to_segment_distance(mid, wa, wb)
            d_e1  = _point_to_segment_distance(p1,  wa, wb)
            d_e2  = _point_to_segment_distance(p2,  wa, wb)
            d_max_end = max(d_e1, d_e2)

            if max_endpoint_wall_distance is not None and d_max_end > float(max_endpoint_wall_distance):
                continue

            nin = int(d.get(inliers_key, 0) or 0)
            key = (float(d_max_end), float(d_mid), -nin)

            # Minimize (max endpoint distance), then minimize midpoint distance, then maximize inliers
            key = (float(d_max_end), float(d_mid), -nin)

            if best_key is None or key < best_key:
                best_key = key
                best_d = d

        if best_d is not None:
            kept.append(best_d)

    return kept

def snap_segments_and_cut_walls(
    segments: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
    threshold: float,
    *,
    endpoints_key: str = "endpoints",
    wall_id_key: str = "wall_id",

    # Matching options
    use_angle: bool = True,
    angle_threshold_deg: float = 10.0,
    angle_weight: float = 0.10,  # meters per radian

    # Cut / validation options
    clamp_to_wall_segment: bool = True,
    min_segment_span: float = 0.10,       # minimum projected span along wall to cut
    min_wall_piece_length: float = 0.05,  # discard tiny leftover wall slivers
    merge_eps: float = 1e-6,              # merges touching/overlapping cut intervals

    # Output / annotation
    keep_unmatched_segments: bool = False,
    annotate_segments: bool = True,
    create_split_wall_ids: bool = True,
    split_wall_id_prefix: str = "w_split_",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Snap arbitrary segments to walls, then cut the matched walls at the snapped segment endpoints.

    Interpretation:
      - Each segment produces a cut interval on its matched wall (t0..t1 along wall).
      - The wall material inside the interval is removed (an "opening"), leaving wall pieces outside.
      - So for one segment on one wall: wall1 < seg1 < seg2 < wall2 -> two wall segments remain.

    Handles multiple segments per wall:
      - Collects all cut intervals per wall, sorts + merges overlaps, then splits wall into remaining pieces.

    Returns:
      new_walls, new_segments
    """

    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _project_point_to_line_param(p: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
        # line: a + t*u, with u unit direction
        return float(np.dot(p - a, u))

    if not walls:
        return [], (segments[:] if keep_unmatched_segments else [])

    # --- Precompute wall geometry ---
    wall_data: List[Tuple[int, Any, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]] = []
    for wi, w in enumerate(walls):
        if endpoints_key not in w:
            continue
        w1, w2 = w[endpoints_key]
        wa = _to_xy(w1)
        wb = _to_xy(w2)
        wid = w.get(wall_id_key, wi)
        wdir = _seg_dir(wa, wb)
        wlen = float(np.linalg.norm(wb - wa))
        if wlen <= 1e-12:
            continue
        wall_data.append((wi, wid, wa, wb, wdir, wlen, w))

    if not wall_data:
        return [], (segments[:] if keep_unmatched_segments else [])

    # Map wall_id -> list of (t0, t1, seg_ref, proj_p1, proj_p2, match_meta)
    per_wall_cuts: Dict[Any, List[Tuple[float, float, Dict[str, Any], np.ndarray, np.ndarray, Dict[str, Any]]]] = {}

    new_segments: List[Dict[str, Any]] = []

    # --- Match + snap each segment and record cut interval ---
    for s in (segments or []):
        if endpoints_key not in s:
            if keep_unmatched_segments:
                new_segments.append(s)
            continue

        p1_raw, p2_raw = s[endpoints_key]
        p1 = _to_xy(p1_raw)
        p2 = _to_xy(p2_raw)
        mid = 0.5 * (p1 + p2)
        sdir = _seg_dir(p1, p2)

        best = None  # (score, dist_mid, ang_rad, ang_deg, wi, wid, wa, wb, wdir, wlen, wall_dict)
        for wi, wid, wa, wb, wdir, wlen, wdict in wall_data:
            dist_mid = _point_to_segment_distance(mid, wa, wb)

            if use_angle:
                ang = _angle_between_dirs(sdir, wdir)  # radians, undirected
                ang_deg = ang * 180.0 / np.pi
                penalty = angle_weight * ang
                if ang_deg > angle_threshold_deg:
                    penalty *= 2.0
                score = dist_mid + penalty
            else:
                ang = 0.0
                ang_deg = 0.0
                score = dist_mid

            if best is None or score < best[0]:
                best = (score, dist_mid, ang, ang_deg, wi, wid, wa, wb, wdir, wlen, wdict)

        if best is None:
            if keep_unmatched_segments:
                new_segments.append(s)
            continue

        score, dist_mid, ang, ang_deg, wi, wid, wa, wb, wdir, wlen, wdict = best

        # hard match reject
        if dist_mid > float(threshold):
            if keep_unmatched_segments:
                new_segments.append(s)
            continue

        # project endpoints onto matched wall line
        t1 = _project_point_to_line_param(p1, wa, wdir)
        t2 = _project_point_to_line_param(p2, wa, wdir)

        if clamp_to_wall_segment:
            t1 = _clamp(t1, 0.0, wlen)
            t2 = _clamp(t2, 0.0, wlen)

        t0, t3 = (t1, t2) if t1 <= t2 else (t2, t1)
        if (t3 - t0) < float(min_segment_span):
            # too small to cut meaningfully
            if keep_unmatched_segments:
                new_segments.append(s)
            continue

        sp1 = wa + t1 * wdir
        sp2 = wa + t2 * wdir

        # annotate + overwrite endpoints to snapped endpoints
        if annotate_segments:
            s["matched_wall_id"] = wid
            s["matched_wall_index"] = wi
            s["matched_wall_distance"] = float(dist_mid)
            s["matched_wall_score"] = float(score)
            s["matched_wall_angle_deg"] = float(ang_deg)
            s["snapped_endpoints"] = (sp1.tolist(), sp2.tolist())
            s["cut_interval_t"] = (float(t0), float(t3))
        # Always snap for downstream consistency:
        s[endpoints_key] = (sp1.tolist(), sp2.tolist())

        # store cut interval for that wall
        meta = {
            "score": float(score),
            "dist_mid": float(dist_mid),
            "angle_deg": float(ang_deg),
            "wall_index": int(wi),
        }
        per_wall_cuts.setdefault(wid, []).append((float(t0), float(t3), s, sp1, sp2, meta))
        new_segments.append(s)

    # --- Split walls by cut intervals ---
    new_walls: List[Dict[str, Any]] = []
    split_counter = 0

    # quick lookup: wall_id -> (wa, wb, wdir, wlen, original_wall_dict)
    wall_by_id: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]] = {}
    for _wi, wid, wa, wb, wdir, wlen, wdict in wall_data:
        wall_by_id[wid] = (wa, wb, wdir, wlen, wdict)

    def _make_wall_like(original: Dict[str, Any], a: np.ndarray, b: np.ndarray, parent_wid: Any) -> Dict[str, Any]:
        nonlocal split_counter
        w2 = dict(original)
        w2[endpoints_key] = (a.tolist(), b.tolist())
        w2["parent_wall_id"] = parent_wid
        if create_split_wall_ids:
            w2[wall_id_key] = f"{split_wall_id_prefix}{parent_wid}_{split_counter}"
            split_counter += 1
        return w2

    # Iterate original walls; if no cuts => keep as-is; if cuts => replace by pieces
    for w in walls:
        if endpoints_key not in w:
            continue
        wid = w.get(wall_id_key, None)
        # If wall has no id, fall back to searching its matching geometry entry index
        # but in your pipeline walls usually have wall_id anyway.
        if wid is None:
            new_walls.append(w)
            continue

        cuts = per_wall_cuts.get(wid, [])
        if not cuts:
            new_walls.append(w)
            continue

        wa, wb, wdir, wlen, original = wall_by_id[wid]

        # sort cuts along wall
        cuts_sorted = sorted(cuts, key=lambda x: (x[0], x[1]))

        # merge overlapping/touching cut intervals
        merged: List[Tuple[float, float]] = []
        for t0, t1, *_ in cuts_sorted:
            if not merged:
                merged.append((t0, t1))
                continue
            m0, m1 = merged[-1]
            if t0 <= m1 + merge_eps:
                merged[-1] = (m0, max(m1, t1))
            else:
                merged.append((t0, t1))

        # Build remaining wall pieces outside merged intervals:
        # [0, m0), (m1, m2), ..., (mk, wlen]
        cursor = 0.0
        for (open0, open1) in merged:
            # piece before opening
            if open0 - cursor >= min_wall_piece_length:
                a = wa + cursor * wdir
                b = wa + open0 * wdir
                if float(np.linalg.norm(b - a)) >= min_wall_piece_length:
                    new_walls.append(_make_wall_like(original, a, b, wid))
            cursor = max(cursor, open1)

        # tail after last opening
        if wlen - cursor >= min_wall_piece_length:
            a = wa + cursor * wdir
            b = wa + wlen * wdir
            if float(np.linalg.norm(b - a)) >= min_wall_piece_length:
                new_walls.append(_make_wall_like(original, a, b, wid))

    return new_walls, new_segments
    


def ransac_line_fitting(x, y, max_iterations, threshold, min_inliers, angle_threshold, wall_post_processing=False, door_post_processing=False):
    '''
    Docstring for ransac_line_fitting of 2 dimensional data (lines)
    
    Tuneable parameters\n:
    :param points: Our data, which are x and y coordinates of wall points.
    :param max_iterations: How many iterations we try, more iterations mean a higher probability of finding a good model. 
    :param threshold: Distance threshold to determine if a point is an inlier to the line model, on the centimeter scale.
    :param min_inliers: The number of close inliers required to assert that the line model is a good fit.
    :param angle_threshold: Maximum angle threshold, beyond which lines are not considered. Meant to enforce walls that are vertical and horizontal.

    returns a list of dictionaries:
    'model': (a, b, c) coefficients of the line ax + by + c = 0
    'inliers': indices of inlier points
    endpoints p1(x1, y1), p2(x2, y2) of the line segment defined by the inliers
    '''

    xy_matrix = np.column_stack([x, y]).astype(np.float64)

    available_points = set(range(len(xy_matrix)))

    models = []

    wall_id = 0

    org_id = 0 # We can use this later if we want to keep track of which original wall segment a split segment came from, for example.

    # Find ALL models.
    while len(available_points) >= min_inliers:
        best_inlier_indices = None
        best_model = None
        best_count = 0
        # Find one model.

        iterations_per_model = max_iterations
        available_points_list = sorted(available_points)
        
        while iterations_per_model > 0:

            # First check if we even have two points to sample left
            if len(available_points) < 2:
                break

            # Randomly sample 2 points to define a line.            
            sample_indices = random.sample(range(len(available_points_list)), 2)

            p1_idx = available_points_list[sample_indices[0]]
            p2_idx = available_points_list[sample_indices[1]]

            #print(f"Sampled points: {xy_matrix[p1][0]}, {xy_matrix[p1][1]} and {xy_matrix[p2][0]}, {xy_matrix[p2][1]}")
            # We only want to fit lines that are horizontal or vertical, according to manhattan world assumption
            if not line_angle(xy_matrix[p1_idx], xy_matrix[p2_idx], angle_threshold):
                iterations_per_model -= 1
                #print("Rejected line due to angle constraint.")
                continue
            
            # Just for simplification and clarity...
            x_1 = xy_matrix[p1_idx][0]
            y_1 = xy_matrix[p1_idx][1]
            x_2 = xy_matrix[p2_idx][0]
            y_2 = xy_matrix[p2_idx][1]

            # Extract line model parameters a, b, c for the line ax + by + c = 0
            a = float(y_2 - y_1)
            b = float(x_1 - x_2)
            c = float((x_2 * y_1) - (x_1 * y_2))

            # Normalize coefficients, which make distance/angle calculations behave correctly further in the program.
            normalized = normalize_coefficients(a, b, c)

            if normalized is None:
                iterations_per_model -= 1
                continue

            (a, b, c) = normalized
            #print(f"Coefficients (a,b,c) after normalization: {a, b, c}")

            tempInliers = []
            # Measure distance from all points to the line, and find inliers
            available_xy = xy_matrix[available_points_list]
            point_to_line_distance = distance_to_line(a, b, c, available_xy[:, 0], available_xy[:, 1])
            #print(f"Point to line distance: {point_to_line_distance}")

            inlier_mask = point_to_line_distance < threshold
            inlier_local_indices = np.where(inlier_mask)[0]
            tempInliers = [available_points_list[i] for i in inlier_local_indices]

            #print(tempInliers)
        
            # Check if this is a good model, it doesnt have to be the best.
            count = int(len(tempInliers))
            if count > best_count:
                best_count = count
                best_inlier_indices = tempInliers # Save the indices of the inliers.
                best_model = (a, b, c) # Save the best model parameters.
                #print(max_iterations)
                #print(f"Found a better model with {best_count} inliers.")
            iterations_per_model -= 1
            
        
        if best_model is None or best_count < min_inliers:
            #print(f"Failed to find a model with sufficient inliers. Best count: {best_count}")
            break

        # Calculate endpoints
        p1, p2 = calculate_endpoints(xy_matrix[best_inlier_indices])

        models.append({
        'original_id': org_id,
        'wall_id': wall_id,
        'model': best_model,
        'inliers': best_inlier_indices,
        'endpoints': (p1, p2),
        'num_inliers': best_count,
        'is_parallel': False}) # False initially...

        # Increment the wall_id variable
        wall_id += 1
        org_id  += 1

        # Remove the inliers from available points
        available_points -= set(best_inlier_indices)
        #print(f"Model {len(models)} added. Removed {best_count} inliers. {len(available_points)} points remaining.")

    parallel_pairs = []
    parallel_groups = []
    if door_post_processing:
        # Parallell Lines check
        models, parallel_pairs, parallel_groups = detect_parallel_lines(models, 0.40, 0.5) # Default: 0.25, 0.5
        # Remove parallel lines
        models = remove_parallel_walls(models, parallel_groups)
    
    if wall_post_processing:
        models, parallel_pairs, parallel_groups = detect_parallel_lines(models, 0.40, 0.5) # Default: 0.25, 0.5
        # Remove parallel lines
        models = remove_parallel_walls(models, parallel_groups)
        # Now detect gaps.
        # Split walls iwhose inlier points are seperarated by a gap > 15 cm
        # along the line direction so that we get two segments.
        
        models = apply_gap_detection(models, xy_matrix, gap_treshold=2.1)

        # Extend walls to intersection point.
        models = extend_wall_to_intersection(models, max_extension=0.5)

        # Join co-linear walls that are close to each other, and have endpoints that are close to each other. This should help with holes in the wall segments.
        models = bridge_collinear_walls(models, separation_threshold=0.50)

    return models, parallel_pairs, parallel_groups 
 
# Relabel walls before plotting so that it looks nicer in the plot.
def relabel_walls(models):
    for i, w in enumerate(models):
        w.setdefault('original_id', w.get('wall_id', i))
        w['wall_id'] = i
    return models

# Plotting utilities for debugging
""" def plot(points, walls, centroid):
    Utility to plot points for debugging.

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1, c="#72a5d1",alpha=0.5)
    plt.scatter(centroid[0], centroid[1], c="red", label="Room Centroid")
    plt.axis("equal")
    plt.title("Wall Points + Detected Walls")
    colors = plt.cm.tab10(np.linspace(0, 1, len(walls)))
    for i, wall in enumerate(walls):
        p1, p2 = wall['endpoints']
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color = colors[i], linewidth=3,
                 label=f"Wall {wall['original_id']} {wall['num_inliers']} pts",
                 zorder=10,
                 linestyle='--')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.2, linestyle='--', color='#cccccc')
    plt.tight_layout()
    plt.show() """

def plot(wall_points, walls, centroid, door_points=None, doors=None, window_points=None, windows=None):
    """Utility to plot wall/door points + detected wall/door segments for debugging.

    Parameters
    ----------
    wall_points : (N,2) array-like
    walls       : list[dict] with 'endpoints', 'original_id', 'num_inliers'
    centroid    : (2,) array-like
    door_points : (M,2) array-like, optional
    doors       : list[dict], optional
    """

    if doors is None:
        doors = []

    plt.figure(figsize=(8, 8))

    # --- Points ---
    wall_points = np.asarray(wall_points)
    plt.scatter(wall_points[:, 0], wall_points[:, 1],
                s=1, alpha=0.35, label="Wall points", color="lightblue")

    if door_points is not None:
        door_points = np.asarray(door_points)
        if door_points.size > 0:
            plt.scatter(door_points[:, 0], door_points[:, 1],
                        s=6, alpha=0.6, label="Door points", color="lightpink")

    if window_points is not None:
        window_points = np.asarray(window_points)
        if window_points.size > 0:
            plt.scatter(window_points[:, 0], window_points[:, 1],
                        s=6, alpha=0.6, label="Window points", color="lightgreen")

    # --- Centroid ---
    plt.scatter(centroid[0], centroid[1], c="red", label="Room Centroid", zorder=20)

    plt.axis("equal")
    plt.title("Wall/Door/Window Points + Detected Segments")

    # --- Segments ---
    #wall_color = "#FFD400"   # bright, easy to see
    door_color = "#FF00FF"   # bright magenta
    window_color = "#00FF00" # bright green

    # Walls
    colors = plt.cm.tab10(np.linspace(0, 1, len(walls)))
    for i, wall in enumerate(walls):
        p1, p2 = wall["endpoints"]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=colors[i], linewidth=3,
                 label=f"Wall {wall.get('wall_id','?')} (orig {wall.get('original_id','?')}) ({wall.get('num_inliers', 0)} pts)",
                 zorder=10, linestyle=":")

    # Doors
    for door in doors:
        p1, p2 = door["endpoints"]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=door_color, linewidth=3,
                 label=f"Door {door.get('original_id', '?')} ({door.get('num_inliers', 0)} pts)",
                 zorder=11, linestyle="-.")
        
    for window in windows:
        p1, p2 = window["endpoints"]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=window_color, linewidth=3,
                label=f"Window {window.get('original_id', '?')} ({window.get('num_inliers', 0)} pts)",
                zorder=11, linestyle="-.")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    # Avoid a huge legend if you have many segments
    handles, labels = plt.gca().get_legend_handles_labels()
    dedup = dict(zip(labels, handles))  # keeps last occurrence
    plt.legend(dedup.values(), dedup.keys(), loc="best", fontsize=9)

    plt.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.show()

# Read x,y from CSV
def load_points(csv_path: str):
    """Load points from a CSV file. Expects columns named x,y,z (case-insensitive)."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    x_col = cols.get("x")
    y_col = cols.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"{csv_path} must contain 'x' and 'y' columns. Found: {list(df.columns)}")
    return df[x_col].to_numpy(), df[y_col].to_numpy()

def main():
    # CSV Paths
    wall_csv   = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_wall_coords_cloud3.csv"
    door_csv   = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_door_coords_cloud3.csv"
    window_csv = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_window_coords_cloud3.csv"

    # Step 1: Load wall data.
    wx, wy = load_points(wall_csv)
    dx, dy = load_points(door_csv)
    vx, vy = load_points(window_csv)

    # Apply RANSAC line fitting to points 
    walls, parallel_pairs, parallel_groups = ransac_line_fitting(
                        wx, # Wall x coordinates
                        wy, # Wall y coordinates
                        max_iterations=500,
                        threshold=0.025,
                        min_inliers=1500,
                        angle_threshold=5.0,
                        wall_post_processing=True,
                        door_post_processing = False
                        )

    doors, _, _ = ransac_line_fitting(
                        dx,
                        dy,
                        max_iterations=2000,
                        threshold=0.04,
                        min_inliers=180,
                        angle_threshold=5.0,
                        wall_post_processing=False,
                        door_post_processing=True)
    
    windows, _, _ = ransac_line_fitting(
                        vx,
                        vy,
                        max_iterations=2000,
                        threshold=0.05,
                        min_inliers=100,
                        angle_threshold=5.0,
                        wall_post_processing=False,
                        door_post_processing=True)

    # Keep only the door which lies closest to a wall.
    doors = filter_door_segments_keep_frame_opening(
                        doors,
                        walls,
                        threshold=0.10,           
                        use_angle=True,
                        angle_threshold_deg=15.0,
                        angle_weight=0.10,
                        max_endpoint_wall_distance=0.02
                    )
    
    new_walls, snapped_doors = snap_segments_and_cut_walls(
    segments=doors,
    walls=walls,
    threshold=0.20,           # tune
    min_segment_span=0.20,    # doors typically > 0.3m
    )
    
    new_walls, snapped_windows = snap_segments_and_cut_walls(
        segments=windows,
        walls=new_walls,
        threshold=0.30,
        min_segment_span=0.20,
    )

    walls = new_walls
    #walls, _, pg = detect_parallel_lines(walls, 0.15, 0.5)
    #walls = remove_parallel_walls(walls, pg)
    doors = snapped_doors
    windows = snapped_windows

    # Cleanup wall ids
    walls = relabel_walls(walls)

    centroid = calculate_centroid(wx, wy)
    # DEBUGGING
    plot(
        np.column_stack((wx, wy)), 
        walls, 
        centroid,
        door_points=np.column_stack((dx, dy)),
        doors=doors, 
        windows=windows, 
        window_points=np.column_stack((vx, vy)))

if __name__ == "__main__":
    main()