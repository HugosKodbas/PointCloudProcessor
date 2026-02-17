import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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

    print("Parallel pairs (wall_id_i, wall_id_j, distance, dtheta):", parallel_pairs)
    print("Parallel groups:", parallel_groups)

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
def line_angle(p1, p2, max_angle=0.25):
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

def ransac_line_fitting(wx, wy, max_iterations, threshold, min_inliers):
    '''
    Docstring for ransac_line_fitting of 2 dimensional data (lines)
    
    Tuneable parameters\n:
    :param points: Our data, which are x and y coordinates of wall points.
    :param max_iterations: How many iterations we try, more iterations mean a higher probability of finding a good model. 
    :param threshold: Distance threshold to determine if a point is an inlier to the line model, on the centimeter scale.
    :param min_inliers: The number of close inliers required to assert that the line model is a good fit.

    returns a list of dictionaries:
    'model': (a, b, c) coefficients of the line ax + by + c = 0
    'inliers': indices of inlier points
    endpoints p1(x1, y1), p2(x2, y2) of the line segment defined by the inliers
    '''

    xy_matrix = np.column_stack([wx, wy]).astype(np.float64)

    available_points = set(range(len(xy_matrix)))

    models = []

    wall_id = 0

    # Find ALL models.
    while len(available_points) >= min_inliers:
        best_inlier_indices = None
        best_model = None
        best_count = 0
        # Find one model.

        iterations_per_model = max_iterations
        available_points_list = list(available_points)
        
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
            if  not line_angle(xy_matrix[p1_idx], xy_matrix[p2_idx]):
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
            print(f"Failed to find a model with sufficient inliers. Best count: {best_count}")
            break

        # Calculate endpoints
        p1, p2 = calculate_endpoints(xy_matrix[best_inlier_indices])

        models.append({
        'wall_id': wall_id,
        'model': best_model,
        'inliers': best_inlier_indices,
        'endpoints': (p1, p2),
        'num_inliers': best_count,
        'is_parallel': False}) # False initially...

        # Increment the wall_id variable
        wall_id += 1

        # Remove the inliers from available points
        available_points -= set(best_inlier_indices)
        #print(f"Model {len(models)} added. Removed {best_count} inliers. {len(available_points)} points remaining.")

    # Parallell Lines check
    models, parallel_pairs, parallel_groups = detect_parallel_lines(models, 0.25, 0.5)

    models = remove_parallel_walls(models, parallel_groups)

    return models, parallel_pairs, parallel_groups 
 # Get the inlier points.
    

# Plotting utilities for debugging
def plot(points, walls, centroid):
    """Utility to plot points for debugging."""

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
                 label=f"Wall {wall['wall_id']} {wall['num_inliers']} pts)",
                 zorder=10,
                 linestyle='--')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.2, linestyle='--', color='#cccccc')
    plt.tight_layout()
    plt.show()

# Read x,y from CSV
def load_wall_points(csv_path: str):
    """Load wall points from a CSV file. Expects columns named x,y,z (case-insensitive)."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    x_col = cols.get("x")
    y_col = cols.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"{csv_path} must contain 'x' and 'y' columns. Found: {list(df.columns)}")
    return df[x_col].to_numpy(), df[y_col].to_numpy()

def main():
    # CSV Paths
    wall_csv   = "/home/hugni/PC_Processor/PC_PostProcessor/CSV_Predictions/pred_wall_coords_cloud0.csv"
    
    # Step 1: Load wall data.
    wx, wy = load_wall_points(wall_csv)
    print(f"Loaded X: {wx[::-10]}, Y: {wy[::-10]}")

    # Apply RANSAC line fitting to points 
    walls, parallel_pairs, parallel_groups = ransac_line_fitting(wx, # Wall x coordinates
                        wy, # Wall y coordinates
                        max_iterations=500,
                        threshold=0.05,
                        min_inliers=1250)
    #print(f'Detected {len(walls)} wall(s). {walls[0]["model"]}')

    print(f"Groups of parallel segments: {parallel_groups}")

    #detect_parallel_lines(walls, 0.25, 0.5)
    for wall in range(len(walls)):
        print(f"Wall: {walls[wall]['wall_id']} is parallel {walls[wall]['is_parallel']}")

    centroid = calculate_centroid(wx, wy)

    # DEBUGGING
    plot(np.column_stack((wx, wy)), walls, centroid)

if __name__ == "__main__":
    main()