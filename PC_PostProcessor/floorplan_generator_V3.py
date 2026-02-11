import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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
def line_angle(p1, p2, max_angle=5.0):
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
    return abs(a * x0 + b * y0 + c) / math.hypot(a, b)

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

    best_inlier_indices = None

    models = []

    best_model = None

    best_count = 0

    while max_iterations > 0:
        # Randomly sample 2 points to define a line.
        p1, p2 = random.sample(range(len(xy_matrix)), 2)
        tempInliers = []
        tempModel = None
        #print(f"Sampled points: {xy_matrix[p1][0]}, {xy_matrix[p1][1]} and {xy_matrix[p2][0]}, {xy_matrix[p2][1]}")
        # We only want to fit lines that are horizontal or vertical, according to manhattan world assumption
        if  not line_angle(xy_matrix[p1], xy_matrix[p2]):
            max_iterations -= 1
            #print("Rejected line due to angle constraint.")
            continue
        
        # Just for simplification and clarity...
        x_1 = xy_matrix[p1][0]
        y_1 = xy_matrix[p1][1]
        x_2 = xy_matrix[p2][0]
        y_2 = xy_matrix[p2][1]

        # Extract line model parameters a, b, c for the line ax + by + c = 0
        a = float(y_2 - y_1)
        b = float(x_1 - x_2)
        c = float((x_2 * y_1) - (x_1 * y_2))

        tempModel = (a, b, c)

        # Measure distance from all points to the line, and find inliers
        point_to_line_distance = distance_to_line(a, b, c, xy_matrix[:, 0], xy_matrix[:, 1])
        #print(f"Point to line distance: {point_to_line_distance}")
        for indx, dist in enumerate(point_to_line_distance):
            if dist < threshold:
                #print(f"Found inlier with distance: {dist} and index: {indx}")
                tempInliers.append(indx)

        #print(tempInliers)
      
        # Check if this is a good model, it doesnt have to be the best.
        count = int(len(tempInliers))
        if count > best_count:
            best_count = count
            best_inlier_indices = tempInliers # Save the indices of the inliers.
            best_model = tempModel # Save the best model parameters.
            print(max_iterations)
            print(f"Found a better model with {best_count} inliers.")
            max_iterations -= 1
        
    
        if best_model is None or best_count < min_inliers:
            print(f"Failed to find a model with sufficient inliers. Best count: {best_count}")
            return None
        
        models.append({
        'model': best_model,
        'inliers': best_inlier_indices})

        print(best_inlier_indices[:10]) 
        # TODO FIX REMOVING PROCESSED POINTS
        xy_matrix = np.delete(xy_matrix, best_inlier_indices, axis=0) # Remove inliers from the dataset to find new lines.
    
    print(max_iterations)
    return models
 # Get the inlier points.
    

# Plotting utilities for debugging
def plot_points(points):
    """Utility to plot points for debugging."""
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    plt.axis("equal")
    plt.title("Wall Points")
    plt.xlabel("X")
    plt.ylabel("Y")
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
    
    # DEBUGGING
    plot_points(np.column_stack((wx, wy)))

    # Apply RANSAC line fitting to points 
    walls = ransac_line_fitting(wx, # Wall x coordinates
                        wy, # Wall y coordinates
                        max_iterations=1000,
                        threshold=0.05,
                        min_inliers=1000)
    print(f"Detected {len(walls)} wall(s).")
if __name__ == "__main__":
    main()