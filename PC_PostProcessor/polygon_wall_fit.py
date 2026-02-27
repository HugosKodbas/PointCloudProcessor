
#!/usr/bin/env python3
"""
polygon_wall_fit_fixed.py

Improved polygon "snake" fitter for wall XY points.

Key fixes vs previous version:
1) Distance field bbox includes BOTH points and initial polygon bbox (prevents OOB at start).
2) Out-of-bounds sampling adds a smooth penalty with a useful gradient instead of silent clamping.
3) Better initialization options:
   - convex hull (default) with simplification
   - AABB
   - PCA OBB
4) Starts with a richer initial polygon (hull) and allows merge to simplify.

Input: CSV with columns x,y (or x,y,z) header optional.
Output: outdir/polygon.json, overlay.png, debug.npz

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations
import argparse, csv, json, math, os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass
class FitConfig:
    # Distance field grid
    cell_size: float = 0.03
    grid_padding: float = 0.75

    # Downsampling
    voxel_size: float = 0.02  # meters; set 0 to disable
    max_points: int = 120000   # cap after downsampling

    # OOB penalty (meters added per meter outside bbox)
    oob_alpha: float = 2.0

    # Sampling
    k_samples_per_edge: int = 30

    # Robust loss
    huber_delta: float = 0.06

    # Regularization
    lambda_smooth: float = 0.10
    lambda_corner: float = 0.03
    lambda_n: float = 0.01
    corner_tol_rad: float = math.radians(8.0)

    # Corner sharpening post-pass (fit lines to edge-supporting points, then intersect)
    snap_corners: bool = True
    snap_band: float = 0.08          # meters: point-to-edge band to collect support points
    snap_min_points: int = 80        # minimum points to fit a line
    snap_max_points: int = 4000      # cap points used per edge fit
    snap_refine_steps: int = 40      # extra optimize steps after snapping

    # Optimization
    max_outer_iters: int = 60
    max_opt_steps: int = 180
    step_size: float = 0.05
    grad_clip: float = 5.0
    eps_move: float = 1e-4

    # Split / Merge
    split_residual_thresh: float = 0.10
    split_min_run: int = 6
    merge_angle_tol_rad: float = math.radians(6.0)
    min_energy_drop: float = 1e-3
    patience: int = 6
    max_vertices: int = 96
    min_vertices: int = 4

    # Init
    init_mode: str = "convex_hull"  # convex_hull | aabb | pca_bbox
    init_simplify_eps: float = 0.06  # meters for hull simplification (RDP)
    max_init_vertices: int = 40

    # Plot
    plot_dpi: int = 170


# -----------------------------
# CSV reader
# -----------------------------
def read_xy_csv(path: str) -> np.ndarray:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    start_idx = 0
    if rows and rows[0] and not is_number(rows[0][0]):
        start_idx = 1

    xs, ys = [], []
    for r in rows[start_idx:]:
        if not r or len(r) < 2:
            continue
        try:
            x = float(r[0]); y = float(r[1])
        except Exception:
            continue
        xs.append(x); ys.append(y)

    if len(xs) < 50:
        raise ValueError(f"Too few points in CSV: {len(xs)}")

    return np.column_stack([xs, ys]).astype(np.float64)



def voxel_downsample(points_xy: np.ndarray, voxel: float, max_points: int) -> np.ndarray:
    """Fast voxel/grid downsample using quantization + unique."""
    if voxel is None or voxel <= 0:
        pts = points_xy
    else:
        q = np.floor(points_xy / voxel).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        pts = points_xy[idx]
    if max_points is not None and len(pts) > max_points:
        # deterministic subsample for reproducibility
        step = max(1, len(pts) // max_points)
        pts = pts[::step][:max_points]
    return pts.astype(np.float64)


# -----------------------------
# Geometry
# -----------------------------
def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly[::-1].copy() if polygon_area(poly) < 0 else poly

def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)

def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12) -> bool:
    o1 = _orient(a, b, c); o2 = _orient(a, b, d)
    o3 = _orient(c, d, a); o4 = _orient(c, d, b)

    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    if abs(o1) <= eps and _on_segment(a, b, c, eps): return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps): return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps): return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps): return True
    return False

def is_simple_polygon(poly: np.ndarray) -> bool:
    n = len(poly)
    if n < 4:
        return False
    for i in range(n):
        a = poly[i]; b = poly[(i+1) % n]
        for j in range(i+1, n):
            if j == i: continue
            if (j+1) % n == i: continue
            if (i+1) % n == j: continue
            c = poly[j]; d = poly[(j+1) % n]
            if segments_intersect(a, b, c, d):
                return False
    return True

def interior_angle(poly: np.ndarray, i: int) -> float:
    n = len(poly)
    v_prev = poly[(i-1) % n] - poly[i]
    v_next = poly[(i+1) % n] - poly[i]
    a = float(np.linalg.norm(v_prev)); b = float(np.linalg.norm(v_next))
    if a < 1e-12 or b < 1e-12:
        return math.pi
    cosang = float(np.clip(np.dot(v_prev, v_next) / (a*b), -1.0, 1.0))
    return float(math.acos(cosang))


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized distance from points (M,2) to segment AB."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = ((points - a[None, :]) @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=1)


def fit_line_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit 2D line by PCA. Returns (point_on_line, unit_direction)."""
    mu = points.mean(axis=0)
    X = points - mu
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)
    d = V[:, int(np.argmax(w))]
    d = d / (np.linalg.norm(d) + 1e-12)
    return mu, d


def intersect_lines(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray, eps: float = 1e-10) -> Optional[np.ndarray]:
    """Intersection of two infinite 2D lines p0+t*d0 and p1+s*d1. Returns None if nearly parallel."""
    A = np.array([[d0[0], -d1[0]], [d0[1], -d1[1]]], dtype=np.float64)
    b = (p1 - p0).astype(np.float64)
    det = float(np.linalg.det(A))
    if abs(det) < eps:
        return None
    t_s = np.linalg.solve(A, b)
    t = float(t_s[0])
    return p0 + t * d0


def snap_polygon_corners(poly: np.ndarray, points_xy: np.ndarray, cfg: FitConfig) -> np.ndarray:
    """Sharpen corners by fitting a supporting line to points near each edge, then intersecting adjacent lines."""
    n = len(poly)
    if n < 4:
        return poly

    # Fit a line for each edge
    line_p = np.zeros((n, 2), dtype=np.float64)
    line_d = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        dists = point_segment_distance(points_xy, a, b)
        idx = np.where(dists <= cfg.snap_band)[0]
        if idx.size > cfg.snap_max_points:
            # subsample deterministically
            step = max(1, idx.size // cfg.snap_max_points)
            idx = idx[::step][:cfg.snap_max_points]
        if idx.size >= cfg.snap_min_points:
            p0, d0 = fit_line_pca(points_xy[idx])
        else:
            # fallback to geometric edge direction
            p0 = 0.5 * (a + b)
            d0 = (b - a)
            d0 = d0 / (np.linalg.norm(d0) + 1e-12)
        line_p[i] = p0
        line_d[i] = d0

    # Intersect consecutive lines to get new vertices
    new_poly = poly.copy()
    for i in range(n):
        # vertex i is intersection of edge (i-1) line and edge i line
        pA, dA = line_p[(i - 1) % n], line_d[(i - 1) % n]
        pB, dB = line_p[i], line_d[i]
        x = intersect_lines(pA, dA, pB, dB)
        if x is None or not np.all(np.isfinite(x)):
            continue
        new_poly[i] = x

    # Keep result only if it stays simple
    if is_simple_polygon(new_poly):
        return ensure_ccw(new_poly)
    return poly


# -----------------------------
# RDP simplification
# -----------------------------
def _perp_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def rdp(poly: np.ndarray, eps: float) -> np.ndarray:
    # poly is open polyline here
    if len(poly) <= 2:
        return poly
    a = poly[0]; b = poly[-1]
    dmax = -1.0; idx = -1
    for i in range(1, len(poly)-1):
        d = _perp_dist(poly[i], a, b)
        if d > dmax:
            dmax = d; idx = i
    if dmax > eps:
        left = rdp(poly[:idx+1], eps)
        right = rdp(poly[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])

def simplify_closed_polygon(poly: np.ndarray, eps: float) -> np.ndarray:
    # Apply RDP on closed polygon by treating as open and rotating start
    n = len(poly)
    if n <= 4:
        return poly
    # choose a stable start: lowest x then y
    start = np.lexsort((poly[:,1], poly[:,0]))[0]
    p = np.roll(poly, -start, axis=0)
    # open by duplicating first at end
    open_poly = np.vstack([p, p[0:1]])
    simp = rdp(open_poly, eps)
    # remove last duplicate
    simp = simp[:-1]
    if len(simp) < 4:
        return poly
    return ensure_ccw(simp)


# -----------------------------
# Distance field with OOB penalty
# -----------------------------
class DistanceField:
    def __init__(self, dt: np.ndarray, origin: np.ndarray, cell_size: float, bbox: np.ndarray, oob_alpha: float):
        self.dt = dt.astype(np.float64)
        self.origin = origin.astype(np.float64)
        self.cs = float(cell_size)
        self.H, self.W = self.dt.shape
        self.bbox = bbox.astype(np.float64)  # [minx,miny,maxx,maxy] in world(local) coords
        self.oob_alpha = float(oob_alpha)

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        gx = (xy[..., 0] - self.origin[0]) / self.cs
        gy = (xy[..., 1] - self.origin[1]) / self.cs
        return np.stack([gy, gx], axis=-1)

    def _bilinear(self, xy: np.ndarray) -> np.ndarray:
        gij = self.world_to_grid(xy)
        y = gij[..., 0]; x = gij[..., 1]
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = x0 + 1; y1 = y0 + 1
        x0c = np.clip(x0, 0, self.W-1); x1c = np.clip(x1, 0, self.W-1)
        y0c = np.clip(y0, 0, self.H-1); y1c = np.clip(y1, 0, self.H-1)

        Ia = self.dt[y0c, x0c]; Ib = self.dt[y0c, x1c]
        Ic = self.dt[y1c, x0c]; Id = self.dt[y1c, x1c]

        wx = (x - x0).astype(np.float64); wy = (y - y0).astype(np.float64)
        wa = (1-wx)*(1-wy); wb = wx*(1-wy); wc = (1-wx)*wy; wd = wx*wy
        return wa*Ia + wb*Ib + wc*Ic + wd*Id

    def oob_distance_and_grad(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # distance to bbox (0 if inside), gradient pointing inward (world coords)
        minx, miny, maxx, maxy = self.bbox
        dx = np.where(xy[...,0] < minx, xy[...,0]-minx, 0.0)
        dx = np.where(xy[...,0] > maxx, xy[...,0]-maxx, dx)
        dy = np.where(xy[...,1] < miny, xy[...,1]-miny, 0.0)
        dy = np.where(xy[...,1] > maxy, xy[...,1]-maxy, dy)

        dist = np.sqrt(dx*dx + dy*dy)
        # grad of dist is (dx/dist, dy/dist) where dist>0
        eps = 1e-12
        inv = 1.0 / (dist + eps)
        gx = dx * inv
        gy = dy * inv
        grad = np.stack([gx, gy], axis=-1)
        # inside bbox => dist=0 => grad ~0
        return dist, grad

    def query(self, xy: np.ndarray) -> np.ndarray:
        base = self._bilinear(xy)
        dist_oob, _ = self.oob_distance_and_grad(xy)
        return base + self.oob_alpha * dist_oob

    def grad(self, xy: np.ndarray) -> np.ndarray:
        # numeric gradient of bilinear part + analytic oob gradient
        xy_xp = xy.copy(); xy_xm = xy.copy()
        xy_yp = xy.copy(); xy_ym = xy.copy()
        xy_xp[...,0] += self.cs; xy_xm[...,0] -= self.cs
        xy_yp[...,1] += self.cs; xy_ym[...,1] -= self.cs
        dDx = (self._bilinear(xy_xp) - self._bilinear(xy_xm)) / (2*self.cs)
        dDy = (self._bilinear(xy_yp) - self._bilinear(xy_ym)) / (2*self.cs)
        g_base = np.stack([dDx, dDy], axis=-1)
        dist_oob, g_oob_dir = self.oob_distance_and_grad(xy)
        g_oob = self.oob_alpha * g_oob_dir
        return g_base + g_oob


def build_distance_field(points_local: np.ndarray, poly_init_local: np.ndarray, cfg: FitConfig) -> DistanceField:
    # Build bbox that includes both points and initial polygon
    min_xy = np.minimum(points_local.min(axis=0), poly_init_local.min(axis=0)) - cfg.grid_padding
    max_xy = np.maximum(points_local.max(axis=0), poly_init_local.max(axis=0)) + cfg.grid_padding
    bbox = np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float64)

    W = int(math.ceil((max_xy[0]-min_xy[0]) / cfg.cell_size)) + 1
    H = int(math.ceil((max_xy[1]-min_xy[1]) / cfg.cell_size)) + 1
    W = max(W, 50); H = max(H, 50)

    occ = np.zeros((H, W), dtype=bool)
    gx = ((points_local[:,0] - min_xy[0]) / cfg.cell_size).astype(int)
    gy = ((points_local[:,1] - min_xy[1]) / cfg.cell_size).astype(int)
    gx = np.clip(gx, 0, W-1); gy = np.clip(gy, 0, H-1)
    occ[gy, gx] = True

    inv = ~occ
    dt_cells = distance_transform_edt(inv).astype(np.float64)
    dt_m = dt_cells * cfg.cell_size
    origin = np.array([min_xy[0], min_xy[1]], dtype=np.float64)

    return DistanceField(dt_m, origin, cfg.cell_size, bbox=bbox, oob_alpha=cfg.oob_alpha)


# -----------------------------
# Initialization
# -----------------------------
def init_polygon(points_local: np.ndarray, cfg: FitConfig) -> np.ndarray:
    mode = cfg.init_mode
    if mode == "aabb":
        mn = points_local.min(axis=0); mx = points_local.max(axis=0)
        poly = np.array([[mn[0], mn[1]],[mx[0], mn[1]],[mx[0], mx[1]],[mn[0], mx[1]]], dtype=np.float64)
        return ensure_ccw(poly)

    if mode == "pca_bbox":
        X = points_local - points_local.mean(axis=0)
        C = np.cov(X.T)
        w, V = np.linalg.eigh(C)
        V = V[:, np.argsort(w)[::-1]]
        R = V
        Xr = X @ R
        mn = Xr.min(axis=0); mx = Xr.max(axis=0)
        corners_r = np.array([[mn[0], mn[1]],[mx[0], mn[1]],[mx[0], mx[1]],[mn[0], mx[1]]], dtype=np.float64)
        corners = corners_r @ R.T + points_local.mean(axis=0)
        return ensure_ccw(corners)

    if mode == "convex_hull":
        hull = ConvexHull(points_local)
        poly = points_local[hull.vertices]
        poly = ensure_ccw(poly)

        # simplify noisy hull
        poly = simplify_closed_polygon(poly, cfg.init_simplify_eps)

        # cap vertices if still too many: uniform subsample
        if len(poly) > cfg.max_init_vertices:
            idx = np.linspace(0, len(poly)-1, cfg.max_init_vertices, dtype=int)
            poly = poly[idx]
            poly = ensure_ccw(poly)

        return poly

    raise ValueError(f"Unknown init_mode: {mode}")


# -----------------------------
# Energy + gradients
# -----------------------------
def huber(r: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=np.float64)
    abs_r = np.abs(r)
    quad = abs_r <= delta
    loss = np.empty_like(r)
    d = np.empty_like(r)
    loss[quad] = 0.5 * (r[quad] ** 2)
    d[quad] = r[quad]
    loss[~quad] = delta * (abs_r[~quad] - 0.5 * delta)
    d[~quad] = delta * np.sign(r[~quad])
    return loss, d


def sample_edge_points(poly: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(poly)
    ts = np.linspace(0.05, 0.95, k, dtype=np.float64)
    samples = np.empty((n*k, 2), dtype=np.float64)
    edge_ids = np.empty((n*k,), dtype=np.int32)
    tvals = np.empty((n*k,), dtype=np.float64)
    idx = 0
    for i in range(n):
        a = poly[i]; b = poly[(i+1) % n]
        pts = a[None,:]*(1-ts[:,None]) + b[None,:]*ts[:,None]
        samples[idx:idx+k] = pts
        edge_ids[idx:idx+k] = i
        tvals[idx:idx+k] = ts
        idx += k
    return samples, edge_ids, tvals


def smoothness_penalty(poly: np.ndarray) -> float:
    n = len(poly)
    acc = 0.0
    for i in range(n):
        dd = poly[(i-1)%n] - 2*poly[i] + poly[(i+1)%n]
        acc += float(np.dot(dd, dd))
    return acc

def corner_penalty(poly: np.ndarray, cfg: FitConfig) -> float:
    n = len(poly)
    total = 0.0
    for i in range(n):
        theta = interior_angle(poly, i)
        c = abs(math.pi - theta)
        # Penalize the *existence* of a corner (to keep corners sparse),
        # but do not strongly penalize sharper corners once they exist.
        # This avoids rounding real corners.
        x = max(0.0, c - cfg.corner_tol_rad)
        # Saturating penalty in [0, 1): small corners ~0, clear corners ~1
        total += 1.0 - math.exp(-8.0 * x)
    return float(total)

def compute_energy(poly: np.ndarray, df: DistanceField, cfg: FitConfig) -> Tuple[float, Dict[str, float]]:
    if not is_simple_polygon(poly):
        return 1e12, {"total":1e12,"data":1e12,"smooth":0.0,"corner":0.0,"n":0.0}
    samples, _, _ = sample_edge_points(poly, cfg.k_samples_per_edge)
    d = df.query(samples)
    l, _ = huber(d, cfg.huber_delta)
    e_data = float(l.sum())
    e_smooth = cfg.lambda_smooth * smoothness_penalty(poly)
    e_corner = cfg.lambda_corner * corner_penalty(poly, cfg)
    e_n = cfg.lambda_n * float(len(poly))
    total = e_data + e_smooth + e_corner + e_n
    return total, {"total":total,"data":e_data,"smooth":e_smooth,"corner":e_corner,"n":e_n}


def compute_energy_and_grad(poly: np.ndarray, df: DistanceField, cfg: FitConfig) -> Tuple[float, np.ndarray]:
    n = len(poly)
    if not is_simple_polygon(poly):
        return 1e12, np.zeros_like(poly)

    samples, edge_ids, tvals = sample_edge_points(poly, cfg.k_samples_per_edge)
    d = df.query(samples)
    loss, dL_dd = huber(d, cfg.huber_delta)
    e_data = float(loss.sum())

    gradD = df.grad(samples)          # (Nk,2)
    g_samples = gradD * dL_dd[:,None] # dL/dx

    g = np.zeros((n,2), dtype=np.float64)
    i0 = edge_ids
    i1 = (edge_ids + 1) % n
    w0 = (1.0 - tvals)[:, None]
    w1 = tvals[:, None]
    np.add.at(g, i0, g_samples * w0)
    np.add.at(g, i1, g_samples * w1)

    # Smoothness gradient
    lam = cfg.lambda_smooth
    dd = np.zeros_like(poly)
    for i in range(n):
        dd[i] = poly[(i-1)%n] - 2*poly[i] + poly[(i+1)%n]
    e_smooth = float(np.sum(dd*dd)) * lam

    g_s = np.zeros_like(poly)
    for i in range(n):
        ddi = dd[i]
        g_s[(i-1)%n] += 2*lam*ddi
        g_s[i] += 2*lam*(-2)*ddi
        g_s[(i+1)%n] += 2*lam*ddi

    e_corner = cfg.lambda_corner * corner_penalty(poly, cfg)
    e_n = cfg.lambda_n * float(n)
    total = e_data + e_smooth + e_corner + e_n

    return total, (g + g_s)


def optimize_vertices(poly: np.ndarray, df: DistanceField, cfg: FitConfig) -> np.ndarray:
    poly = poly.copy()
    best = poly.copy()
    best_E, _ = compute_energy(best, df, cfg)

    for _ in range(cfg.max_opt_steps):
        E, grad = compute_energy_and_grad(poly, df, cfg)
        if not np.isfinite(E):
            break

        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-12:
            break
        if gnorm > cfg.grad_clip:
            grad = grad * (cfg.grad_clip / (gnorm + 1e-12))

        if gnorm * cfg.step_size < cfg.eps_move:
            break

        step = cfg.step_size
        improved = False
        for _ls in range(12):
            cand = poly - step * grad
            if not is_simple_polygon(cand):
                step *= 0.5
                continue
            cand_E, _ = compute_energy(cand, df, cfg)
            if cand_E <= E:
                poly = cand
                improved = True
                break
            step *= 0.5

        if not improved:
            break

        if E < best_E - 1e-9:
            best_E = E
            best = poly.copy()

    return best


# -----------------------------
# Split / Merge
# -----------------------------
def apply_split(poly: np.ndarray, edge_idx: int, new_pt: np.ndarray) -> np.ndarray:
    return np.insert(poly, edge_idx+1, new_pt, axis=0)

def apply_merge(poly: np.ndarray, vertex_idx: int) -> np.ndarray:
    return np.delete(poly, vertex_idx, axis=0)

def propose_splits(poly: np.ndarray, df: DistanceField, cfg: FitConfig) -> List[Tuple[int, np.ndarray, float]]:
    n = len(poly)
    proposals = []
    k = max(cfg.k_samples_per_edge*2, 40)
    ts = np.linspace(0.02, 0.98, k, dtype=np.float64)

    for i in range(n):
        a = poly[i]; b = poly[(i+1)%n]
        pts = a[None,:]*(1-ts[:,None]) + b[None,:]*ts[:,None]
        r = df.query(pts)

        mask = r > cfg.split_residual_thresh
        if mask.sum() < cfg.split_min_run:
            continue

        # run detection
        run = 0; start = 0
        best = (None, None, -1.0)
        for j in range(k):
            if mask[j]:
                if run == 0: start = j
                run += 1
            else:
                if run >= cfg.split_min_run:
                    mx = float(r[start:j].max())
                    if mx > best[2]:
                        best = (start, j, mx)
                run = 0
        if run >= cfg.split_min_run:
            mx = float(r[start:k].max())
            if mx > best[2]:
                best = (start, k, mx)

        if best[0] is None:
            continue

        s,e,mx = best
        j_star = int(s + np.argmax(r[s:e]))
        new_pt = pts[j_star].copy()

        score = mx * float(np.linalg.norm(b-a))
        proposals.append((i, new_pt, score))

    proposals.sort(key=lambda x: x[2], reverse=True)
    return proposals

def propose_merges(poly: np.ndarray, cfg: FitConfig) -> List[Tuple[int, float]]:
    n = len(poly)
    props = []
    for i in range(n):
        theta = interior_angle(poly, i)
        c = abs(math.pi - theta)
        if c < cfg.merge_angle_tol_rad:
            props.append((i, float(cfg.merge_angle_tol_rad - c)))
    props.sort(key=lambda x: x[1], reverse=True)
    return props

def try_best_edit(poly: np.ndarray, df: DistanceField, cfg: FitConfig) -> Tuple[np.ndarray, bool]:
    base_E, _ = compute_energy(poly, df, cfg)
    best_poly = poly
    best_E = base_E

    # Try a handful of best splits/merges and pick best improvement
    candidates: List[np.ndarray] = []

    if len(poly) < cfg.max_vertices:
        for edge_idx, new_pt, _ in propose_splits(poly, df, cfg)[:12]:
            cand = apply_split(poly, edge_idx, new_pt)
            if len(cand) > cfg.max_vertices or not is_simple_polygon(cand):
                continue
            candidates.append(cand)

    if len(poly) > cfg.min_vertices:
        for vidx, _ in propose_merges(poly, cfg)[:18]:
            cand = apply_merge(poly, vidx)
            if len(cand) < cfg.min_vertices or not is_simple_polygon(cand):
                continue
            candidates.append(cand)

    changed = False
    for cand in candidates:
        cand = optimize_vertices(cand, df, cfg)
        E, _ = compute_energy(cand, df, cfg)
        if E < best_E - cfg.min_energy_drop:
            best_E = E
            best_poly = cand
            changed = True

    return best_poly, changed


# -----------------------------
# Fit + output
# -----------------------------
def save_polygon_json(poly_xy: np.ndarray, out_path: str) -> None:
    data = [{"x": float(x), "y": float(y)} for x,y in poly_xy]
    with open(out_path, "w") as f:
        json.dump({"polygon": data}, f, indent=2)

def save_overlay_plot(points_xy: np.ndarray, init_poly: np.ndarray, poly: np.ndarray, out_path: str, title: str, cfg: FitConfig) -> None:
    plt.figure(figsize=(9,9))
    plt.scatter(points_xy[:,0], points_xy[:,1], s=1, alpha=0.25)
    def draw(p, style, lw, label):
        q = np.vstack([p, p[0:1]])
        plt.plot(q[:,0], q[:,1], style, linewidth=lw, label=label)
        plt.scatter(p[:,0], p[:,1], s=18)
    draw(init_poly, "--", 1.2, "init")
    draw(poly, "-", 2.0, "final")
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.plot_dpi)
    plt.close()

def fit_room_polygon(points_xy: np.ndarray, cfg: FitConfig) -> Tuple[np.ndarray, Dict[str, List[float]], np.ndarray]:
    # Shift to local coords
    shift = points_xy.mean(axis=0)
    pts_local = points_xy - shift

    poly0 = init_polygon(pts_local, cfg)
    poly0 = ensure_ccw(poly0)
    if not is_simple_polygon(poly0):
        # fallback
        fallback = FitConfig(**{**asdict(cfg), "init_mode": "aabb"})
        poly0 = init_polygon(pts_local, fallback)
        poly0 = ensure_ccw(poly0)

    # Build DF with bbox union (fix #1)
    df = build_distance_field(pts_local, poly0, cfg)

    poly = poly0.copy()
    best = poly.copy()
    best_E, _ = compute_energy(best, df, cfg)
    no_edit = 0

    history = {"E_total":[], "E_data":[], "E_smooth":[], "E_corner":[], "E_n":[], "N":[]}

    for outer in range(cfg.max_outer_iters):
        poly = optimize_vertices(poly, df, cfg)
        E, parts = compute_energy(poly, df, cfg)

        history["E_total"].append(parts["total"])
        history["E_data"].append(parts["data"])
        history["E_smooth"].append(parts["smooth"])
        history["E_corner"].append(parts["corner"])
        history["E_n"].append(parts["n"])
        history["N"].append(float(len(poly)))

        if E < best_E:
            best_E = E
            best = poly.copy()

        poly2, changed = try_best_edit(poly, df, cfg)
        if changed:
            poly = poly2
            no_edit = 0
        else:
            no_edit += 1
            if no_edit >= cfg.patience:
                break

    # Corner sharpening post-pass (keeps code changes localized)
    if cfg.snap_corners:
        best_snapped = snap_polygon_corners(best, pts_local, cfg)
        if not np.allclose(best_snapped, best):
            # A short refinement with reduced smoothing helps settle edges without rounding corners
            cfg_ref = FitConfig(**{**asdict(cfg)})
            cfg_ref.lambda_smooth = cfg.lambda_smooth * 0.3
            cfg_ref.max_opt_steps = int(cfg.snap_refine_steps)
            best_snapped = optimize_vertices(best_snapped, df, cfg_ref)
            best_E2, _ = compute_energy(best_snapped, df, cfg)
            if best_E2 < best_E + 1e-6 and is_simple_polygon(best_snapped):
                best = best_snapped

    # Back to world coords
    return best + shift, history, poly0 + shift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cell_size", type=float, default=0.03)
    ap.add_argument("--init_mode", choices=["convex_hull","aabb","pca_bbox"], default="convex_hull")
    ap.add_argument("--voxel_size", type=float, default=0.02, help="Downsample grid size in meters (0 disables).")
    ap.add_argument("--max_points", type=int, default=120000, help="Max points after downsampling.")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pts = read_xy_csv(args.input)

    cfg = FitConfig(cell_size=args.cell_size, init_mode=args.init_mode,
                    voxel_size=args.voxel_size, max_points=args.max_points)

    pts_ds = voxel_downsample(pts, cfg.voxel_size, cfg.max_points)

    poly, hist, init_poly = fit_room_polygon(pts_ds, cfg)

    out_poly = os.path.join(args.outdir, "polygon.json")
    out_png = os.path.join(args.outdir, "overlay.png")
    out_npz = os.path.join(args.outdir, "debug.npz")

    save_polygon_json(poly, out_poly)

    E_final = hist["E_total"][-1] if hist["E_total"] else float("nan")
    title = f"Fitted polygon (N={len(poly)}), E={E_final:.3f}"
    save_overlay_plot(pts_ds, init_poly, poly, out_png, title, cfg)

    np.savez(out_npz, points_xy=pts_ds, polygon_xy=poly, init_polygon_xy=init_poly,
             **{k: np.array(v, dtype=np.float64) for k,v in hist.items()})

    print("Done.")
    print(" polygon:", out_poly)
    print(" overlay:", out_png)
    print(" debug:", out_npz)


if __name__ == "__main__":
    main()
