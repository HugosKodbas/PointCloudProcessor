#!/usr/bin/env python3
"""
compare_floorplans.py
=====================
Compares ground-truth and predicted floor plan JSONs produced by
floorplan_generator_V3.py.

Pairs files by room name:
    <room>_floorplan_gt.json  ↔  <room>_floorplan_pred.json

Computes:
  - Per-segment length error (absolute and relative)
  - Per-segment 1D IoU (projected overlap along wall direction)
  - Room-level 2D IoU (rasterized pixel overlap of all segments)
  - Detection counts (matched / missed / extra)
  - Aggregate statistics across all rooms

Usage:
    python compare_floorplans.py ./output
    python compare_floorplans.py ./output --max_match_dist 1.0 --resolution 0.01
    python compare_floorplans.py ./output --save_csv results.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def midpoint(elem: dict) -> np.ndarray:
    p1, p2 = elem["endpoints"]
    return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])


def seg_direction(elem: dict) -> np.ndarray:
    p1, p2 = elem["endpoints"]
    d = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float64)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.array([1.0, 0.0])


def project_to_1d(elem: dict, origin: np.ndarray, direction: np.ndarray) -> Tuple[float, float]:
    """Project a segment's endpoints onto a 1D axis defined by origin + t*direction."""
    p1, p2 = elem["endpoints"]
    t1 = float(np.dot(np.array(p1) - origin, direction))
    t2 = float(np.dot(np.array(p2) - origin, direction))
    return (min(t1, t2), max(t1, t2))


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """IoU of two 1D intervals [a0, a1] and [b0, b1]."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 1e-12 else 0.0


def segment_iou(gt_elem: dict, pred_elem: dict) -> float:
    """
    Compute 1D IoU between a GT and pred segment.
    Projects both onto the GT segment's direction.
    """
    p1_gt = np.array(gt_elem["endpoints"][0], dtype=np.float64)
    direction = seg_direction(gt_elem)

    gt_t0, gt_t1 = project_to_1d(gt_elem, p1_gt, direction)
    pr_t0, pr_t1 = project_to_1d(pred_elem, p1_gt, direction)

    return iou_1d(gt_t0, gt_t1, pr_t0, pr_t1)


# ──────────────────────────────────────────────────────────────────────────────
# ROOM-LEVEL 2D IoU (rasterized)
# ──────────────────────────────────────────────────────────────────────────────

def rasterize_segments(elements: dict, resolution: float = 0.01) -> Tuple[np.ndarray, float, float]:
    """
    Rasterize all wall/door/window segments onto a 2D binary grid.
    Returns (grid, x_min, y_min) so both GT and pred use the same coordinate frame.
    """
    # Collect all endpoints to determine bounding box
    all_pts = []
    for elem_type in ["walls", "doors", "windows"]:
        for e in elements.get(elem_type, []):
            all_pts.extend(e["endpoints"])

    if not all_pts:
        return np.zeros((1, 1), dtype=bool), 0.0, 0.0

    pts = np.array(all_pts)
    return pts, None, None  # placeholder — actual rasterization below


def room_iou_2d(gt_data: dict, pred_data: dict, resolution: float = 0.01) -> float:
    """
    Compute 2D pixel-level IoU between GT and pred floor plans.
    Rasterizes all segments (walls, doors, windows) from both onto a shared grid.
    """
    # Collect all endpoints for shared bounding box
    all_pts = []
    for data in [gt_data, pred_data]:
        for elem_type in ["walls", "doors", "windows"]:
            for e in data["elements"].get(elem_type, []):
                all_pts.extend(e["endpoints"])

    if not all_pts:
        return 0.0

    pts = np.array(all_pts)
    x_min, y_min = pts.min(axis=0) - 0.05
    x_max, y_max = pts.max(axis=0) + 0.05

    nx = int(np.ceil((x_max - x_min) / resolution)) + 1
    ny = int(np.ceil((y_max - y_min) / resolution)) + 1

    def draw(data: dict) -> np.ndarray:
        grid = np.zeros((nx, ny), dtype=bool)
        for elem_type in ["walls", "doors", "windows"]:
            for e in data["elements"].get(elem_type, []):
                p1 = np.array(e["endpoints"][0], dtype=np.float64)
                p2 = np.array(e["endpoints"][1], dtype=np.float64)
                seg_len = np.linalg.norm(p2 - p1)
                if seg_len < 1e-12:
                    continue
                # Walk along segment at sub-pixel steps
                n_steps = max(int(seg_len / resolution * 2), 2)
                for t in np.linspace(0, 1, n_steps):
                    p = p1 + t * (p2 - p1)
                    xi = int((p[0] - x_min) / resolution)
                    yi = int((p[1] - y_min) / resolution)
                    if 0 <= xi < nx and 0 <= yi < ny:
                        grid[xi, yi] = True
        return grid

    gt_grid = draw(gt_data)
    pred_grid = draw(pred_data)

    intersection = np.logical_and(gt_grid, pred_grid).sum()
    union = np.logical_or(gt_grid, pred_grid).sum()

    return float(intersection) / float(union) if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ELEMENT MATCHING
# ──────────────────────────────────────────────────────────────────────────────

def match_elements(gt_elems: list, pred_elems: list, max_dist: float) -> dict:
    """
    Match pred elements to GT by nearest midpoint. Only accepts matches
    within max_dist meters. Returns dict with matches, unmatched GT, unmatched pred.
    """
    used_gt = set()
    matches = []

    for pe in pred_elems:
        pm = midpoint(pe)
        best_dist = float("inf")
        best_idx = None
        for i, ge in enumerate(gt_elems):
            if i in used_gt:
                continue
            d = float(np.linalg.norm(pm - midpoint(ge)))
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_dist <= max_dist:
            ge = gt_elems[best_idx]
            length_err = abs(pe["length"] - ge["length"])
            rel_err = 100 * length_err / ge["length"] if ge["length"] > 1e-6 else 0.0
            seg_iou = segment_iou(ge, pe)

            matches.append({
                "gt": ge,
                "pred": pe,
                "midpoint_dist": best_dist,
                "length_error": length_err,
                "rel_error": rel_err,
                "segment_iou": seg_iou,
            })
            used_gt.add(best_idx)

    missed = [gt_elems[i] for i in range(len(gt_elems)) if i not in used_gt]
    matched_pred_ids = {id(m["pred"]) for m in matches}
    extra = [pe for pe in pred_elems if id(pe) not in matched_pred_ids]

    return {"matches": matches, "missed": missed, "extra": extra}


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-ROOM COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def compare_room(gt_data: dict, pred_data: dict, max_dist: float, resolution: float) -> dict:
    """Compare one GT/pred pair. Returns structured results."""
    result = {"elements": {}}

    for elem_type in ["walls", "doors", "windows"]:
        gt_e = gt_data["elements"].get(elem_type, [])
        pr_e = pred_data["elements"].get(elem_type, [])
        matched = match_elements(gt_e, pr_e, max_dist)
        result["elements"][elem_type] = {
            "gt_count": len(gt_e),
            "pred_count": len(pr_e),
            "matched": len(matched["matches"]),
            "missed": len(matched["missed"]),
            "extra": len(matched["extra"]),
            "matches": matched["matches"],
        }

    # Room-level IoU
    result["room_iou"] = room_iou_2d(gt_data, pred_data, resolution=resolution)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

def discover_pairs(input_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find GT/pred JSON pairs. Returns list of (room_name, gt_path, pred_path).
    """
    gt_suffix = "_floorplan_gt.json"
    pred_suffix = "_floorplan_pred.json"

    gt_files = sorted(Path(input_dir).glob(f"*{gt_suffix}"))
    pairs = []

    for gt_path in gt_files:
        room_name = gt_path.name.replace(gt_suffix, "")
        pred_path = gt_path.parent / f"{room_name}{pred_suffix}"
        if pred_path.exists():
            pairs.append((room_name, str(gt_path), str(pred_path)))
        else:
            print(f"WARNING: no pred file for {room_name}, skipping")

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def print_room_report(room_name: str, result: dict, verbose: bool = False):
    print(f"\n{'─'*70}")
    print(f"  {room_name}    Room IoU: {result['room_iou']:.3f}")
    print(f"{'─'*70}")

    for etype in ["walls", "doors", "windows"]:
        r = result["elements"][etype]
        print(f"  {etype.upper():8s}  GT={r['gt_count']}  Pred={r['pred_count']}  "
              f"Matched={r['matched']}  Missed={r['missed']}  Extra={r['extra']}", end="")

        if r["matches"]:
            errs = [m["length_error"] for m in r["matches"]]
            ious = [m["segment_iou"] for m in r["matches"]]
            print(f"  |  Δ_mean={np.mean(errs)*100:.1f}cm  "
                  f"Δ_med={np.median(errs)*100:.1f}cm  "
                  f"IoU_mean={np.mean(ious):.3f}")
        else:
            print()

        if verbose:
            for m in r["matches"]:
                g, p = m["gt"], m["pred"]
                print(f"    GT id={g['id']} len={g['length']:.3f}m  ↔  "
                      f"Pred id={p['id']} len={p['length']:.3f}m  |  "
                      f"Δ={m['length_error']*100:.1f}cm ({m['rel_error']:.1f}%)  "
                      f"IoU={m['segment_iou']:.3f}  dist={m['midpoint_dist']:.3f}m")


def print_aggregate(all_results: List[Tuple[str, dict]]):
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({len(all_results)} rooms)")
    print(f"{'='*70}")

    # Per-element-type aggregation
    for etype in ["walls", "doors", "windows"]:
        all_errs = []
        all_ious = []
        total_gt = total_pred = total_matched = total_missed = total_extra = 0

        for _, result in all_results:
            r = result["elements"][etype]
            total_gt += r["gt_count"]
            total_pred += r["pred_count"]
            total_matched += r["matched"]
            total_missed += r["missed"]
            total_extra += r["extra"]
            for m in r["matches"]:
                all_errs.append(m["length_error"])
                all_ious.append(m["segment_iou"])

        print(f"\n  {etype.upper()}")
        print(f"    Detection:  GT={total_gt}  Pred={total_pred}  "
              f"Matched={total_matched}  Missed={total_missed}  Extra={total_extra}")

        if all_errs:
            errs = np.array(all_errs)
            ious = np.array(all_ious)
            print(f"    Length error (cm):  mean={errs.mean()*100:.1f}  "
                  f"median={np.median(errs)*100:.1f}  "
                  f"std={errs.std()*100:.1f}  "
                  f"max={errs.max()*100:.1f}")
            print(f"    Segment IoU:       mean={ious.mean():.3f}  "
                  f"median={np.median(ious):.3f}  "
                  f"min={ious.min():.3f}")
        else:
            print(f"    (no matched segments)")

    # Room-level IoU
    room_ious = [result["room_iou"] for _, result in all_results]
    print(f"\n  ROOM-LEVEL IoU")
    print(f"    mean={np.mean(room_ious):.3f}  "
          f"median={np.median(room_ious):.3f}  "
          f"min={np.min(room_ious):.3f}  "
          f"max={np.max(room_ious):.3f}")

    # Overall length error
    all_errs = []
    all_ious = []
    for _, result in all_results:
        for etype in ["walls", "doors", "windows"]:
            for m in result["elements"][etype]["matches"]:
                all_errs.append(m["length_error"])
                all_ious.append(m["segment_iou"])

    if all_errs:
        errs = np.array(all_errs)
        ious = np.array(all_ious)
        print(f"\n  ALL ELEMENTS COMBINED ({len(errs)} matched pairs)")
        print(f"    Length error (cm):  mean={errs.mean()*100:.1f}  "
              f"median={np.median(errs)*100:.1f}  "
              f"std={errs.std()*100:.1f}  "
              f"max={errs.max()*100:.1f}")
        print(f"    Segment IoU:       mean={ious.mean():.3f}  "
              f"median={np.median(ious):.3f}  "
              f"min={ious.min():.3f}")


def save_csv(all_results: List[Tuple[str, dict]], csv_path: str):
    """Save per-room summary as CSV for further analysis / plotting."""
    import csv

    rows = []
    for room_name, result in all_results:
        row = {"room": room_name, "room_iou": result["room_iou"]}
        for etype in ["walls", "doors", "windows"]:
            r = result["elements"][etype]
            errs = [m["length_error"] for m in r["matches"]]
            ious = [m["segment_iou"] for m in r["matches"]]
            row[f"{etype}_gt"] = r["gt_count"]
            row[f"{etype}_pred"] = r["pred_count"]
            row[f"{etype}_matched"] = r["matched"]
            row[f"{etype}_missed"] = r["missed"]
            row[f"{etype}_extra"] = r["extra"]
            row[f"{etype}_err_mean_cm"] = round(np.mean(errs) * 100, 2) if errs else None
            row[f"{etype}_err_median_cm"] = round(np.median(errs) * 100, 2) if errs else None
            row[f"{etype}_iou_mean"] = round(np.mean(ious), 4) if ious else None
        rows.append(row)

    fieldnames = rows[0].keys() if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-room CSV → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare GT vs predicted floor plan JSONs."
    )
    parser.add_argument("input_dir",
                        help="Directory containing *_floorplan_gt.json and *_floorplan_pred.json files")
    parser.add_argument("--max_match_dist", type=float, default=1.0,
                        help="Max midpoint distance (m) to accept a segment match (default: 1.0)")
    parser.add_argument("--resolution", type=float, default=0.01,
                        help="Rasterization resolution in meters for room IoU (default: 0.01)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-segment match details for each room")
    parser.add_argument("--save_csv", default=None,
                        help="Save per-room summary to a CSV file")
    args = parser.parse_args()

    pairs = discover_pairs(args.input_dir)
    if not pairs:
        print(f"No GT/pred pairs found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} GT/pred pairs in {args.input_dir}")
    print(f"Max match distance: {args.max_match_dist}m")
    print(f"Rasterization resolution: {args.resolution}m")

    all_results: List[Tuple[str, dict]] = []

    for room_name, gt_path, pred_path in pairs:
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(pred_path, encoding="utf-8") as f:
            pred_data = json.load(f)

        result = compare_room(gt_data, pred_data, args.max_match_dist, args.resolution)
        all_results.append((room_name, result))
        print_room_report(room_name, result, verbose=args.verbose)

    print_aggregate(all_results)

    if args.save_csv:
        save_csv(all_results, args.save_csv)


if __name__ == "__main__":
    main()