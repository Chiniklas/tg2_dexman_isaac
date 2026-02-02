"""
Interactive depth plausibility check using live cameras and saved stereo calibration.

Flow:
- Loads calibration .npz (k1, d1, k2, d2, R, T).
- Starts both cameras via Ov9732 configs.
- Rectifies live frames, computes disparity (SGBM), converts to depth.
- Measures median depth in a center ROI and compares to an expected distance.
- Exits non‑zero if the measured depth is off by more than the given tolerance.

Usage (example):
    python -m tests.depth_plausibility \
        --calib calibration/jetson_stereo.npz \
        --left-config ov9732_L --right-config ov9732_R \
        --expected 0.50 --tolerance 0.25 --frames 30 --show

Place a flat target at the expected distance (e.g., 0.50 m) centered in both views
and keep the rig steady during capture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from stereo_camera.cameras.ov9732_camera import Ov9732Camera
from stereo_camera.utils.capture_two_stream import DualCameraReader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Depth plausibility check using live stereo capture.")
    p.add_argument("--calib", default="tests/calibration/jetson_stereo.npz", help="Stereo calibration npz file.")
    p.add_argument("--left-config", default="ov9732_L", help="Left camera config name or path.")
    p.add_argument("--right-config", default="ov9732_R", help="Right camera config name or path.")
    p.add_argument("--expected", type=float, default=0.5, help="Expected target distance in meters.")
    p.add_argument("--tolerance", type=float, default=0.25, help="Relative tolerance (0.25 = 25% error allowed).")
    p.add_argument("--frames", type=int, default=30, help="Number of frames to sample.")
    p.add_argument("--roi", type=int, default=80, help="ROI size (square, pixels) around the image center.")
    p.add_argument("--flip", choices=["none", "vertical", "horizontal", "both"], default="vertical", help="Optional flip applied to both streams.")
    p.add_argument("--max-fails", type=int, default=5, help="Consecutive read failures before attempting reconnect.")
    p.add_argument("--reconnect-wait", type=float, default=1.0, help="Initial reconnect backoff (seconds).")
    p.add_argument("--show", action="store_true", help="Show disparity preview with ROI.")
    p.add_argument("--show-rgb", action="store_true", help="Also show rectified RGB pair alongside disparity.")
    return p.parse_args()


def build_sgbm(width: int, height: int) -> cv2.StereoSGBM:
    # Disparity settings tuned for 1280x720 @ ~6-8 cm baseline; adjust if needed.
    num_disp = 16 * 8  # must be divisible by 16
    block = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def main() -> int:
    args = parse_args()
    calib_path = Path(args.calib)
    if not calib_path.exists():
        # helpful hint if the user kept calibration under tests/calibration
        alt = Path("tests/calibration/jetson_stereo.npz")
        msg = f"Calibration file not found: {calib_path}"
        if calib_path != alt and alt.exists():
            msg += f"\nTry --calib {alt}"
        raise FileNotFoundError(msg)

    data = np.load(calib_path)
    k1, d1, k2, d2, R, T = (data[x] for x in ("k1", "d1", "k2", "d2", "R", "T"))

    camL = Ov9732Camera.from_config(args.left_config)
    camR = Ov9732Camera.from_config(args.right_config)
    reader = DualCameraReader([camL, camR], flip=args.flip, max_fails=args.max_fails, reconnect_wait=args.reconnect_wait)
    reader.labels = ["L", "R"]
    reader.start()

    try:
        # Grab one frame to get size
        while True:
            pair = reader.read_pair()
            if pair is None:
                cv2.waitKey(1)
                continue
            h, w = pair[0].shape[:2]
            break

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, (w, h), R, T)
        map1x, map1y = cv2.initUndistortRectifyMap(k1, d1, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(k2, d2, R2, P2, (w, h), cv2.CV_32FC1)

        sgbm = build_sgbm(w, h)
        depths_m = []
        roi_half = args.roi // 2
        cx, cy = w // 2, h // 2
        x0, x1 = cx - roi_half, cx + roi_half
        y0, y1 = cy - roi_half, cy + roi_half

        while len(depths_m) < args.frames:
            pair = reader.read_pair()
            if pair is None:
                cv2.waitKey(1)
                continue

            l_raw, r_raw = pair
            l = cv2.remap(l_raw, map1x, map1y, cv2.INTER_LINEAR)
            r = cv2.remap(r_raw, map2x, map2y, cv2.INTER_LINEAR)
            gray_l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            disp = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
            points = cv2.reprojectImageTo3D(disp, Q)

            roi_depth = points[y0:y1, x0:x1, 2]
            roi_disp = disp[y0:y1, x0:x1]
            valid = np.isfinite(roi_depth) & (roi_disp > 0)
            if np.count_nonzero(valid) == 0:
                if args.show:
                    cv2.imshow("disparity", (disp - disp.min()) / (disp.max() - disp.min() + 1e-6))
                    cv2.waitKey(1)
                continue

            depth_m = np.median(roi_depth[valid]) / 1000.0  # mm -> m (assuming square_size_mm was used)
            depths_m.append(depth_m)

            if args.show:
                disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                cv2.rectangle(disp_vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
                cv2.putText(
                    disp_vis,
                    f"depth={depth_m:.3f} m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if args.show_rgb:
                    l_disp = l.copy()
                    r_disp = r.copy()
                    cv2.rectangle(l_disp, (x0, y0), (x1, y1), (0, 255, 0), 1)
                    cv2.rectangle(r_disp, (x0, y0), (x1, y1), (0, 255, 0), 1)
                    rgb_pair = cv2.hconcat([l_disp, r_disp])
                    top = cv2.resize(rgb_pair, (disp_vis.shape[1], disp_vis.shape[0]))
                    composite = cv2.vconcat([top, disp_vis])
                    window_name = "rectified RGB (top) + disparity (bottom)"
                else:
                    composite = disp_vis
                    window_name = "disparity (q to quit)"

                cv2.imshow(window_name, composite)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if args.show:
            cv2.destroyAllWindows()

        if not depths_m:
            print("No valid depth samples collected.")
            return 2

        median_depth = float(np.median(depths_m))
        rel_err = abs(median_depth - args.expected) / args.expected

        print(f"Median depth: {median_depth:.3f} m (expected {args.expected:.3f} m)")
        print(f"Relative error: {rel_err*100:.1f}% (tolerance {args.tolerance*100:.1f}%)")

        if rel_err > args.tolerance:
            print("FAIL: Depth is off beyond tolerance. Recheck square size and baseline.")
            return 1

        print("PASS: Depth within tolerance.")
        return 0

    finally:
        reader.stop()


if __name__ == "__main__":
    sys.exit(main())
