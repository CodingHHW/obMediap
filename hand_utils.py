import time
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None


HAND_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class Hand3D:
    world_xyz: np.ndarray  # (21, 3) in meters, camera coords
    handedness: str


def download_if_needed(url: str, dst_path: pathlib.Path) -> pathlib.Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_path.exists():
        print(f"[INFO] 下载模型到 {dst_path} ...", flush=True)
        urllib.request.urlretrieve(url, dst_path)
    return dst_path


class HandDetector:
    """Wrapper around MediaPipe Tasks HandLandmarker for VIDEO frames."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def __init__(self, max_hands: int, min_det_conf: float, min_track_conf: float):
        if mp is None:
            raise RuntimeError("mediapipe not installed.")
        
        model_path = download_if_needed(self.MODEL_URL, pathlib.Path("models/hand_landmarker.task"))
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_det_conf,
            min_hand_presence_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.time()

    def detect(self, rgb_frame: np.ndarray):
        """rgb_frame: HxWx3 uint8 RGB."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int((time.time() - self._t0) * 1000)
        return self.detector.detect_for_video(mp_image, ts_ms)


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """Accepts MediaPipe solutions landmarks or tasks landmarks list."""
    pts = np.zeros((21, 2), dtype=np.int32)
    iterable = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    for i, lm in enumerate(iterable):
        px = int(np.clip(lm.x * width, 0, width - 1))
        py = int(np.clip(lm.y * height, 0, height - 1))
        pts[i] = (px, py)
    return pts


def smooth_landmarks(prev: Optional[np.ndarray], curr: np.ndarray, alpha: float, max_jump: float) -> np.ndarray:
    """Temporal EMA + jump rejection; fallback to prev on NaN/large jump."""
    if prev is None or prev.shape != curr.shape:
        return curr

    out = curr.copy()
    mask_curr = np.isfinite(curr).all(axis=1)
    mask_prev = np.isfinite(prev).all(axis=1)

    for i in range(curr.shape[0]):
        if not mask_curr[i] and mask_prev[i]:
            out[i] = prev[i]
            continue
        if mask_curr[i] and mask_prev[i]:
            jump = np.linalg.norm(curr[i] - prev[i])
            if jump > max_jump:
                out[i] = prev[i]
            else:
                out[i] = alpha * curr[i] + (1.0 - alpha) * prev[i]
    return out


def draw_2d_overlay(
    bgr: np.ndarray,
    pixels_xy: np.ndarray,
    handedness: str,
    fps: Optional[float] = None,
) -> np.ndarray:
    out = bgr
    for (a, b) in HAND_CONNECTIONS:
        ax, ay = pixels_xy[a]
        bx, by = pixels_xy[b]
        cv2.line(out, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 0), 2)
    for (x, y) in pixels_xy:
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)

    label = handedness
    if fps is not None:
        label = f"{handedness} | FPS: {fps:.1f}"
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


class LiveHand3DPlot:
    def __init__(self, max_hands: int = 2, title: str = "Hand 3D"):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise RuntimeError(
                "启用 --enable-3d 需要 matplotlib。请先安装 matplotlib。"
            ) from e

        self._plt = plt
        plt.ion()
        self.fig = plt.figure(title, figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.max_hands = max_hands
        self.scatters = []
        self.lines = []
        self._init_artists()

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self._set_default_limits()

    def _set_default_limits(self):
        self.ax.set_xlim(-0.25, 0.25)
        self.ax.set_ylim(-0.25, 0.25)
        self.ax.set_zlim(0.0, 1.0)

    def _init_artists(self):
        for _ in range(self.max_hands):
            sc = self.ax.scatter([], [], [], s=15)
            self.scatters.append(sc)
            segs = []
            for _ in HAND_CONNECTIONS:
                ln, = self.ax.plot([], [], [], linewidth=2)
                segs.append(ln)
            self.lines.append(segs)

    def update(self, hands: Sequence[Hand3D]):
        for i in range(self.max_hands):
            if i >= len(hands):
                self.scatters[i]._offsets3d = ([], [], [])
                for ln in self.lines[i]:
                    ln.set_data([], [])
                    ln.set_3d_properties([])
                continue

            xyz = hands[i].world_xyz
            valid = np.isfinite(xyz).all(axis=1)
            xyzv = xyz[valid]
            if xyzv.shape[0] == 0:
                self.scatters[i]._offsets3d = ([], [], [])
                for ln in self.lines[i]:
                    ln.set_data([], [])
                    ln.set_3d_properties([])
                continue

            # Camera coords: X right, Y down, Z forward -> plot with Y up
            X = xyz[:, 0]
            Y = -xyz[:, 1]
            Z = xyz[:, 2]
            self.scatters[i]._offsets3d = (X[valid], Y[valid], Z[valid])

            for (conn_idx, (a, b)) in enumerate(HAND_CONNECTIONS):
                if not (np.isfinite([X[a], Y[a], Z[a]]).all() and np.isfinite([X[b], Y[b], Z[b]]).all()):
                    self.lines[i][conn_idx].set_data([], [])
                    self.lines[i][conn_idx].set_3d_properties([])
                    continue
                self.lines[i][conn_idx].set_data([X[a], X[b]], [Y[a], Y[b]])
                self.lines[i][conn_idx].set_3d_properties([Z[a], Z[b]])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        try:
            self._plt.ioff()
            self._plt.close('all')
        except:
            pass


def sample_depth_in_patch(depth_map_m: np.ndarray, x: int, y: int, radius: int = 2) -> float:
    """
    Get depth (meters) at (x,y) from a float depth map (meters), 
    using a small neighborhood fallback if the center is invalid (<=0).
    """
    h, w = depth_map_m.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    d = float(depth_map_m[y, x])
    if d > 0:
        return d

    vals: List[float] = []
    for dy in range(-radius, radius + 1):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        for dx in range(-radius, radius + 1):
            xx = x + dx
            if xx < 0 or xx >= w:
                continue
            dd = float(depth_map_m[yy, xx])
            if dd > 0:
                vals.append(dd)

    if not vals:
        return 0.0
    return float(np.median(vals))


def pixel_to_point_pinhole(intr, u: float, v: float, depth: float) -> np.ndarray:
    """Manual deprojection using intrinsics (fx, fy, cx, cy)."""
    x = (u - intr.cx) / intr.fx * depth
    y = (v - intr.cy) / intr.fy * depth
    z = depth
    return np.array([x, y, z], dtype=np.float32)


def pixels_to_3d_pinhole(
    depth_map_m: np.ndarray,
    intr,
    pixels_xy: np.ndarray,
    depth_radius: int,
) -> np.ndarray:
    """Depth (meters) + pinhole intrinsics -> XYZ (meters)."""
    xyz = np.zeros((pixels_xy.shape[0], 3), dtype=np.float32)
    for i, (x, y) in enumerate(pixels_xy.tolist()):
        depth_m = sample_depth_in_patch(depth_map_m, x, y, radius=depth_radius)
        if depth_m <= 0:
            xyz[i] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            continue
        xyz[i] = pixel_to_point_pinhole(intr, float(x), float(y), depth_m)
    return xyz


def pixels_to_3d_with_deproject(
    depth_map_m: np.ndarray,
    intr,
    pixels_xy: np.ndarray,
    depth_radius: int,
    deproject_fn,
) -> np.ndarray:
    """
    Generic converter using a provided deproject_fn(intr, [u,v], depth)->(X,Y,Z).
    Suitable for RealSense rs.rs2_deproject_pixel_to_point or similar APIs.
    """
    xyz = np.zeros((pixels_xy.shape[0], 3), dtype=np.float32)
    for i, (x, y) in enumerate(pixels_xy.tolist()):
        depth_m = sample_depth_in_patch(depth_map_m, x, y, radius=depth_radius)
        if depth_m <= 0:
            xyz[i] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            continue
        X, Y, Z = deproject_fn(intr, [float(x), float(y)], depth_m)
        xyz[i] = np.array([X, Y, Z], dtype=np.float32)
    return xyz


def pixels_to_3d_realsense_frame(
    depth_frame,
    intr,
    pixels_xy: np.ndarray,
    depth_radius: int,
    deproject_fn,
) -> np.ndarray:
    """
    RealSense专用：直接从 depth_frame.get_distance 采样（米），并做邻域中值补洞，
    再用设备提供的 deproject_fn(intr, [u,v], depth)->(X,Y,Z)。
    行为对齐旧版 mediaP1。
    """
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    xyz = np.zeros((pixels_xy.shape[0], 3), dtype=np.float32)

    def _sample(x: int, y: int) -> float:
        x_clamped = int(np.clip(x, 0, w - 1))
        y_clamped = int(np.clip(y, 0, h - 1))
        d = float(depth_frame.get_distance(x_clamped, y_clamped))
        if d > 0:
            return d
        vals: List[float] = []
        for dy in range(-depth_radius, depth_radius + 1):
            yy = y_clamped + dy
            if yy < 0 or yy >= h:
                continue
            for dx in range(-depth_radius, depth_radius + 1):
                xx = x_clamped + dx
                if xx < 0 or xx >= w:
                    continue
                dd = float(depth_frame.get_distance(xx, yy))
                if dd > 0:
                    vals.append(dd)
        if not vals:
            return 0.0
        return float(np.median(vals))

    for i, (x, y) in enumerate(pixels_xy.tolist()):
        depth_m = _sample(x, y)
        if depth_m <= 0:
            xyz[i] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            continue
        X, Y, Z = deproject_fn(intr, [float(x), float(y)], depth_m)
        xyz[i] = np.array([X, Y, Z], dtype=np.float32)
    return xyz
