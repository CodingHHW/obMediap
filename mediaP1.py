import argparse
import sys
import time
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


try:
	import cv2
except Exception as e:  # pragma: no cover
	raise RuntimeError(
		"无法导入 OpenCV(cv2)。请先安装 opencv-python。"
	) from e


try:
	import pyrealsense2 as rs
except Exception as e:  # pragma: no cover
	raise RuntimeError(
		"无法导入 pyrealsense2。请确认已安装 librealsense/pyrealsense2 并在正确环境中运行。"
	) from e


try:
	import mediapipe as mp
except Exception as e:  # pragma: no cover
	raise RuntimeError(
		"无法导入 mediapipe。请先安装 mediapipe。"
	) from e

try:
	from mediapipe.tasks import python as mp_python
	from mediapipe.tasks.python import vision as mp_vision
except Exception as e:  # pragma: no cover
	raise RuntimeError(
		"当前 mediapipe 版本缺少 tasks API，请升级到 0.10.x 及以上。"
	) from e


HAND_CONNECTIONS: Sequence[Tuple[int, int]] = (
	(0, 1),
	(1, 2),
	(2, 3),
	(3, 4),
	(0, 5),
	(5, 6),
	(6, 7),
	(7, 8),
	(5, 9),
	(9, 10),
	(10, 11),
	(11, 12),
	(9, 13),
	(13, 14),
	(14, 15),
	(15, 16),
	(13, 17),
	(17, 18),
	(18, 19),
	(19, 20),
	(0, 17),
)


@dataclass
class Hand3D:
	world_xyz: np.ndarray  # (21, 3) in meters, camera coords
	handedness: str


def _sample_depth_m(depth_frame: "rs.depth_frame", x: int, y: int, radius: int = 2) -> float:
	"""Get depth (meters) at (x,y), with small neighborhood fallback."""
	w = depth_frame.get_width()
	h = depth_frame.get_height()
	x = int(np.clip(x, 0, w - 1))
	y = int(np.clip(y, 0, h - 1))

	d = float(depth_frame.get_distance(x, y))
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
			dd = float(depth_frame.get_distance(xx, yy))
			if dd > 0:
				vals.append(dd)

	if not vals:
		return 0.0
	return float(np.median(vals))


def _landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
	"""Accepts MediaPipe solutions landmarks or tasks landmarks list."""
	pts = np.zeros((21, 2), dtype=np.int32)
	iterable = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
	for i, lm in enumerate(iterable):
		px = int(np.clip(lm.x * width, 0, width - 1))
		py = int(np.clip(lm.y * height, 0, height - 1))
		pts[i] = (px, py)
	return pts


def _pixels_to_3d(
	depth_frame: "rs.depth_frame",
	intr: "rs.intrinsics",
	pixels_xy: np.ndarray,
    depth_radius: int,
) -> np.ndarray:
	xyz = np.zeros((pixels_xy.shape[0], 3), dtype=np.float32)
	for i, (x, y) in enumerate(pixels_xy.tolist()):
		depth_m = _sample_depth_m(depth_frame, x, y, radius=depth_radius)
		if depth_m <= 0:
			xyz[i] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
			continue
		X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], depth_m)
		xyz[i] = np.array([X, Y, Z], dtype=np.float32)
	return xyz


def _smooth_landmarks(prev: Optional[np.ndarray], curr: np.ndarray, alpha: float, max_jump: float) -> np.ndarray:
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


def _draw_2d_overlay(
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
	def __init__(self, max_hands: int = 2):
		try:
			import matplotlib.pyplot as plt  # type: ignore
		except Exception as e:  # pragma: no cover
			raise RuntimeError(
				"启用 --enable-3d 需要 matplotlib。请先安装 matplotlib。"
			) from e

		self._plt = plt
		plt.ion()
		self.fig = plt.figure("Hand 3D", figsize=(7, 6))
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


def _download_if_needed(url: str, dst_path: pathlib.Path) -> pathlib.Path:
	dst_path.parent.mkdir(parents=True, exist_ok=True)
	if not dst_path.exists():
		print(f"[INFO] 下载模型到 {dst_path} ...", flush=True)
		urllib.request.urlretrieve(url, dst_path)
	return dst_path


class HandDetector:
	"""Wrapper around MediaPipe Tasks HandLandmarker for VIDEO frames."""

	MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

	def __init__(self, max_hands: int, min_det_conf: float, min_track_conf: float):
		model_path = _download_if_needed(self.MODEL_URL, pathlib.Path("models/hand_landmarker.task"))
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


def run(args: argparse.Namespace) -> int:
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
	config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

	profile = pipeline.start(config)
	align = rs.align(rs.stream.color)

	plot3d = LiveHand3DPlot(max_hands=args.max_hands) if args.enable_3d else None
	landmarker = HandDetector(
		max_hands=args.max_hands,
		min_det_conf=args.min_det_conf,
		min_track_conf=args.min_track_conf,
	)
	prev_xyz_list: List[Optional[np.ndarray]] = [None for _ in range(args.max_hands)]

	last_t = time.time()
	fps_ema: Optional[float] = None

	try:
		while True:
			frames = pipeline.wait_for_frames()
			aligned = align.process(frames)

			color_frame = aligned.get_color_frame()
			depth_frame = aligned.get_depth_frame()
			if not color_frame or not depth_frame:
				continue

			color = np.asanyarray(color_frame.get_data())
			h, w, _ = color.shape

			rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
			results = landmarker.detect(rgb)

			intr = depth_frame.profile.as_video_stream_profile().intrinsics

			hands3d: List[Hand3D] = []
			if results and results.hand_landmarks:
				for idx, hand_lms in enumerate(results.hand_landmarks):
					pixels = _landmarks_to_pixels(hand_lms, w, h)
					xyz_raw = _pixels_to_3d(depth_frame, intr, pixels, depth_radius=args.depth_radius)
					xyz = _smooth_landmarks(prev_xyz_list[idx], xyz_raw, alpha=args.smooth_alpha, max_jump=args.smooth_max_jump)
					prev_xyz_list[idx] = xyz

					handed = "Unknown"
					if results.handedness and idx < len(results.handedness):
						if len(results.handedness[idx]) > 0:
							handed = results.handedness[idx][0].category_name

					hands3d.append(Hand3D(world_xyz=xyz, handedness=handed))

					_draw_2d_overlay(color, pixels, handedness=handed, fps=None)

			# FPS
			now = time.time()
			dt = max(1e-6, now - last_t)
			inst_fps = 1.0 / dt
			fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)
			last_t = now
			cv2.putText(
				color,
				f"FPS: {fps_ema:.1f}",
				(10, h - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(255, 255, 255),
				2,
			)

			if plot3d is not None:
				plot3d.update(hands3d)

			cv2.imshow("RealSense + MediaPipe Hands (2D)", color)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

	finally:
		pipeline.stop()
		cv2.destroyAllWindows()
	return 0


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Intel RealSense + MediaPipe Hands: 2D关键点 -> 3D反投影并可视化",
	)
	p.add_argument("--width", type=int, default=640)
	p.add_argument("--height", type=int, default=480)
	p.add_argument("--fps", type=int, default=30)
	p.add_argument("--max-hands", type=int, default=2)
	p.add_argument("--min-det-conf", type=float, default=0.5)
	p.add_argument("--min-track-conf", type=float, default=0.5)
	p.add_argument("--depth-radius", type=int, default=2, help="深度邻域半径（像素），用于缺测插值")
	p.add_argument("--smooth-alpha", type=float, default=0.3, help="3D坐标EMA平滑系数，0-1，越大越跟随实时")
	p.add_argument("--smooth-max-jump", type=float, default=0.12, help="若两帧距离跳变超过此值(米)，则使用上一帧值抑制抖动")
	p.add_argument("--enable-3d", action="store_true", help="开启 Matplotlib 3D 骨架可视化")
	p.add_argument(
		"--draw-mediapipe-style",
		action="store_true",
		help="2D叠加使用 MediaPipe 默认绘制样式（否则使用简易点线）",
	)
	return p


def main() -> int:
	args = build_argparser().parse_args()
	return run(args)


if __name__ == "__main__":
	raise SystemExit(main())

