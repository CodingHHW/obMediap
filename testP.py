"""
RealSense 彩色流 + 深度 + MediaPipe Hands：2D 关键点 + 3D 坐标（中心点）
- 启用彩色与深度流，深度对齐到彩色
- 使用深度将 2D 手部关键点/中心点反投影为 3D (米)
- 按 q 退出

依赖：pyrealsense2, mediapipe, opencv-python
"""

import argparse
import time
from typing import Optional

import numpy as np

try:
	import cv2
except Exception as e:
	raise RuntimeError("请先安装 opencv-python") from e

try:
	import pyrealsense2 as rs
except Exception as e:
	raise RuntimeError("请先安装 pyrealsense2 / librealsense") from e

from hand_utils import (
	Hand3D,
	HandDetector,
	LiveHand3DPlot,
	draw_2d_overlay,
	landmarks_to_pixels,
	pixels_to_3d_with_deproject,
)


def run(args: argparse.Namespace) -> int:
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
	config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

	profile = pipeline.start(config)
	align = rs.align(rs.stream.color)
	depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
	print("RealSense color+depth stream started (depth aligned to color).")

	landmarker = HandDetector(
		max_hands=args.max_hands,
		min_det_conf=args.min_det_conf,
		min_track_conf=args.min_track_conf,
	)

	plot3d = LiveHand3DPlot(max_hands=args.max_hands) if args.enable_3d else None

	fps_ema: Optional[float] = None
	last_t = time.time()

	try:
		while True:
			frames = pipeline.wait_for_frames()
			aligned = align.process(frames)

			color_frame = aligned.get_color_frame()
			depth_frame = aligned.get_depth_frame()
			if not color_frame or not depth_frame:
				continue

			color = np.asanyarray(color_frame.get_data())
			depth_raw = np.asanyarray(depth_frame.get_data())
			depth_map_m = depth_raw.astype(np.float32) * depth_scale
			h, w, _ = color.shape
			rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

			results = landmarker.detect(rgb)

			# 对齐后使用彩色相机内参（aligned depth 已在彩色坐标系）
			intr = color_frame.profile.as_video_stream_profile().intrinsics

			hands3d = []
			if results and results.hand_landmarks:
				for idx, hand_lms in enumerate(results.hand_landmarks):
					pixels = landmarks_to_pixels(hand_lms, w, h)
					xyz = pixels_to_3d_with_deproject(
						depth_map_m,
						intr,
						pixels,
						depth_radius=args.depth_radius,
						deproject_fn=rs.rs2_deproject_pixel_to_point,
					)

					draw_2d_overlay(color, pixels, handedness="Hand", fps=None)

					valid_mask = np.isfinite(xyz).all(axis=1)
					if valid_mask.any():
						center_xyz = xyz[valid_mask].mean(axis=0)
						center_px = pixels[valid_mask].mean(axis=0)
						cx, cy = int(center_px[0]), int(center_px[1])
						cv2.circle(color, (cx, cy), 5, (0, 255, 255), -1)
						cv2.putText(
							color,
							f"({center_xyz[0]:.3f}, {center_xyz[1]:.3f}, {center_xyz[2]:.3f}) m",
							(cx + 8, cy - 8),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.5,
							(0, 255, 255),
							2,
						)

					hands3d.append(Hand3D(world_xyz=xyz, handedness="Hand"))

			if plot3d is not None:
				plot3d.update(hands3d)

			# FPS 显示
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

			cv2.imshow("RealSense RGB+Depth + MediaPipe Hands (2D->3D)", color)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

	finally:
		pipeline.stop()
		cv2.destroyAllWindows()
		if plot3d is not None:
			plot3d.close()
	print("Stopped.")
	return 0


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="RealSense RGB+Depth + MediaPipe Hands (2D->3D) demo")
	p.add_argument("--width", type=int, default=640)
	p.add_argument("--height", type=int, default=480)
	p.add_argument("--fps", type=int, default=30)
	p.add_argument("--max-hands", type=int, default=2)
	p.add_argument("--min-det-conf", type=float, default=0.5)
	p.add_argument("--min-track-conf", type=float, default=0.5)
	p.add_argument("--depth-radius", type=int, default=2, help="深度邻域半径（像素），用于缺测插值")
	p.add_argument("--enable-3d", action="store_true", help="开启 Matplotlib 3D 骨架可视化")
	return p


def main() -> int:
	args = build_argparser().parse_args()
	return run(args)


if __name__ == "__main__":
	raise SystemExit(main())
