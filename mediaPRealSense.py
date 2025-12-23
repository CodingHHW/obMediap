import argparse
import sys
import time
from typing import List, Optional

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

from hand_utils import (
	Hand3D,
	HandDetector,
	LiveHand3DPlot,
	draw_2d_overlay,
	landmarks_to_pixels,
	pixels_to_3d_with_deproject,
	pixels_to_3d_realsense_frame,
	smooth_landmarks,
)


def run(args: argparse.Namespace) -> int:
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
	config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

	profile = pipeline.start(config)
	align = rs.align(rs.stream.color)
	depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

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
			depth_raw = np.asanyarray(depth_frame.get_data())
			depth_map_m = depth_raw.astype(np.float32) * depth_scale
			h, w, _ = color.shape

			rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
			results = landmarker.detect(rgb)

			intr = depth_frame.profile.as_video_stream_profile().intrinsics

			hands3d: List[Hand3D] = []
			if results and results.hand_landmarks:
				for idx, hand_lms in enumerate(results.hand_landmarks):
					pixels = landmarks_to_pixels(hand_lms, w, h)
					if args.depth_mode == "frame":
						xyz_raw = pixels_to_3d_realsense_frame(
							depth_frame,
							intr,
							pixels,
							depth_radius=args.depth_radius,
							deproject_fn=rs.rs2_deproject_pixel_to_point,
						)
					else:
						xyz_raw = pixels_to_3d_with_deproject(
							depth_map_m,
							intr,
							pixels,
							depth_radius=args.depth_radius,
							deproject_fn=rs.rs2_deproject_pixel_to_point,
						)
					xyz = smooth_landmarks(prev_xyz_list[idx], xyz_raw, alpha=args.smooth_alpha, max_jump=args.smooth_max_jump)
					prev_xyz_list[idx] = xyz

					handed = "Unknown"
					if results.handedness and idx < len(results.handedness):
						if len(results.handedness[idx]) > 0:
							handed = results.handedness[idx][0].category_name

					hands3d.append(Hand3D(world_xyz=xyz, handedness=handed))

					draw_2d_overlay(color, pixels, handedness=handed, fps=None)

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
	p.add_argument(
		"--depth-mode",
		choices=["frame", "map"],
		default="frame",
		help="frame: 使用 get_distance + 邻域补洞(与 mediaP1 一致)；map: 使用 raw z16 * depth_scale 再反投影。",
	)
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

