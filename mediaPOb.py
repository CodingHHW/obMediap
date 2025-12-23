import argparse
import sys
import time
from typing import List, Optional

import numpy as np
import cv2

try:
    from pyorbbecsdk import (
        Config,
        FrameSet,
        OBError,
        OBFormat,
        OBSensorType,
        Pipeline,
    )
    import pyorbbecsdk as ob
except ImportError as e:
    raise RuntimeError(
        "无法导入 pyorbbecsdk。请确认已安装 Orbbec SDK Python 绑定。"
    ) from e

from utils import frame_to_bgr_image
from hand_utils import (
    Hand3D,
    HandDetector,
    LiveHand3DPlot,
    draw_2d_overlay,
    landmarks_to_pixels,
    pixels_to_3d_pinhole,
    smooth_landmarks,
)
 


def get_stream_profile(pipeline, sensor_type, width, height, fmt, fps):
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    try:
        return profile_list.get_video_stream_profile(width, height, fmt, fps)
    except OBError:
        return profile_list.get_default_video_stream_profile()


def run(args: argparse.Namespace) -> int:
    config = Config()
    pipeline = Pipeline()

    # Enable streams
    color_profile = get_stream_profile(pipeline, OBSensorType.COLOR_SENSOR, args.width, args.height, OBFormat.RGB, args.fps)
    depth_profile = get_stream_profile(pipeline, OBSensorType.DEPTH_SENSOR, args.width, args.height, OBFormat.Y16, args.fps)
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    # Try alignment
    align_enabled = False
    try:
        if hasattr(config, "set_align_mode"):
            align_const = getattr(ob, "ALIGN_D2C_HW_MODE", None) or getattr(ob, "ALIGN_D2C_SW_MODE", None)
            if align_const is not None:
                config.set_align_mode(align_const)
                align_enabled = True
                print(f"Alignment: ENABLED ({'HW' if 'HW' in str(align_const) else 'SW'})")
            else:
                print("Alignment: constants not found; running unaligned")
        else:
            print("Alignment: set_align_mode not available; running unaligned")
    except Exception as exc:
        print(f"Alignment: failed ({exc}); running unaligned")

    pipeline.start(config)

    # Intrinsics
    camera_param = pipeline.get_camera_param()
    depth_intr = camera_param.depth_intrinsic
    color_intr = getattr(camera_param, "color_intrinsic", depth_intr)
    intr_for_proj = color_intr if align_enabled else depth_intr

    # Depth scale
    depth_scale = 0.001
    try:
        dev = pipeline.get_device()
        depth_scale = dev.get_depth_scale()
    except Exception:
        pass

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
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            color_bgr = frame_to_bgr_image(color_frame)
            if color_bgr is None:
                continue

            # Depth data
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            depth_map_m = depth_data.astype(np.float32) * depth_scale

            h, w, _ = color_bgr.shape
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            results = landmarker.detect(rgb)

            hands3d: List[Hand3D] = []
            if results and results.hand_landmarks:
                for idx, hand_lms in enumerate(results.hand_landmarks):
                    pixels = landmarks_to_pixels(hand_lms, w, h)
                    xyz_raw = pixels_to_3d_pinhole(
                        depth_map_m,
                        intr_for_proj,
                        pixels,
                        depth_radius=args.depth_radius,
                    )
                    xyz = smooth_landmarks(prev_xyz_list[idx], xyz_raw, alpha=args.smooth_alpha, max_jump=args.smooth_max_jump)
                    prev_xyz_list[idx] = xyz

                    handed = "Unknown"
                    if results.handedness and idx < len(results.handedness):
                        if len(results.handedness[idx]) > 0:
                            handed = results.handedness[idx][0].category_name

                    hands3d.append(Hand3D(world_xyz=xyz, handedness=handed))

                    draw_2d_overlay(color_bgr, pixels, handedness=handed, fps=None)

            # FPS
            now = time.time()
            dt = max(1e-6, now - last_t)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)
            last_t = now
            cv2.putText(
                color_bgr,
                f"FPS: {fps_ema:.1f}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if plot3d is not None:
                plot3d.update(hands3d)

            cv2.imshow("Orbbec + MediaPipe Hands (2D)", color_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if plot3d:
            try:
                import matplotlib.pyplot as plt
                plt.ioff()
                plt.close('all')
            except:
                pass
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orbbec + MediaPipe Hands: 2D关键点 -> 3D反投影并可视化",
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
    return p


def main() -> int:
    args = build_argparser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
