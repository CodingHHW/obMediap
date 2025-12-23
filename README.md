# RealSense / Orbbec + MediaPipe Hands (2D -> 3D)

使用 Intel RealSense 或 Orbbec 深度相机采集彩色+深度，MediaPipe Hands 识别手部 2D 关键点，并用深度+相机内参反投影成 3D 点，实时可视化 3D 骨架。共用逻辑已抽到 `hand_utils.py`，两端脚本只保留设备差异部分。

## 依赖

- 公共：`opencv-python`、`mediapipe`、`numpy`、`matplotlib`
- RealSense：`pyrealsense2`（来自 librealsense）
- Orbbec：`pyorbbecsdk`

建议使用 conda-forge 一次装齐（以名为 `realsense` 的环境为例，可更名）：

```bash
conda install -n realsense -c conda-forge -y \
  numpy<2 matplotlib opencv mediapipe portaudio python-sounddevice
```

再按设备安装：

- RealSense: 按官方方式安装 librealsense（带 Python 绑定），或 `pip install pyrealsense2`
- Orbbec: 安装 Orbbec SDK Python 绑定（`pyorbbecsdk`），按官方说明配置

pip 方式（可选）：

```bash
conda activate realsense
pip install -r requirements.txt
```

若 `pip install mediapipe` 报 `sounddevice` 相关问题，通常是缺少 `portaudio/python-sounddevice`，用上面的 conda-forge 安装即可。

## 运行

### Orbbec

```bash
python mediaPOb.py            # 仅 2D 叠加
python mediaPOb.py --enable-3d  # 叠加 + 3D 骨架
```

### RealSense

```bash
python mediaPRealSense.py
python mediaPRealSense.py --enable-3d
```

可选参数：

- `--width 640 --height 480 --fps 30`
- `--max-hands 2`
- `--depth-radius 2` 深度邻域半径（像素）；缺测时取邻域中值，适当增大可减抖但略降精度
- `--smooth-alpha 0.3` 3D 坐标 EMA 平滑系数，0-1，越大跟随越快、越小越平滑
- `--smooth-max-jump 0.12` 单点两帧间距离若超过该阈值（米）则使用上一帧值抑制抖动

退出：在 2D 窗口按 `q`。

提示：首次运行会自动下载 MediaPipe Tasks 的 `hand_landmarker.task` 模型到 `models/hand_landmarker.task`。

若 3D 关键点偶尔跳变，可尝试：

```bash
# Orbbec
python mediaPOb.py --enable-3d --depth-radius 3 --smooth-alpha 0.3 --smooth-max-jump 0.12

# RealSense
python mediaPRealSense.py --enable-3d --depth-radius 3 --smooth-alpha 0.3 --smooth-max-jump 0.12
```

抖动大 → 降低 `smooth-alpha` 或减小 `smooth-max-jump`；响应慢 → 提高 `smooth-alpha` 或放宽 `smooth-max-jump`。

## 说明

- 3D 坐标系为相机坐标（单位：米）：X 向右、Y 向下、Z 向前；3D 图里为了更直观把 Y 取了相反数显示为“向上”。
- 深度为 0 时会对关键点做小范围邻域采样以提高稳定性。
