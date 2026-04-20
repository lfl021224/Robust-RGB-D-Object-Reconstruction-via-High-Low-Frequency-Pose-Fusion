# -*- coding: utf-8 -*-
# stereo_pair_to_pcd_kitti2015.py
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import re

# ---------- 1) 读取 calib_cam_to_cam ----------
def load_cam_to_cam(calib_path):
    """
    支持两种用法：
    - calib_path 指向具体文件：.../calib_cam_to_cam/000000.txt
    - calib_path 指向目录：.../calibration/training/ 或 .../calib_cam_to_cam/
      （这种情况下请先用 infer_calib_file() 得到具体文件路径）

    返回: fx, cx, cy, baseline_m
    """
    p = Path(calib_path)
    assert p.exists(), f"标定路径不存在: {p}"

    # 读取文本
    txt = p.read_text(encoding="utf-8", errors="ignore")

    def parse_P(line):
        vals = np.fromstring(line.split(":", 1)[1], sep=" ")
        P = vals.reshape(3, 4)
        fx, cx, cy, Tx = P[0, 0], P[0, 2], P[1, 2], P[0, 3]
        return fx, cx, cy, Tx

    # 兼容两种键名：P_rect_02/03（RAW）与 P2/P3（2015）
    lines = txt.splitlines()
    keyL = next((k for k in ("P_rect_02:", "P2:") if any(l.startswith(k) for l in lines)), None)
    keyR = next((k for k in ("P_rect_03:", "P3:") if any(l.startswith(k) for l in lines)), None)
    assert keyL and keyR, f"未在 {p.name} 找到 P_rect_02/03 或 P2/P3"

    lineL = next(l for l in lines if l.startswith(keyL))
    lineR = next(l for l in lines if l.startswith(keyR))
    fxL, cx, cy, TxL = parse_P(lineL)
    fxR, _,  _,  TxR = parse_P(lineR)

    # 基线（米）： b = |Tx_right/(-fx_right) - Tx_left/(-fx_left)|
    b = abs(TxR / (-fxR) - TxL / (-fxL))
    return float(fxL), float(cx), float(cy), float(b)

def infer_calib_file(left_img_path, calib_root):
    """
    根据左图文件名推断对应的 000000.txt。
    例如： left = .../training/image_2/000123_10.png
    优先寻找：
      <calib_root>/calib_cam_to_cam/000123.txt
    若 <calib_root> 本身就是 calib_cam_to_cam/，则直接拼 000123.txt
    """
    left = Path(left_img_path)
    root = Path(calib_root)

    # 提取 6 位 id（如 000123），既兼容 000123_10.png，也兼容 000123.png
    m = re.match(r"(\d{6})", left.stem)
    assert m, f"无法从文件名解析样本 ID：{left.name}"
    sid = m.group(1)

    # 校准文件可能位于：<root>/calib_cam_to_cam/000123.txt 或 <root>/000123.txt
    cand1 = root / "calib_cam_to_cam" / f"{sid}.txt"
    cand2 = root / f"{sid}.txt"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"未找到 {sid}.txt，尝试路径：\n  {cand1}\n  {cand2}")

# ---------- 2) 立体匹配（SGBM） ----------
def compute_disparity_sgbm(left_gray, right_gray,
                           min_disp=0, num_disp=128, block=5):
    # num_disp 必须是 16 的倍数
    num_disp = int(np.ceil(num_disp / 16) * 16)
    P1 = 8 * 1 * block * block
    P2 = 32 * 1 * block * block
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp < 0] = 0.0
    return disp

# ---------- 3) 反投影到点云 ----------
def backproject_to_pcd(disp, color_bgr, fx, cx, cy, baseline_m, max_z=80.0):
    h, w = disp.shape
    eps = 1e-6
    Z = fx * baseline_m / (disp + eps)

    valid = (disp > 0) & np.isfinite(Z) & (Z > 0) & (Z < max_z)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us = us[valid]; vs = vs[valid]; Zv = Z[valid]

    X = (us - cx) * Zv / fx
    Y = (vs - cy) * Zv / fx  # 若需要更严谨可用 fy，这里假设 fx≈fy
    pts = np.stack([X, Y, Zv], axis=1)

    cols = color_bgr[valid][:, ::-1] / 255.0  # BGR->RGB
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd

# ---------- 4) 主流程 ----------
def run_once(left_img_path, right_img_path,
             calib_file_or_dir, save_ply="pcd_out.ply",
             num_disp=128, block=5, max_z=80.0, show_windows=True):
    left_img_path = Path(left_img_path)
    right_img_path = Path(right_img_path)
    assert left_img_path.exists() and right_img_path.exists(), "左右图像路径不存在"

    # 若给的是目录，先推断具体的 000000.txt
    calib_path = Path(calib_file_or_dir)
    if calib_path.is_dir():
        calib_path = infer_calib_file(left_img_path, calib_path)

    fx, cx, cy, b = load_cam_to_cam(calib_path)
    print(f"[calib] fx={fx:.2f}, cx={cx:.2f}, cy={cy:.2f}, baseline={b:.5f} m")
    print(f"[info ] left: {left_img_path.name}  right: {right_img_path.name}")
    print(f"[info ] calib: {calib_path}")

    left = cv2.imread(str(left_img_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_img_path), cv2.IMREAD_COLOR)
    assert left is not None and right is not None, "读取图像失败"
    assert left.shape == right.shape, "左右图尺寸不一致，请确认使用整流后的 image_2/image_3"

    left_gray  = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disp = compute_disparity_sgbm(left_gray, right_gray, num_disp=num_disp, block=block)

    if show_windows:
        disp_vis = (disp / (disp.max() + 1e-6) * 255).astype(np.uint8)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
        cv2.imshow("left", left)
        cv2.imshow("disparity", disp_vis)
        cv2.waitKey(1)

    pcd = backproject_to_pcd(disp, left, fx, cx, cy, b, max_z=max_z)
    # 轻量清理
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    if len(pcd.points) > 1000:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

    o3d.io.write_point_cloud(save_ply, pcd)
    print(f"[done ] saved -> {save_ply}  points={np.asarray(pcd.points).shape[0]}")

    if show_windows:
        o3d.visualization.draw_geometries([pcd], window_name="Stereo Point Cloud")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 你的本地路径（注意 r"" 原始字符串，避免 \U 转义）
    img_root   = Path(r"D:\Projects\Underwater\Datasets\kitti\data_scene_flow")
    calib_root = Path(r"D:\Projects\Underwater\Datasets\kitti\data_scene_flow_calib") / "training"

    left  = img_root / "training" / "image_2" / "000000_10.png"
    right = img_root / "training" / "image_3" / "000000_10.png"

    # 传“calibration 的 split 目录”（自动匹配 000000.txt）
    run_once(left, right, calib_root,
             save_ply="pcd_kitti2015_000000_10.ply",
             num_disp=128, block=5, max_z=80.0, show_windows=True)
