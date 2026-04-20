# -*- coding: utf-8 -*-
import os
import time
from pathlib import Path
import threading
import queue
import numpy as np
import cv2
import traceback
import open3d as o3d
import contextlib
import io
import math

# ==========================
# helpers: twist clamp
# ==========================
def clamp_twist(xi, max_r_rad: float, max_t: float):
    """Clamp se(3) twist: ||w|| <= max_r_rad, ||v|| <= max_t."""
    xi = np.asarray(xi, dtype=np.float64).copy()
    w = xi[:3]; v = xi[3:]
    nw = float(np.linalg.norm(w))
    if nw > max_r_rad:
        xi[:3] = w * (max_r_rad / max(1e-12, nw))
    nv = float(np.linalg.norm(v))
    if nv > max_t:
        xi[3:] = v * (max_t / max(1e-12, nv))
    return xi

# ==========================
# GT + SE(3) complementary filter utils
# ==========================
def skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=np.float64)

def so3_log(R):
    # returns w such that Exp(w)=R
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        # small angle approx
        w = 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
        return w
    w_hat = (theta / (2.0 * math.sin(theta))) * np.array(
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64
    )
    return w_hat

def so3_exp(w):
    theta = float(np.linalg.norm(w))
    K = skew(w)
    I = np.eye(3, dtype=np.float64)
    if theta < 1e-8:
        return I + K
    A = math.sin(theta) / theta
    B = (1.0 - math.cos(theta)) / (theta * theta)
    return I + A * K + B * (K @ K)

def se3_exp(xi):
    # xi = [w(3), v(3)]
    w = np.asarray(xi[:3], dtype=np.float64)
    v = np.asarray(xi[3:], dtype=np.float64)
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    I = np.eye(3, dtype=np.float64)
    K = skew(w)

    if theta < 1e-8:
        V = I + 0.5 * K
    else:
        A = (1.0 - math.cos(theta)) / (theta * theta)
        B = (theta - math.sin(theta)) / (theta * theta * theta)
        V = I + A * K + B * (K @ K)

    t = V @ v
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def se3_log(T):
    R = T[:3,:3]
    t = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    I = np.eye(3, dtype=np.float64)
    K = skew(w)

    if theta < 1e-8:
        V_inv = I - 0.5 * K
    else:
        # V^{-1} = I - 0.5K + (1/θ^2 - (1+cosθ)/(2θ sinθ))K^2
        A = 1.0/(theta*theta)
        B = (1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta))
        V_inv = I - 0.5 * K + (A - B) * (K @ K)

    v = V_inv @ t
    xi = np.zeros(6, dtype=np.float64)
    xi[:3] = w
    xi[3:] = v
    return xi

def quat_to_R(qx, qy, qz, qw):
    # quaternion (x,y,z,w) -> R
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    R = np.array([
        [1.0 - (yy + zz),       xy - wz,         xz + wy],
        [      xy + wz,   1.0 - (xx + zz),       yz - wx],
        [      xz - wy,         yz + wx,   1.0 - (xx + yy)]
    ], dtype=np.float64)
    return R

def load_tum_groundtruth(gt_path: Path):
    if not gt_path.exists():
        print(f"[gt] not found: {gt_path} (GT fusion disabled)")
        return None
    ts = []
    T_list = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if (not ln) or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            R = quat_to_R(qx, qy, qz, qw)
            T = np.eye(4, dtype=np.float64)
            T[:3,:3] = R
            T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
            ts.append(t)
            T_list.append(T)
    if len(ts) < 2:
        print("[gt] too few samples (GT fusion disabled)")
        return None
    ts = np.asarray(ts, dtype=np.float64)
    return ts, T_list

def query_gt_nearest(gt_data, t_query: float, max_dt: float = 0.02):
    # nearest neighbor (good enough for TUM)
    if gt_data is None:
        return None
    ts, T_list = gt_data
    j = int(np.argmin(np.abs(ts - t_query)))
    if abs(ts[j] - t_query) > max_dt:
        return None
    return T_list[j]


# ==========================
# 1) 环境检测
# ==========================
HAS_O3D_CUDA = hasattr(o3d, "core") and hasattr(o3d.core, "cuda") and o3d.core.cuda.is_available()
print(f"[env] Open3D CUDA = {HAS_O3D_CUDA}")

HAS_CV_CUDA = (hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0)
print(f"[env] OpenCV CUDA = {HAS_CV_CUDA}, devices = {cv2.cuda.getCudaEnabledDeviceCount() if HAS_CV_CUDA else 0}")

# print('has cv2.cuda =', hasattr(cv2, 'cuda'))
# print(cv2.getBuildInformation())

# ==========================
# 2) 路径（你只需要改这里）
# ==========================
SEQ_DIR = Path(r"D:\Projects\Underwater\Datasets\TUM_RGBD\rgbd_dataset_freiburg3_cabinet")
OUT_DIR = Path(r"D:\Projects\Underwater\Outputs\TUM_RGBD\rgbd_dataset_freiburg3_cabinet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not SEQ_DIR.exists():
    raise FileNotFoundError(f"[path] SEQ_DIR 不存在: {SEQ_DIR}")

# ==========================
# 3) 运行/显示节奏
# ==========================
TARGET_FPS_BASE   = 12
KEYFRAME_STEP     = 2
LIVE_UPDATE_EVERY = 3
SHOW_MESH         = False

# ==========================
# 4) 深度窗 / ROI
# ==========================
NEAR_CLIP      = 0.55
DEPTH_TRUNC    = 1.60     # 注意：这个只是“默认”，实际每帧会用 far_auto（见后面修复）
USE_CENTER_ROI = False
ROI_SCALE      = 0.60

# ==========================
# 5) ICP/里程计阈值
# ==========================
MIN_PTS_FOR_ICP   = 1500
ICP_MIN_FIT_ODO   = 0.35
ICP_MAX_RMSE_ODO  = 0.015
ODOM_MAX_T        = 0.10
ODOM_MAX_R_DEG    = 10
LOW_OV_THR        = 0.08

# ==========================
# 6) 初值：FPFH / RANSAC
# ==========================
USE_FPFH_INIT = True
FPFH_VOXEL    = 0.025
RANSAC_DIST   = 0.035
RANSAC_N_IT   = 30000

# ==========================
# 7) TSDF 参数
# ==========================
VOXEL_LEN   = 0.008
SDF_TRUNC_K = 2.0

MAP_SAMPLE_EVERY = 15
MAP_TRACK_EVERY  = 3
_map_pcd_sample  = None

# ==========================
# 8) 自动深度裁剪/背景剔除
# ==========================
AUTO_DEPTH       = True
AUTO_Q_LOW       = 0.08
AUTO_Q_HIGH      = 0.75
AUTO_MARGIN_NEAR = 0.04
AUTO_MARGIN_FAR  = 0.05
AUTO_SMOOTH      = 0.7
AUTO_BG_OTSU     = False

# ==========================
# 9) 小工具
# ==========================
class _SilentStderr:
    def __enter__(self):
        self._buf = io.StringIO()
        self._cm = contextlib.redirect_stderr(self._buf)
        self._cm.__enter__()
        return self._buf
    def __exit__(self, exc_type, exc, tb):
        self._cm.__exit__(exc_type, exc, tb)

def depth_to_meters(depth_raw: np.ndarray):
    d = depth_raw.astype(np.float32)
    if d.size == 0:
        return d
    if depth_raw.dtype == np.uint16:
        # TUM RGB-D: depth = raw / 5000.0 (meters)
        return d / 5000.0
    return d

class CudaPreproc:
    """GPU 预处理（若不可用自动回落 CPU）"""
    def __init__(self):
        self.use_cuda = HAS_CV_CUDA
        if self.use_cuda:
            try:
                self.stream = cv2.cuda.Stream()
                self.orb = cv2.cuda_ORB.create(nfeatures=5000, fastThreshold=7, scaleFactor=1.2, nlevels=8)
                self.bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
            except Exception:
                self.use_cuda = False

    def to_gpu(self, img):
        g = cv2.cuda_GpuMat()
        g.upload(img)
        return g

    def bilateral(self, img32f, d=5, sigmaColor=0.05, sigmaSpace=3):
        if not self.use_cuda:
            return cv2.bilateralFilter(img32f, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        g = self.to_gpu(img32f)
        out = cv2.cuda.bilateralFilter(g, -1, sigmaColor, sigmaSpace)
        return out.download()

    def bgr2rgb(self, img):
        if not self.use_cuda:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        g = self.to_gpu(img)
        out = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2RGB)
        return out.download()

    def orb_match(self, img0_rgb, img1_rgb, max_pts=2500):
        img0 = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2BGR)
        img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        if self.use_cuda:
            g0 = self.to_gpu(img0); g1 = self.to_gpu(img1)
            kps0, des0 = self.orb.detectAndComputeAsync(g0, None, stream=self.stream)
            kps1, des1 = self.orb.detectAndComputeAsync(g1, None, stream=self.stream)
            if des0 is None or des1 is None:
                return [], None, None
            matches = self.bf.match(des0, des1)
            matches = sorted(matches, key=lambda x: x.distance)[:max_pts]
            kps0 = self.orb.convert(g0, kps0)
            kps1 = self.orb.convert(g1, kps1)
            return matches, kps0, kps1
        else:
            orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7, scaleFactor=1.2, nlevels=8)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            kps0, des0 = orb.detectAndCompute(img0, None)
            kps1, des1 = orb.detectAndCompute(img1, None)
            if des0 is None or des1 is None:
                return [], None, None
            matches = bf.match(des0, des1)
            matches = sorted(matches, key=lambda x: x.distance)[:max_pts]
            return matches, kps0, kps1

GP = CudaPreproc()

def _try_float_stem(p: Path):
    try:
        return float(p.stem)
    except Exception:
        return None

def list_rgbd_paths(seq_dir: Path):
    # 1) associations.txt / associate.txt 优先
    for name in ["associations.txt", "associate.txt"]:
        f = seq_dir / name
        if f.exists():
            pairs = []
            with open(f, "r", encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    if len(parts) >= 4:
                        _, rgb_rel, _, d_rel = parts[:4]
                        pairs.append((seq_dir / rgb_rel, seq_dir / d_rel))
            if not pairs:
                raise FileNotFoundError(f"{name} 为空")
            print(f"[io] use {name}, pairs={len(pairs)}")
            return [p[0] for p in pairs], [p[1] for p in pairs]

    # 2) timestamp-nearest 配对
    rgb_dir, dep_dir = seq_dir / "rgb", seq_dir / "depth"
    rgbs = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
    deps = sorted(list(dep_dir.glob("*.png")))
    if not rgbs or not deps:
        raise FileNotFoundError("未找到 rgb/*.png(jpg) 或 depth/*.png")

    rgb_ts = [(p, _try_float_stem(p)) for p in rgbs]
    dep_ts = [(p, _try_float_stem(p)) for p in deps]
    rgb_ts = [(p, t) for p, t in rgb_ts if t is not None]
    dep_ts = [(p, t) for p, t in dep_ts if t is not None]
    if not rgb_ts or not dep_ts:
        raise RuntimeError("文件名不是时间戳，且没有 associations.txt")

    dep_times = np.array([t for _, t in dep_ts], dtype=np.float64)
    pairs = []
    for p_rgb, t in rgb_ts:
        j = int(np.argmin(np.abs(dep_times - t)))
        if abs(dep_times[j] - t) <= 0.050:
            pairs.append((p_rgb, dep_ts[j][0]))
    if not pairs:
        raise RuntimeError("时间最近配对失败（阈值太严格？）")
    print(f"[io] timestamp-nearest pairing: pairs={len(pairs)} (rgb={len(rgbs)}, depth={len(deps)})")
    return [p[0] for p in pairs], [p[1] for p in pairs]

def make_rgbd(color_path: Path, depth_path: Path,
              depth_trunc=DEPTH_TRUNC, near_clip=NEAR_CLIP,
              use_center_roi: bool = True):
    c = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if c is None or d is None:
        raise FileNotFoundError(f"读取失败: {color_path.name}, {depth_path.name}")

    dm = depth_to_meters(d).astype(np.float32)
    dm[(dm < near_clip) | (dm > depth_trunc) | ~np.isfinite(dm)] = 0.0

    if (dm > 0).any():
        dm_f = GP.bilateral(dm, d=5, sigmaColor=0.05, sigmaSpace=3)
        dm[dm > 0] = dm_f[dm > 0]

    if use_center_roi:
        H, W = dm.shape
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        rx, ry = ROI_SCALE * W, ROI_SCALE * H
        mask = ((xs - cx) ** 2) / (rx ** 2) + ((ys - cy) ** 2) / (ry ** 2) <= 1.0
        dm[~mask] = 0.0

    color_rgb = GP.bgr2rgb(c)
    return color_rgb, dm

def build_o3d_rgbd(color_rgb: np.ndarray, dm: np.ndarray, depth_trunc: float):
    # 注意：这里 depth_trunc 必须用每帧 far_auto / last_far
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(dm.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False
    )
    return rgbd

def make_pcd_from_rgbd_o3d(rgbd, intr, voxel=0.014):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    return pc

def _is_planar_xyz(X, thr=1e-2, min_n=20):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < min_n:
        return False
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return True
    denom = float(s.sum())
    if denom <= 1e-9:
        return True
    return float(s[-1] / denom) < thr

def orb_pnp_init_cuda(color_prev_rgb, depth_prev_m, color_cur_rgb, intr_mat,
                      max_pts=2500):
    matches, kps0, kps1 = GP.orb_match(color_prev_rgb, color_cur_rgb, max_pts=max_pts)
    if not matches:
        return None

    fx, fy, cx, cy = intr_mat[0, 0], intr_mat[1, 1], intr_mat[0, 2], intr_mat[1, 2]
    objp, imgp = [], []
    dep = depth_prev_m
    zz = dep[(dep > 0) & np.isfinite(dep)]
    if zz.size > 100:
        zmin = float(np.quantile(zz, 0.05)); zmax = float(np.quantile(zz, 0.95))
        zmin = max(zmin, 0.2); zmax = min(zmax, 6.0)
    else:
        zmin, zmax = 0.2, 6.0

    for mm in matches:
        u0, v0 = kps0[mm.queryIdx].pt
        u1, v1 = kps1[mm.trainIdx].pt
        z = dep[int(round(v0)), int(round(u0))]
        if not np.isfinite(z) or z <= zmin or z >= zmax:
            continue
        X = (u0 - cx) * z / fx
        Y = (v0 - cy) * z / fy
        objp.append([X, Y, z])
        imgp.append([u1, v1])

    if len(objp) < 40:
        return None

    objp = np.asarray(objp, np.float32)
    imgp = np.asarray(imgp, np.float32)

    if _is_planar_xyz(objp, thr=0.010):
        return None

    K = intr_mat.astype(np.float64)
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        objp, imgp, K, None,
        iterationsCount=1500,
        reprojectionError=2.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def angle_from_R(R: np.ndarray) -> float:
    t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(t)))

def reasonable(T: np.ndarray, max_t: float, max_r_deg: float) -> bool:
    return (np.linalg.norm(T[:3, 3]) <= max_t) and (angle_from_R(T[:3, :3]) <= max_r_deg)

def fast_overlap_xform(src_pts, dst_pts, T, grid=0.03, sample_n=2000):
    if src_pts.shape[0] == 0 or dst_pts.shape[0] == 0:
        return 0.0
    ids = np.random.choice(src_pts.shape[0], size=min(sample_n, src_pts.shape[0]), replace=False)
    src = src_pts[ids]
    src = (src @ T[:3, :3].T) + T[:3, 3]

    def g(P):
        Q = np.floor(P / grid)
        return (Q[:, 0] * 73856093 + Q[:, 1] * 19349663 + Q[:, 2] * 83492791).astype(np.int64)

    g_dst = set(g(dst_pts).tolist())
    hit = sum((v in g_dst) for v in g(src))
    return hit / max(1, len(ids))

def estimate_depth_bounds_for_file(depth_path: Path, prev_near: float, prev_far: float):
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return prev_near, prev_far
    dm = depth_to_meters(d).astype(np.float32)
    z = dm[(dm > 0) & np.isfinite(dm)]
    if z.size < 500:
        return prev_near, prev_far
    ql = float(np.quantile(z, AUTO_Q_LOW))
    qh = float(np.quantile(z, AUTO_Q_HIGH))
    near = max(0.05, ql - AUTO_MARGIN_NEAR)
    far  = max(near + 0.10, qh + AUTO_MARGIN_FAR)
    near = AUTO_SMOOTH * prev_near + (1.0 - AUTO_SMOOTH) * near
    far  = AUTO_SMOOTH * prev_far  + (1.0 - AUTO_SMOOTH) * far
    far  = max(far, near + 0.10)
    return near, far

def cut_background_otsu(dm: np.ndarray):
    z = dm[(dm > 0) & np.isfinite(dm)]
    if z.size < 1000:
        return dm
    lo = float(np.quantile(z, 0.02)); hi = float(np.quantile(z, 0.98))
    if hi <= lo + 1e-6:
        return dm
    s = np.clip((dm - lo) / (hi - lo), 0.0, 1.0)
    s8 = (s * 255).astype(np.uint8)
    _, thr = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    keep = s8 <= thr
    if keep.sum() < 0.25 * s8.size:
        keep = ~keep
    dm[~keep] = 0.0
    return dm

def _is_low_texture_or_planar(color_rgb: np.ndarray, dm: np.ndarray,
                              pts_prev: np.ndarray, pts_cur: np.ndarray):
    planar = _is_planar_xyz(pts_prev, 0.010) and _is_planar_xyz(pts_cur, 0.010)
    g = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    texture = float(np.mean(np.hypot(gx, gy)))
    low_tex = texture < 8.0
    return planar or low_tex

def load_intrinsics_dataset_aware(w: int, h: int, seq_root: Path):
    root_str = str(seq_root).lower()
    if "fr1" in root_str:
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    elif "fr2" in root_str:
        fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
    elif "fr3" in root_str:
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    else:
        sfx = 525.0 * (w / 640.0)
        sfy = 525.0 * (h / 480.0)
        fx, fy = sfx, sfy
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return dict(fx=fx, fy=fy, cx=cx, cy=cy)

# ==========================
# 10) TSDF Wrapper（你原版保留）
# ==========================
class TSDFWrapper:
    def __init__(self,
                 voxel_len: float,
                 sdf_trunc: float,
                 coarse_factor: float = 3.0,
                 fuse_mode: str = "holefill",
                 fuse_voxel: float = None,
                 hole_dist: float = None,
                 fuse_alpha: float = 0.6,
                 fuse_clamp: float = 0.02):
        self.mode = "gpu" if HAS_O3D_CUDA else "cpu"

        self.voxel_fine = float(voxel_len)
        self.voxel_coarse = float(voxel_len) * max(1.0, float(coarse_factor))
        self.sdf_trunc_fine = float(sdf_trunc)
        self.sdf_trunc_coarse = float(sdf_trunc) * max(1.0, float(coarse_factor))

        self.fuse_mode = (fuse_mode or "holefill").lower()
        self.fuse_voxel = float(fuse_voxel) if (fuse_voxel is not None) else (self.voxel_fine * 1.2)
        self.hole_dist = float(hole_dist) if (hole_dist is not None) else (self.voxel_coarse * 0.9)

        self.fuse_alpha = float(fuse_alpha)
        self.fuse_clamp = float(fuse_clamp)

        if self.mode == "gpu":
            self.dev = o3d.core.Device("CUDA:0")
            try:
                self.grid_fine = o3d.t.geometry.TSDFVoxelGrid(
                    {"tsdf": o3d.core.Dtype.Float32,
                     "weight": o3d.core.Dtype.Float32,
                     "color": o3d.core.Dtype.UInt8},
                    voxel_size=self.voxel_fine,
                    sdf_trunc=self.sdf_trunc_fine,
                    block_resolution=16,
                    block_count=120000,
                    device=self.dev
                )
                self.grid_coarse = o3d.t.geometry.TSDFVoxelGrid(
                    {"tsdf": o3d.core.Dtype.Float32,
                     "weight": o3d.core.Dtype.Float32,
                     "color": o3d.core.Dtype.UInt8},
                    voxel_size=self.voxel_coarse,
                    sdf_trunc=self.sdf_trunc_coarse,
                    block_resolution=16,
                    block_count=80000,
                    device=self.dev
                )
            except TypeError:
                self.grid_fine = o3d.t.geometry.TSDFVoxelGrid(
                    voxel_size=self.voxel_fine, sdf_trunc=self.sdf_trunc_fine, device=self.dev
                )
                self.grid_coarse = o3d.t.geometry.TSDFVoxelGrid(
                    voxel_size=self.voxel_coarse, sdf_trunc=self.sdf_trunc_coarse, device=self.dev
                )
        else:
            self.grid_fine = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_fine,
                sdf_trunc=self.sdf_trunc_fine,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            self.grid_coarse = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_coarse,
                sdf_trunc=self.sdf_trunc_coarse,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

        self.grid = self.grid_fine

    def integrate(self, color_rgb: np.ndarray, depth_m: np.ndarray,
                  K_3x3: np.ndarray, intr_legacy, T_cw: np.ndarray, depth_max: float):
        if self.mode == "gpu":
            depth_t = o3d.t.geometry.Image(
                o3d.core.Tensor(depth_m, dtype=o3d.core.Dtype.Float32, device=self.dev)
            )
            color_t = o3d.t.geometry.Image(
                o3d.core.Tensor(color_rgb, dtype=o3d.core.Dtype.UInt8, device=self.dev)
            )
            K_t = o3d.core.Tensor(K_3x3, dtype=o3d.core.Dtype.Float64, device=self.dev)
            ext_t = o3d.core.Tensor(np.linalg.inv(T_cw), dtype=o3d.core.Dtype.Float64, device=self.dev)

            self.grid_fine.integrate(depth_t, color_t, K_t, ext_t, depth_scale=1.0, depth_max=float(depth_max))
            self.grid_coarse.integrate(depth_t, color_t, K_t, ext_t, depth_scale=1.0, depth_max=float(depth_max))
        else:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_rgb),
                o3d.geometry.Image(depth_m.astype(np.float32)),
                depth_scale=1.0,
                depth_trunc=float(depth_max),
                convert_rgb_to_intensity=False
            )
            Twc = np.linalg.inv(T_cw)
            self.grid_fine.integrate(rgbd, intr_legacy, Twc)
            self.grid_coarse.integrate(rgbd, intr_legacy, Twc)

    def _extract_legacy_pcds(self):
        if self.mode == "gpu":
            pc_fine = self.grid_fine.extract_point_cloud().to_legacy()
            pc_coarse = self.grid_coarse.extract_point_cloud().to_legacy()
        else:
            pc_fine = self.grid_fine.extract_point_cloud()
            pc_coarse = self.grid_coarse.extract_point_cloud()
        return pc_fine, pc_coarse

    def extract_point_cloud(self, mode: str = None) -> o3d.geometry.PointCloud:
        mode = (mode or self.fuse_mode).lower()
        pc_f, pc_c = self._extract_legacy_pcds()

        if mode == "fine":
            return pc_f
        if mode == "coarse":
            return pc_c

        if len(pc_f.points) == 0:
            return pc_c
        if len(pc_c.points) == 0:
            return pc_f

        if mode == "merge":
            pc_all = pc_f + pc_c
            return pc_all.voxel_down_sample(self.fuse_voxel) if self.fuse_voxel > 0 else pc_all

        if mode in ("holefill", "fused"):
            d = np.asarray(pc_c.compute_point_cloud_distance(pc_f), dtype=np.float32)
            keep = np.where(d > self.hole_dist)[0]
            pc_fill = pc_c.select_by_index(keep) if keep.size > 0 else o3d.geometry.PointCloud()
            pc_all = pc_f + pc_fill
            return pc_all.voxel_down_sample(self.fuse_voxel) if self.fuse_voxel > 0 else pc_all

        return pc_f

    def extract_triangle_mesh(self, which: str = "fine") -> o3d.geometry.TriangleMesh:
        which = (which or "fine").lower()
        if which == "coarse":
            if self.mode == "gpu":
                return self.grid_coarse.extract_triangle_mesh().to_legacy()
            return self.grid_coarse.extract_triangle_mesh()
        if self.mode == "gpu":
            return self.grid_fine.extract_triangle_mesh().to_legacy()
        return self.grid_fine.extract_triangle_mesh()

# ==========================
# 11) 线程加载器
# ==========================
class FrameLoader(threading.Thread):
    def __init__(self, colors, depths):
        super().__init__(daemon=True)
        self.colors = colors
        self.depths = depths
        self.q = queue.Queue(maxsize=4)
        self.stop_flag = False
        self.auto_near = NEAR_CLIP
        self.auto_far = DEPTH_TRUNC

    def run(self):
        try:
            for cpath, dpath in zip(self.colors, self.depths):
                if self.stop_flag:
                    break

                if AUTO_DEPTH:
                    self.auto_near, self.auto_far = estimate_depth_bounds_for_file(dpath, self.auto_near, self.auto_far)
                    color_rgb, dm = make_rgbd(
                        cpath, dpath,
                        depth_trunc=self.auto_far,
                        near_clip=self.auto_near,
                        use_center_roi=USE_CENTER_ROI
                    )
                    if AUTO_BG_OTSU:
                        dm = cut_background_otsu(dm)

                    self.q.put((cpath.name, dpath.name, color_rgb, dm, self.auto_near, self.auto_far))
                else:
                    color_rgb, dm = make_rgbd(
                        cpath, dpath,
                        depth_trunc=DEPTH_TRUNC,
                        near_clip=NEAR_CLIP,
                        use_center_roi=USE_CENTER_ROI
                    )
                    self.q.put((cpath.name, dpath.name, color_rgb, dm, NEAR_CLIP, DEPTH_TRUNC))
        except Exception as e:
            self.q.put(("__ERR__", str(e), None, None, None, None))

    def stop(self):
        self.stop_flag = True

# ==========================
# 12) ICP（你原版保留）
# ==========================
def colored_icp_multiscale(src, dst, init=np.eye(4)):
    T = init.copy()

    def prep(pcd, v):
        q = pcd.voxel_down_sample(v)
        if len(q.points) == 0:
            return q
        q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 3, max_nn=48))
        q.orient_normals_towards_camera_location(np.array([0, 0, 0], dtype=np.float64))
        return q

    scales = [0.06, 0.04, 0.02]
    dists  = [0.12, 0.08, 0.05]
    reg_last = None

    for v, corr in zip(scales, dists):
        src_lvl = prep(src, v)
        dst_lvl = prep(dst, v)
        if len(src_lvl.points) < 300 or len(dst_lvl.points) < 300:
            return None

        try:
            with _SilentStderr():
                reg = o3d.pipelines.registration.registration_colored_icp(
                    src_lvl, dst_lvl, corr, T,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=12
                    )
                )
        except Exception:
            reg = None

        if (reg is None) or (getattr(reg, "fitness", 0.0) <= 1e-6):
            try:
                with _SilentStderr():
                    reg = o3d.pipelines.registration.registration_colored_icp(
                        src_lvl, dst_lvl, corr * 1.5, T,
                        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=10
                        )
                    )
            except Exception:
                reg = None

        if reg is None:
            continue

        T = reg.transformation
        reg_last = reg

    return reg_last

def icp_point2plane_multiscale_robust(pc_prev, pc_cur, init=np.eye(4)):
    T = init.copy()
    pyr = [(0.08, 0.20, 20), (0.04, 0.12, 30), (0.02, 0.08, 50)]
    loss = o3d.pipelines.registration.TukeyLoss(k=0.03)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_last = None
    for vxl, radius, it in pyr:
        sp = pc_prev.voxel_down_sample(vxl)
        dp = pc_cur.voxel_down_sample(vxl)
        if len(sp.points) < 300 or len(dp.points) < 300:
            return reg_last
        sp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60))
        dp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60))
        reg = o3d.pipelines.registration.registration_icp(
            sp, dp, radius, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it)
        )
        if reg is None:
            break
        T = reg.transformation
        reg_last = reg
    return reg_last

# ==========================
# 13) 主流程
# ==========================
def main():
    global _map_pcd_sample

    FORCE_GT_ONLY = False

    gt_match = 0
    gt_total = 0

    colors, depths = list_rgbd_paths(SEQ_DIR)
    total = len(colors)

    # ===== GT (TUM groundtruth) =====
    gt_path = SEQ_DIR / "groundtruth.txt"
    gt_data = load_tum_groundtruth(gt_path)

    # complementary filter parameters
    # ===== GT + ICP fusion params (for INTEGRATION pose) =====
    FUSE_ALPHA = 0.08          # 0~1：越大越贴GT，但更抖；一般 0.03~0.15
    FUSE_MAX_R_DEG = 1.0       # 每次更新最多纠正 1 deg
    FUSE_MAX_T = 0.01          # 每次更新最多纠正 1 cm
    FUSE_REJECT_R_DEG = 45.0   # 如果GT与估计差太离谱，认为GT匹配异常/时间错，直接拒绝
    FUSE_REJECT_T = 0.8

    # 可选：把GT当“低频外部定位”，每 N 个 keyframe 用一次（论文叙事更自然）
    GT_STRIDE = 2  # 1=每次用，2=每2个keyframe用一次，5=更低频

    # state: global pose used for TSDF integration
    T_cw = np.eye(4, dtype=np.float64)   # camera -> GLOBAL (this is what you integrate with)


    # ==========================
    # GT as absolute reference for SUBMAP
    # ==========================
    USE_GT_CONSTRAINT = (gt_data is not None)
    GT_MAX_DT = 0.05

    # --- submap cut params (drift budget) ---
    DEV_T_HI = 0.12            # meters  (trigger threshold)
    DEV_R_HI_DEG = 8.0         # degrees
    DEV_T_LO = 0.08            # meters  (release threshold, hysteresis)
    DEV_R_LO_DEG = 6.0         # degrees
    CUT_N = 4                  # consecutive deviated frames to cut
    COOLDOWN_FRAMES = 12       # after cut, ignore cutting for a while

    # --- global accumulation / visualization ---
    GLOBAL_DS = 0.02           # meters, downsample for global merged pcd
    SAVE_SUBMAPS = True        # whether to save each submap mesh/pcd when cut

    # state: GT alignment
    S_align = np.eye(4, dtype=np.float64)     # GT-world -> GLOBAL(world)
    last_gt_ts = None

    # state: submap
    T_sub_global = np.eye(4, dtype=np.float64)   # submapLocal -> GLOBAL
    T_local = np.eye(4, dtype=np.float64)        # camera -> submapLocal (high-freq)
    bad_cnt = 0
    cooldown = 0
    submap_idx = 0

    # ===== visualization-only GT correction (does NOT affect integration) =====
    T_sub_global_vis = T_sub_global.copy()
    VIS_ALPHA = 0.25          # 0~1, larger => closer to GT but more jitter
    VIS_MAX_R_DEG = 2.0       # per-update rotation clamp (deg)
    VIS_MAX_T = 0.03          # per-update translation clamp (m)

    global_pcd = o3d.geometry.PointCloud()        # merged finalized submaps (GLOBAL)


    # ===== 找首个有效深度帧 =====
    start = 0
    near0, far0 = 0.15, 5.0
    while start < len(colors):
        c0, d0 = colors[start], depths[start]
        near0, far0 = estimate_depth_bounds_for_file(d0, near0, far0)
        color0_rgb, dm0 = make_rgbd(c0, d0, depth_trunc=far0, near_clip=near0, use_center_roi=False)
        z0 = dm0[(dm0 > 0) & np.isfinite(dm0)]
        print(f"[bootstrap] k0={start}, nz={z0.size}, near={near0:.2f}, far={far0:.2f}")
        if z0.size >= 3000:
            break
        start += 1
    if start == len(colors):
        raise RuntimeError("没有找到有效深度的起始帧")

    H, W = dm0.shape
    Kd = load_intrinsics_dataset_aware(W, H, SEQ_DIR)
    K = np.array([[Kd["fx"], 0, Kd["cx"]],
                  [0, Kd["fy"], Kd["cy"]],
                  [0, 0, 1]], np.float64)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, Kd["fx"], Kd["fy"], Kd["cx"], Kd["cy"])
    print(f"[K] fx={Kd['fx']:.1f} fy={Kd['fy']:.1f} cx={Kd['cx']:.1f} cy={Kd['cy']:.1f}")

    # 关键修复：rgbd0 用 far0
    rgbd0 = build_o3d_rgbd(color0_rgb, dm0, depth_trunc=far0)

    # ===== init GT alignment using the first valid frame =====
    GT_TIME_SHIFT = 0.0

    if USE_GT_CONSTRAINT:
        t0 = _try_float_stem(colors[start])
        if t0 is None:
            print("[gt] cannot parse start timestamp from filename (GT fusion disabled)")
            USE_GT_CONSTRAINT = False
        else:
            ts_gt, _ = gt_data
            # --- estimate a time shift so that t0+shift is near GT timeline ---
            j0 = int(np.argmin(np.abs(ts_gt - t0)))
            # if already close, shift=0; otherwise shift to the nearest GT sample
            GT_TIME_SHIFT = float(ts_gt[j0] - t0)

            # sanity: if the gap is huge, still allow (it just means different epoch)
            print(f"[gt] time shift estimated: GT_TIME_SHIFT={GT_TIME_SHIFT:.6f} (sec), |dt|={abs(ts_gt[j0]-t0):.3f}")

            # now query using shifted time
            T_gt0 = query_gt_nearest(gt_data, t0 + GT_TIME_SHIFT, max_dt=max(GT_MAX_DT, 0.5))
            if T_gt0 is None:
                print(f"[gt] no GT near (t0+shift)={t0+GT_TIME_SHIFT:.6f} (GT fusion disabled)")
                USE_GT_CONSTRAINT = False
            else:
                # align GT so that the first pose becomes Identity in TSDF-world
                S_align = np.linalg.inv(T_gt0)
                last_gt_ts = t0 + GT_TIME_SHIFT
                print(f"[gt] enabled. alignment set by first frame, t0={t0:.6f}")



    tsdf = TSDFWrapper(
        VOXEL_LEN,
        VOXEL_LEN * SDF_TRUNC_K,
        coarse_factor=3.0,
        fuse_mode="holefill",
        fuse_voxel=VOXEL_LEN * 1.2,
        hole_dist=VOXEL_LEN * 3.0 * 0.9
    )

    # submap-local pose used for TSDF integration
    T_cw = np.eye(4, dtype=np.float64)
    T_local = np.eye(4, dtype=np.float64)  # camera->submapLocal

    def _add_current_submap_to_global(tsdf_obj, T_sub_glb, ds=GLOBAL_DS):
        """Extract a coarse pcd from current tsdf, transform to GLOBAL, return downsampled."""
        pcd_loc = tsdf_obj.extract_point_cloud(mode="coarse")
        if len(pcd_loc.points) == 0:
            return o3d.geometry.PointCloud()
        pcd_loc = pcd_loc.voxel_down_sample(ds)
        pcd_loc.transform(T_sub_glb)
        return pcd_loc

    def _finalize_submap(tsdf_obj, T_sub_glb_used, submap_id: int):
        nonlocal global_pcd
        pcd_g = _add_current_submap_to_global(tsdf_obj, T_sub_glb_used, ds=GLOBAL_DS)
        if len(pcd_g.points) > 0:
            global_pcd += pcd_g
            global_pcd = global_pcd.voxel_down_sample(GLOBAL_DS)

        if SAVE_SUBMAPS:
            ts = time.strftime("%Y%m%d_%H%M%S")
            mesh = tsdf_obj.extract_triangle_mesh("fine")
            mesh.compute_vertex_normals()
            mesh.transform(T_sub_glb_used)
            o3d.io.write_triangle_mesh(str(OUT_DIR / f"submap_{submap_id:03d}_{ts}.ply"), mesh)
            if len(pcd_g.points) > 0:
                o3d.io.write_point_cloud(str(OUT_DIR / f"submap_{submap_id:03d}_{ts}_pcd.ply"), pcd_g)
            print(f"[submap] saved submap {submap_id}")


    def _reset_new_submap(anchor_T_global: np.ndarray):
        """Start a new empty TSDF submap anchored at anchor_T_global (camera pose in GLOBAL for the first frame)."""
        nonlocal tsdf, T_sub_global, T_local, bad_cnt, cooldown, submap_idx
        _finalize_submap(tsdf, T_sub_global, submap_idx)
        submap_idx += 1

        # new volume
        tsdf = TSDFWrapper(
            VOXEL_LEN,
            VOXEL_LEN * SDF_TRUNC_K,
            coarse_factor=3.0,
            fuse_mode="holefill",
            fuse_voxel=VOXEL_LEN * 1.2,
            hole_dist=VOXEL_LEN * 3.0 * 0.9
        )

        T_sub_global = anchor_T_global.copy()   # IMPORTANT: anchor to GT in GLOBAL
        T_local = np.eye(4, dtype=np.float64)   # restart local odom at identity
        T_sub_global_vis = T_sub_global.copy()
        bad_cnt = 0
        cooldown = COOLDOWN_FRAMES
        print(f"[submap] start new submap {submap_idx} anchored to GT")


    tsdf.integrate(color0_rgb, dm0, K, intr, T_cw, depth_max=far0)
    _pc0 = tsdf.extract_point_cloud(mode="coarse")
    print("[dbg] after first integrate coarse pts =", len(_pc0.points))

    print(f"[online] size={W}x{H}, depth[min,max]={float(dm0[dm0>0].min()) if (dm0>0).any() else 0:.3f}/{float(dm0.max()) if (dm0>0).any() else 0:.3f}")

    cv2.namedWindow("Left RGB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left RGB", 640, 480)
    cv2.imshow("Left RGB", cv2.cvtColor(color0_rgb, cv2.COLOR_RGB2BGR))

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Right Reconstruction", width=960, height=720, visible=True)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(coord)
    pcd_show = tsdf.extract_point_cloud()
    vis.add_geometry(pcd_show)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])
    render_opt.point_size = 2.0

    loader = FrameLoader(colors[start + 1:], depths[start + 1:])
    loader.start()

    last_color = color0_rgb
    last_dm = dm0
    last_rgbd = rgbd0
    last_far = far0          # 关键修复：记住上一帧的 far，用于 build_o3d_rgbd(prev)
    ov_after = 0.0

    # tracking reference (ONLY update when a frame is accepted/integrated)
    trk_color = color0_rgb
    trk_dm    = dm0
    trk_rgbd  = rgbd0
    trk_far   = far0

    processed = 1
    show_mesh = SHOW_MESH
    ref_plane = None

    try:
        k = 1
        while k < total:
            try:
                item = loader.q.get(timeout=1.0)
            except queue.Empty:
                # 保持 UI 活着（否则 Windows 会直接显示未响应）
                vis.poll_events(); vis.update_renderer()
                cv2.waitKey(1)

                if not loader.is_alive():
                    print("[main] loader finished and queue empty -> exit")
                    break
                continue
            if item[0] == "__ERR__":
                print("[loader-error]", item[1])
                break

            name_c, name_d, color_rgb, dm, near_auto, far_auto = item
            # ===== Sanity A: GT match ratio (independent of pass_gate) =====
            if USE_GT_CONSTRAINT:
                t_img_dbg = _try_float_stem(Path(name_c))
                if t_img_dbg is not None:
                    t_q = t_img_dbg + GT_TIME_SHIFT
                    T_gt_dbg = query_gt_nearest(gt_data, t_q, max_dt=GT_MAX_DT)
                    gt_total += 1
                    gt_match += int(T_gt_dbg is not None)

                    if (gt_total % 20) == 0:
                        ts_gt, _ = gt_data
                        j = int(np.argmin(np.abs(ts_gt - t_q)))
                        dt_near = float(abs(ts_gt[j] - t_q))
                        print(f"[gt] match={gt_match}/{gt_total}={gt_match/max(1,gt_total):.2f}, "
                             f"GT_MAX_DT={GT_MAX_DT}, GT_TIME_SHIFT={GT_TIME_SHIFT:.6f}, dt_near={dt_near:.4f}")
                else:
                    gt_total += 1
                    if (gt_total % 20) == 0:
                        print(f"[gt] match={gt_match}/{gt_total}={gt_match/max(1,gt_total):.2f} (timestamp parse failed)")

            cv2.imshow("Left RGB", cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))

            if (k % KEYFRAME_STEP) == 0:
                # 关键修复：rgbd_prev / rgbd_cur 使用 last_far / far_auto
                rgbd_prev = build_o3d_rgbd(trk_color, trk_dm, depth_trunc=trk_far)
                rgbd_cur  = build_o3d_rgbd(color_rgb, dm, depth_trunc=far_auto)

                pc_prev = make_pcd_from_rgbd_o3d(rgbd_prev, intr, voxel=0.014)
                pc_cur  = make_pcd_from_rgbd_o3d(rgbd_cur,  intr, voxel=0.014)

                if len(pc_prev.points) < MIN_PTS_FOR_ICP or len(pc_cur.points) < MIN_PTS_FOR_ICP:
                    print(f"[k={k:04d}] skip: too few points prev={len(pc_prev.points)} cur={len(pc_cur.points)} "
                          f"near={near_auto:.2f} far={far_auto:.2f} last_far={last_far:.2f}")
                    last_color = color_rgb
                    last_dm = dm
                    last_rgbd = rgbd_cur
                    last_far = far_auto
                    k += 1
                    vis.poll_events(); vis.update_renderer()
                    continue

                def fast_overlap(p_prev_pts, p_cur_pts, sample_n=1200):
                    if p_prev_pts.shape[0] == 0 or p_cur_pts.shape[0] == 0:
                        return 0.0
                    ids = np.random.choice(p_prev_pts.shape[0], size=min(sample_n, p_prev_pts.shape[0]), replace=False)
                    src = p_prev_pts[ids]
                    grid = 0.04
                    g_prev = np.floor(p_cur_pts / grid)
                    g_prev = g_prev[:, 0] * 73856093 + g_prev[:, 1] * 19349663 + g_prev[:, 2] * 83492791
                    g_prev = set(g_prev.tolist())
                    g_src = np.floor(src / grid)
                    g_src = g_src[:, 0] * 73856093 + g_src[:, 1] * 19349663 + g_src[:, 2] * 83492791
                    hit = sum((v in g_prev) for v in g_src)
                    return hit / max(1, len(ids))

                pts_prev = np.asarray(pc_prev.points)
                pts_cur = np.asarray(pc_cur.points)
                ov = fast_overlap(pts_prev, pts_cur)

                # 低重叠：放宽深度窗再读一次，并同步更新 near_auto / far_auto（修复点）
                if ov < LOW_OV_THR:
                    zz = dm[(dm > 0) & np.isfinite(dm)]
                    near_this, far_this, use_roi_this = NEAR_CLIP, DEPTH_TRUNC, False
                    if zz.size > 100:
                        ql, qh = np.quantile(zz, 0.03), np.quantile(zz, 0.97)
                        near_this = float(max(0.05, ql - 0.03))
                        far_this  = float(max(near_this + 0.3, qh + 0.05))

                    # 注意：这里依赖 SEQ_DIR/rgb 与 SEQ_DIR/depth 结构
                    color_rgb, dm = make_rgbd(SEQ_DIR / "rgb" / name_c, SEQ_DIR / "depth" / name_d,
                                              depth_trunc=far_this, near_clip=near_this, use_center_roi=use_roi_this)

                    near_auto, far_auto = near_this, far_this  # 关键修复：后面用新的 far_auto

                    rgbd_cur = build_o3d_rgbd(color_rgb, dm, depth_trunc=far_auto)
                    pc_cur = make_pcd_from_rgbd_o3d(rgbd_cur, intr, voxel=0.014)
                    if len(pc_cur.points) < MIN_PTS_FOR_ICP:
                        print(f"[k={k:04d}] skip: low-ov re-read but still too few points cur={len(pc_cur.points)} "
                              f"near={near_auto:.2f} far={far_auto:.2f}")
                        last_color = color_rgb
                        last_dm = dm
                        last_rgbd = rgbd_cur
                        last_far = far_auto
                        k += 1
                        vis.poll_events(); vis.update_renderer()
                        continue

                # 初值：ORB+PnP
                # 初值候选：I / PnP / FPFH
                init_candidates = [("I", np.eye(4, dtype=np.float64))]

                T_pnp = orb_pnp_init_cuda(trk_color, trk_dm, color_rgb, K)
                if (T_pnp is not None) and reasonable(T_pnp, 0.30, 20.0):
                    init_candidates.append(("pnp", T_pnp))

                if USE_FPFH_INIT and (len(pc_prev.points) > 0) and (len(pc_cur.points) > 0):
                    try:
                        p1 = pc_prev.voxel_down_sample(FPFH_VOXEL)
                        p2 = pc_cur.voxel_down_sample(FPFH_VOXEL)
                        if len(p1.points) > 0 and len(p2.points) > 0:
                            p1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_VOXEL * 3, max_nn=48))
                            p2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_VOXEL * 3, max_nn=48))
                            f1 = o3d.pipelines.registration.compute_fpfh_feature(
                                p1, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_VOXEL * 5, max_nn=96)
                            )
                            f2 = o3d.pipelines.registration.compute_fpfh_feature(
                                p2, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_VOXEL * 5, max_nn=96)
                            )
                            res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                                p1, p2, f1, f2, True, RANSAC_DIST,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
                                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DIST)],
                                o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_N_IT, 500)
                            )
                            if (res is not None) and (res.transformation is not None) and reasonable(res.transformation, 0.30, 25.0):
                                init_candidates.append(("fpfh", res.transformation))
                    except Exception:
                        pass

                # 用 overlap_xform 评分选最优初值（都在 prev->cur 的同一语义下）
                pts_prev = np.asarray(pc_prev.points)
                pts_cur  = np.asarray(pc_cur.points)

                def score_init(T):
                    return fast_overlap_xform(pts_prev, pts_cur, T, grid=0.03, sample_n=1500)

                best_name, initT = max(init_candidates, key=lambda kv: score_init(kv[1]))
                print(f"[init] choose {best_name} score={score_init(initT):.3f}")

                reg = None
                prefer_geometric = _is_low_texture_or_planar(color_rgb, dm, np.asarray(pc_prev.points), np.asarray(pc_cur.points))

                if (not prefer_geometric) and (ov >= 0.06) and reasonable(initT, 0.30, 20.0):
                    reg = colored_icp_multiscale(pc_prev, pc_cur, init=initT)

                if reg is None:
                    reg = icp_point2plane_multiscale_robust(pc_prev, pc_cur, init=initT)

                # 兜底 ICP（只在 reg None 时）
                if reg is None:
                    pc_prev2 = pc_prev.voxel_down_sample(0.02)
                    pc_cur2 = pc_cur.voxel_down_sample(0.02)
                    pc_prev2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=48))
                    pc_cur2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=48))
                    try:
                        with _SilentStderr():
                            reg = o3d.pipelines.registration.registration_icp(
                                pc_prev2, pc_cur2, 0.15, initT,
                                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
                            )
                    except Exception:
                        reg = None

                ok = (reg is not None)
                fit, rmse = -1.0, -1.0
                if ok:
                    Tij = reg.transformation
                    fit = float(getattr(reg, "fitness", 0.0))
                    rmse = float(getattr(reg, "inlier_rmse", 999.0))
                    ok = (fit >= ICP_MIN_FIT_ODO) and (rmse <= ICP_MAX_RMSE_ODO) and reasonable(Tij, ODOM_MAX_T, ODOM_MAX_R_DEG)

                # 兜底 RGB-D odometry
                if not ok:
                    try:
                        with _SilentStderr():
                            ok_odom, T_odom, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                                rgbd_prev, rgbd_cur, intr, np.eye(4),
                                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                o3d.pipelines.odometry.OdometryOption()
                            )
                        if ok_odom and reasonable(T_odom, ODOM_MAX_T, ODOM_MAX_R_DEG):
                            Tij = T_odom
                            ok = True
                            fit, rmse = 0.5, 0.02
                    except Exception:
                        pass

                # 若仍不 ok，Tij 不存在就直接跳
                if not ok:
                    print(f"[k={k:04d}] fail: reg/odom not ok  ov={ov:.3f} fit={fit:.3f} rmse={rmse:.4f}")
                    last_color = color_rgb
                    last_dm = dm
                    last_rgbd = rgbd_cur
                    last_far = far_auto
                    k += 1
                    vis.poll_events(); vis.update_renderer()
                    continue

                T_local_cand = T_local @ np.linalg.inv(Tij)

                # 二次重叠门控（在融合前）
                ov_after = fast_overlap_xform(
                    np.asarray(pc_prev.points),
                    np.asarray(pc_cur.points),
                    Tij,
                    grid=0.03, sample_n=1500
                )

                # tracking vs integration 两个阈值分开
                TRACK_OV_MIN     = 0.10   # 让 tracking 连续
                INTEGRATE_OV_MIN = 0.20   # integrate 更严格（你也可先降到 0.20 看看）

                pass_track = (ok and (ov_after >= TRACK_OV_MIN))
                pass_int   = (ok and (ov_after >= INTEGRATE_OV_MIN))

                if not pass_track:
                    print(f"[k={k:04d}] track-reject ov_after={ov_after:.3f} fit={fit:.3f} rmse={rmse:.4f}")
                    # 注意：这里不要更新 trk_*，因为 tracking 都不信
                    k += 1
                    vis.poll_events(); vis.update_renderer()
                    continue

                # ---- 1) tracking: 先提交 pose + 更新 tracking reference（保证连续性）----
                T_local = T_local_cand
                trk_color = color_rgb
                trk_dm    = dm
                trk_rgbd  = rgbd_cur
                trk_far   = far_auto

                # ---- 2) GT cut 判定：用更新后的 T_local（或用 cand 都一样）----
                cut_now = False
                
                if USE_GT_CONSTRAINT:
                    t_img = _try_float_stem(Path(name_c))
                    if t_img is not None:
                        t_q = t_img + GT_TIME_SHIFT
                        T_gt = query_gt_nearest(gt_data, t_q, max_dt=GT_MAX_DT)

                        if T_gt is not None:
                            T_gt_aligned = S_align @ T_gt          # camera -> GLOBAL(cam0)
                            T_est_global = T_sub_global @ T_local  # camera -> GLOBAL(cam0)

                            # --- (1) visualization-only smoothing ---
                            C_full = T_gt_aligned @ np.linalg.inv(T_est_global)  # GLOBAL->GLOBAL
                            xi = se3_log(C_full)
                            xi = clamp_twist(
                                xi * VIS_ALPHA,
                                max_r_rad=np.deg2rad(VIS_MAX_R_DEG),
                                max_t=VIS_MAX_T
                            )
                            C_smooth = se3_exp(xi)
                            T_sub_global_vis = C_smooth @ T_sub_global

                            # --- (2) cut decision uses full correction ---
                            xi_e = se3_log(C_full)
                            err_t = float(np.linalg.norm(xi_e[3:]))
                            err_r = float(np.linalg.norm(xi_e[:3]))

                            cut_now = False
                            if cooldown > 0:
                                cooldown -= 1
                                bad_cnt = 0
                            else:
                                hi = (err_t > DEV_T_HI) or (err_r > np.deg2rad(DEV_R_HI_DEG))
                                lo = (err_t < DEV_T_LO) and (err_r < np.deg2rad(DEV_R_LO_DEG))
                
                                if hi:
                                    bad_cnt += 1
                                elif lo:
                                    bad_cnt = 0

                                if bad_cnt >= CUT_N:
                                    cut_now = True
                                    bad_cnt = 0

                            if (processed % 50) == 0:
                                print(f"[gt-sub] err_t={err_t:.3f}m err_r={np.degrees(err_r):.2f}deg "
                                      f"bad_cnt={bad_cnt} cut={int(cut_now)} cooldown={cooldown}")

                            # --- (3) DO CUT HERE (this is what you were missing) ---
                            if cut_now:
                                # 用 full correction 把旧 submap 放回 GT 对齐的 GLOBAL
                                T_sub_glb_used = C_full @ T_sub_global
                                _finalize_submap(tsdf, T_sub_glb_used, submap_idx)
                                submap_idx += 1

                                # 新 submap：锚定 GT
                                tsdf = TSDFWrapper(
                                    VOXEL_LEN,
                                    VOXEL_LEN * SDF_TRUNC_K,
                                    coarse_factor=3.0,
                                    fuse_mode="holefill",
                                    fuse_voxel=VOXEL_LEN * 1.2,
                                    hole_dist=VOXEL_LEN * 3.0 * 0.9
                                )
                                T_sub_global = T_gt_aligned.copy()
                                T_sub_global_vis = T_sub_global.copy()
                                T_local = np.eye(4, dtype=np.float64)
                
                                cooldown = COOLDOWN_FRAMES
                                bad_cnt = 0

                                # 当前帧在新 submap 里积分（local = I）
                                if pass_int:
                                    tsdf.integrate(color_rgb, dm, K, intr, np.eye(4), depth_max=far_auto)
                                    processed += 1
                                continue

                        else:
                            # GT 缺失：只做可视化回退
                            T_sub_global_vis = T_sub_global
                    else:
                        # timestamp parse failed
                        T_sub_global_vis = T_sub_global

                # ---- 3) integrate: 更严格的门控 ----
                if pass_int:
                    tsdf.integrate(color_rgb, dm, K, intr, T_local, depth_max=far_auto)
                    processed += 1

                if (processed % MAP_SAMPLE_EVERY) == 0:
                    _map_pcd_sample = tsdf.extract_point_cloud(mode="coarse").voxel_down_sample(0.03)

            # 可视化更新
            if (processed % LIVE_UPDATE_EVERY) == 0:
                if show_mesh:
                    mesh = tsdf.extract_triangle_mesh()
                    mesh.compute_vertex_normals()
                    vis.clear_geometries()
                    vis.add_geometry(coord)
                    vis.add_geometry(mesh)
                else:
                    # show GLOBAL = (finalized submaps) + (current submap transformed)
                    pcd_cur_g = _add_current_submap_to_global(tsdf, T_sub_global_vis, ds=GLOBAL_DS)

                    if len(global_pcd.points) > 0:
                        pcd_vis = global_pcd + pcd_cur_g
                    else:
                        pcd_vis = pcd_cur_g

                    # keep it light
                    if len(pcd_vis.points) > 0:
                        pcd_vis = pcd_vis.voxel_down_sample(GLOBAL_DS)

                    pcd_show.points = pcd_vis.points
                    pcd_show.colors = pcd_vis.colors
                    vis.update_geometry(pcd_show)


            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                ts = time.strftime("%Y%m%d_%H%M%S")
                mesh = tsdf.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                pcd_cur_g = _add_current_submap_to_global(tsdf, T_sub_global, ds=GLOBAL_DS)
                pcd_save = global_pcd + pcd_cur_g if len(global_pcd.points) > 0 else pcd_cur_g
                o3d.io.write_point_cloud(str(OUT_DIR / f"pcd_global_live_{ts}.ply"), pcd_save)
                print("[save]", ts)
            if key in (ord("m"), ord("M")):
                show_mesh = not show_mesh

            vis.poll_events()
            vis.update_renderer()
            k += 1

        print(f"[done-live] processed keyframes = {processed}")
        # finalize last submap into GLOBAL
        _finalize_submap(tsdf, T_sub_global, submap_idx)

        ts = time.strftime("%Y%m%d_%H%M%S")
        print("[dbg] global pts =", len(global_pcd.points))
        pcd_path = OUT_DIR / f"pcd_global_{ts}.ply"
        o3d.io.write_point_cloud(str(pcd_path), global_pcd)
        print(f"[final] saved GLOBAL pcd:\n  {pcd_path}")

        try:
            vis.clear_geometries()
            vis.add_geometry(coord)

            # show GLOBAL = finalized global_pcd + current submap transformed
            pcd_cur_g = _add_current_submap_to_global(tsdf, T_sub_global, ds=GLOBAL_DS)
            pcd_vis = global_pcd + pcd_cur_g if len(global_pcd.points) > 0 else pcd_cur_g
            if len(pcd_vis.points) > 0:
                pcd_vis = pcd_vis.voxel_down_sample(GLOBAL_DS)

            vis.add_geometry(pcd_vis)
            vis.poll_events(); vis.update_renderer()
        except Exception as e:
            print("[final-vis] failed:", repr(e))


        print("[final] 按 ESC / Q 退出；或直接关闭任一窗口。")
        while True:
            alive = vis.poll_events()
            if alive:
                vis.update_renderer()
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if not alive and cv2.getWindowProperty("Left RGB", cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception:
        logp = OUT_DIR / "live_crash_nocupoch.log"
        with open(logp, "a", encoding="utf-8") as f:
            f.write("".join(traceback.format_exc()) + "\n")
        print(f"[fatal] 运行异常，详见日志 {logp}")

    finally:
        try:
            vis.destroy_window()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()