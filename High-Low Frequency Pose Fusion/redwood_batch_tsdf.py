# -*- coding: utf-8 -*-
import time
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import random
import io, contextlib
from typing import Tuple, List

# ===================== 路径与数据集 =====================
SEQ_DIR     = Path(r"D:\Projects\Underwater\Datasets\redwood\02832")
OUT_DIR     = Path(r"D:\Projects\Underwater\Outputs\redwood\02832")

# ===================== 在线流式参数 =====================
TARGET_FPS           = 4          # 目标输入帧率（不追求严格实时，节流用）
KEYFRAME_STEP        = 1          # 关键帧间隔（1=逐帧处理；>1=抽帧）
ONLINE_OPT_EVERY_N   = 120        # 每处理多少（关键）帧做一次全局优化
ONLINE_OPT_MAX_TIME  = 8.0        # （提示用）本次优化的最大预算秒数（Open3D无硬预算，这里仅备注）
REINTEGRATE_AFTER_OPT= True       # 优化后是否清空TSDF并重融（精度优先）

ENABLE_ONLINE_LOOPS  = True       # 在线回环开关
LOOP_CAND_EVERY      = 30         # 每隔多少新关键帧尝试回环
LOOP_INTERVAL        = 25         # 回环候选的帧间隔（关键帧尺度）
LOOP_MAX_BACK        = 6          # 回看多少个间隔（最多尝试 LOOP_MAX_BACK 个间隔）

# 可选：用 FPFH+RANSAC 求粗初值（弱纹理/水下环境非常有用）
USE_FPFH_INIT        = True
FPFH_VOXEL           = 0.02       # 计算FPFH的体素
RANSAC_DIST          = 0.03       # RANSAC 内点阈值（米）
RANSAC_N_ITER        = 50000
RANSAC_CONF          = 0.999

# ===================== I/O/采样 =====================
MAX_FRAMES      = None            # 限制总帧数（调试用）；None 则不限制
MIN_PTS_FOR_ICP = 2000            # 每帧下采样后的最少点数，低于阈值跳过

# ===================== 深度窗口 =====================
NEAR_CLIP   = 0.60
DEPTH_TRUNC = 1.60
AUTO_DEPTH_WINDOW = False
AUTO_SAMPLES       = 30
AUTO_Q_LOW, AUTO_Q_HIGH = 0.05, 0.95
AUTO_MARGIN_NEAR, AUTO_MARGIN_FAR = 0.03, 0.05

# ===================== 预处理 =====================
DOWNSCALE   = 1                   # 输入太小就保持1
USE_CENTER_ROI = True
ROI_SCALE      = 0.38             # 椭圆ROI相对宽高比例

# ===================== TSDF 参数 =====================
VOXEL       = 0.007               # TSDF 体素（米）
SDF_TRUNC_K = 5.0                 # 截断倍数：sdf_trunc = VOXEL * K

# ===================== 位姿图/配准参数 =====================
PREF_LOOP         = 0.10          # 全局优化里回环权重偏好（越小越信任里程计）
ICP_FINE_DIST     = 0.04          # 细层对应门限（米）
ICP_MIN_FIT_ODO   = 0.40          # 相邻边最小 fitness
ICP_MAX_RMSE_ODO  = 0.012         # 相邻边最大 inlier rmse
ICP_MIN_FIT_LP    = 0.55          # 回环边最小 fitness
ICP_MAX_RMSE_LP   = 0.010         # 回环边最大 rmse
ODOM_MAX_T        = 0.12          # 相邻帧最大平移（米）
ODOM_MAX_R_DEG    = 12.0          # 相邻帧最大自转（度）
LOOP_MAX_T        = 0.70          # 回环候选最大平移（米）
LOOP_MAX_R_DEG    = 30.0          # 回环候选最大自转（度）

# ===================== 日志控制 =====================
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class _SilentStderr:
    """上下文管理器：暂时重定向 stderr，屏蔽 Open3D 内部异常打印（如 No correspondences）。"""
    def __enter__(self):
        self._buf = io.StringIO()
        self._cm = contextlib.redirect_stderr(self._buf)
        self._cm.__enter__()
        return self._buf
    def __exit__(self, exc_type, exc, tb):
        self._cm.__exit__(exc_type, exc, tb)

# ===================== 常用工具函数 =====================
def depth_to_meters(depth_raw: np.ndarray):
    if depth_raw.dtype == np.uint16:
        d = depth_raw.astype(np.float32)
        return d / 1000.0 if d.max() > 100.0 else d
    d = depth_raw.astype(np.float32)
    return d if d.max() < 1000 else d / 1000.0

def load_intrinsics_default(w: int, h: int):
    fx = 525.0 * (w / 640.0)
    fy = 525.0 * (h / 480.0)
    cx = w/2.0 - 0.5
    cy = h/2.0 - 0.5
    return dict(fx=fx, fy=fy, cx=cx, cy=cy)

def make_rgbd(color_path: Path, depth_path: Path,
              depth_trunc=1.6, near_clip=0.6, downscale=1):
    c = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    assert c is not None and d is not None, f"读取失败: {color_path.name}, {depth_path.name}"
    if downscale != 1:
        c = cv2.resize(c, (c.shape[1]//downscale, c.shape[0]//downscale), interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, (d.shape[1]//downscale, d.shape[0]//downscale), interpolation=cv2.INTER_NEAREST)

    dm = depth_to_meters(d).astype(np.float32)
    dm[(dm < near_clip) | (dm > depth_trunc) | ~np.isfinite(dm)] = 0.0
    if (dm > 0).any():
        dm_f = cv2.bilateralFilter(dm, d=5, sigmaColor=0.05, sigmaSpace=3)
        dm[dm > 0] = dm_f[dm > 0]

    if USE_CENTER_ROI:
        H, W = dm.shape
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        cx, cy = (W-1)/2.0, (H-1)/2.0
        rx, ry = ROI_SCALE*W, ROI_SCALE*H
        mask = ((xs-cx)**2)/(rx**2) + ((ys-cy)**2)/(ry**2) <= 1.0
        dm[~mask] = 0.0

    color_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    assert color_rgb.shape[:2] == dm.shape[:2], f"color/depth 尺寸不一致: {color_rgb.shape[:2]} vs {dm.shape[:2]}"

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(dm),
        depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )
    return rgbd, c.shape[1], c.shape[0], dm

def list_rgbd_paths(seq_dir: Path):
    color_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    colors = sorted(list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.png")))
    depths = sorted(list(depth_dir.glob("*.png")))
    n = min(len(colors), len(depths))
    assert n > 0, "未找到有效的彩色/深度帧"
    return colors[:n], depths[:n]

def estimate_depth_window(color_list, depth_list):
    vals = []
    K = min(AUTO_SAMPLES, len(depth_list))
    for i in range(K):
        _, _, _, dm = make_rgbd(color_list[i], depth_list[i],
                                depth_trunc=10.0, near_clip=0.0, downscale=DOWNSCALE)
        v = dm[(dm > 0) & np.isfinite(dm)]
        if v.size:
            vals.append(v)
    if not vals:
        return NEAR_CLIP, DEPTH_TRUNC
    vals = np.concatenate(vals)
    ql = float(np.quantile(vals, AUTO_Q_LOW))
    qh = float(np.quantile(vals, AUTO_Q_HIGH))
    near = max(0.05, ql + AUTO_MARGIN_NEAR)
    far  = max(near + 0.2, qh - AUTO_MARGIN_FAR)
    return near, far

def overlap_ratio(src: o3d.geometry.PointCloud,
                  dst: o3d.geometry.PointCloud,
                  T_src_to_dst: np.ndarray,
                  radius: float = 0.06,
                  sample_n: int = 2000) -> float:
    """估计 src 经 T 后与 dst 的重叠比例（半径邻域命中率）。"""
    if len(src.points) == 0 or len(dst.points) == 0:
        return 0.0
    src_t = o3d.geometry.PointCloud(src)
    src_t.transform(T_src_to_dst)
    kd = o3d.geometry.KDTreeFlann(dst)
    ids = random.sample(range(len(src_t.points)), min(sample_n, len(src_t.points)))
    hit = 0
    pts = np.asarray(src_t.points)
    for i in ids:
        k, _, _ = kd.search_radius_vector_3d(pts[i], radius)
        if k > 0:
            hit += 1
    return hit / max(1, len(ids))

def _angle_from_R(R: np.ndarray) -> float:
    t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(t)))

def _reasonable(T: np.ndarray, max_t: float, max_r_deg: float) -> bool:
    return (np.linalg.norm(T[:3, 3]) <= max_t) and (_angle_from_R(T[:3, :3]) <= max_r_deg)

# ===================== 粗初值（FPFH+RANSAC，可选） =====================
def _preprocess_for_fpfh(pcd: o3d.geometry.PointCloud, voxel: float):
    p = pcd.voxel_down_sample(voxel)
    if len(p.points) == 0:
        return p, None
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=50))
    with _SilentStderr():
        feat = o3d.pipelines.registration.compute_fpfh_feature(
            p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100)
        )
    return p, feat

def fpfh_init_transform(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud) -> np.ndarray:
    """用 FPFH+RANSAC 求粗初值；失败则返回单位阵。"""
    src_p, src_f = _preprocess_for_fpfh(src, FPFH_VOXEL)
    dst_p, dst_f = _preprocess_for_fpfh(dst, FPFH_VOXEL)
    if (src_f is None) or (dst_f is None) or (len(src_p.points) == 0) or (len(dst_p.points) == 0):
        return np.eye(4)
    try:
        with _SilentStderr():
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_p, dst_p, src_f, dst_f, True, RANSAC_DIST,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DIST)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_N_ITER, 1000)
            )
        if result.transformation is None:
            return np.eye(4)
        return result.transformation
    except Exception:
        return np.eye(4)

# ===================== 多尺度 Colored-ICP（失败自动降级） =====================
def colored_icp_multiscale(src, dst, init=np.eye(4), fine_corr=0.04,
                           use_geom_fallback=True, rgbd_init=None, intrinsic=None):
    """
    多尺度 Colored-ICP：
      - 半径：0.05 -> 0.03 -> 0.015
      - 对应距离：0.12 -> 0.08 -> max(0.06, fine_corr)
      - colored 失败放宽一次；仍失败则几何 ICP 降级（用已带法线的最后一层）
    """
    T = init.copy()

    # 可选：RGB-D 里程计初值
    if rgbd_init is not None and intrinsic is not None:
        try:
            with _SilentStderr():
                ok, T_odom, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_init[0], rgbd_init[1], intrinsic, np.eye(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    o3d.pipelines.odometry.OdometryOption()
                )
            if ok:
                T = T_odom @ T
        except Exception:
            pass

    def prep(pcd, v):
        q = pcd.voxel_down_sample(v)
        if len(q.points) == 0:
            return q
        q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*3, max_nn=50))
        q.orient_normals_towards_camera_location(np.array([0,0,0], dtype=np.float64))
        return q

    scales = [0.05, 0.03, 0.015]
    dists  = [0.12, 0.08, max(0.06, float(fine_corr))]

    def _run_colored(src_lvl, dst_lvl, corr):
        with _SilentStderr():
            return o3d.pipelines.registration.registration_colored_icp(
                src_lvl, dst_lvl, corr, T,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30
                )
            )

    reg_last = None
    # 记录最后（最细）一层带法线的点云，供降级 ICP 使用
    last_src_lvl = None
    last_dst_lvl = None

    for v, corr in zip(scales, dists):
        src_lvl = prep(src, v)
        dst_lvl = prep(dst, v)
        if len(src_lvl.points) == 0 or len(dst_lvl.points) == 0:
            return None, None
        last_src_lvl, last_dst_lvl = src_lvl, dst_lvl  # 保存

        try:
            reg = _run_colored(src_lvl, dst_lvl, corr)
        except RuntimeError as e:
            if "No correspondences" in str(e):
                try:
                    reg = _run_colored(src_lvl, dst_lvl, corr*1.5)
                except RuntimeError:
                    reg = None
            else:
                raise

        if reg is None:
            continue

        T = reg.transformation
        reg_last = reg

    # colored 全部失败 -> 降级几何 ICP（必须用带法线的 last_*lvl）
    if reg_last is None or reg_last.fitness <= 1e-3:
        if not use_geom_fallback or last_src_lvl is None or last_dst_lvl is None:
            return None, None
        with _SilentStderr():
            geom = o3d.pipelines.registration.registration_icp(
                last_src_lvl, last_dst_lvl, max(0.10, dists[-1]*2), T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )
        if geom is None or geom.fitness <= 1e-3:
            return None, None
        reg_last = geom
        T = geom.transformation

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        last_src_lvl, last_dst_lvl, dists[-1], T
    )
    return reg_last, info

# ===================== 在线主流程 =====================
def main_streaming():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    color_list, depth_list = list_rgbd_paths(SEQ_DIR)
    total = min(len(color_list), len(depth_list))
    if MAX_FRAMES:
        total = min(total, MAX_FRAMES)

    # 自动深度窗（可选）
    global NEAR_CLIP, DEPTH_TRUNC
    if AUTO_DEPTH_WINDOW:
        NEAR_CLIP, DEPTH_TRUNC = estimate_depth_window(color_list, depth_list)
        print(f"[auto] NEAR_CLIP={NEAR_CLIP:.3f}  DEPTH_TRUNC={DEPTH_TRUNC:.3f}")

    # 第一帧与内参
    rgbd0, w, h, dm0 = make_rgbd(color_list[0], depth_list[0], DEPTH_TRUNC, NEAR_CLIP, DOWNSCALE)
    pos = dm0[dm0 > 0]
    print(f"[online] size={w}x{h}, depth[min,max]={float(pos.min()) if pos.size else 0:.3f}/{float(pos.max()) if pos.size else 0:.3f}")
    Kd = load_intrinsics_default(w, h)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, Kd["fx"], Kd["fy"], Kd["cx"], Kd["cy"])

    # TSDF 容器
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL,
        sdf_trunc=VOXEL * SDF_TRUNC_K,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # 在线位姿图
    pg = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.eye(4)
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans_odometry)))

    # 已处理关键帧缓存（用于回环 & 重融）
    processed_rgbds: List[o3d.geometry.RGBDImage] = [rgbd0]
    kept_idx: List[int] = [0]

    # 保存 frame0 以便快速检查
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic)
    pcd0.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    o3d.io.write_point_cloud(str(OUT_DIR/"frame0.ply"), pcd0)

    frame_period = 1.0 / max(1e-6, float(TARGET_FPS))

    def is_keyframe(k: int) -> bool:
        return (k % max(1, KEYFRAME_STEP)) == 0

    def try_add_loops():
        if not ENABLE_ONLINE_LOOPS:
            return 0
        added = 0
        if len(processed_rgbds) < 2:
            return 0
        # 每隔 LOOP_CAND_EVERY 个关键帧尝试一次
        if (len(processed_rgbds)-1) % LOOP_CAND_EVERY != 0:
            return 0

        cur_i = len(processed_rgbds)-1
        # 当前帧点云（用于配准）
        pc_cur = o3d.geometry.PointCloud.create_from_rgbd_image(processed_rgbds[cur_i], intrinsic)
        pc_cur = pc_cur.voxel_down_sample(0.012)

        for b in range(1, LOOP_MAX_BACK+1):
            j = cur_i - b*LOOP_INTERVAL
            if j <= 0:
                break
            pc_ref = o3d.geometry.PointCloud.create_from_rgbd_image(processed_rgbds[j], intrinsic)
            pc_ref = pc_ref.voxel_down_sample(0.012)
            if len(pc_ref.points) < MIN_PTS_FOR_ICP or len(pc_cur.points) < MIN_PTS_FOR_ICP:
                continue

            # 预筛重叠
            ov = overlap_ratio(pc_ref.voxel_down_sample(0.03),
                               pc_cur.voxel_down_sample(0.03),
                               np.eye(4), radius=0.08, sample_n=1500)
            if ov < 0.05 and not USE_FPFH_INIT:
                continue

            # 粗初值
            initT = fpfh_init_transform(pc_ref, pc_cur) if USE_FPFH_INIT else np.eye(4)

            reg, info = colored_icp_multiscale(
                pc_ref, pc_cur, init=initT, fine_corr=ICP_FINE_DIST,
                use_geom_fallback=True,
                rgbd_init=(processed_rgbds[j], processed_rgbds[cur_i]),
                intrinsic=intrinsic
            )
            if reg is None:
                continue
            Tij = reg.transformation
            if (_reasonable(Tij, LOOP_MAX_T, LOOP_MAX_R_DEG) and
                reg.fitness >= ICP_MIN_FIT_LP and
                reg.inlier_rmse <= ICP_MAX_RMSE_LP):
                pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    j, cur_i, Tij, info, uncertain=True))
                added += 1

        if added:
            print(f"[loop] added {added} loop edges at keyframe #{kept_idx[-1]}")
        return added

    def do_opt_and_reintegrate():
        if len(pg.edges) == 0:
            return False
        print(f"[opt] optimizing online: nodes={len(pg.nodes)} edges={len(pg.edges)}")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=ICP_FINE_DIST,
            edge_prune_threshold=0.25,
            preference_loop_closure=PREF_LOOP,
            reference_node=0
        )
        with _SilentStderr():
            o3d.pipelines.registration.global_optimization(
                pg,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option
            )
        if REINTEGRATE_AFTER_OPT:
            # 重新融合所有已处理关键帧
            tsdf_local = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=VOXEL,
                sdf_trunc=VOXEL * SDF_TRUNC_K,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            Tcw_all = [np.linalg.inv(n.pose) for n in pg.nodes]
            for rgbd_i, Tcw_i in zip(processed_rgbds, Tcw_all):
                tsdf_local.integrate(rgbd_i, intrinsic, Tcw_i)
            # 用新 TSDF 覆盖旧
            return tsdf_local
        return True

    # —— 主循环：按帧率顺序读入 —— #
    for k in range(1, total):
        t0 = time.time()
        if not is_keyframe(k):
            # 非关键帧策略：为了简洁，默认忽略（你也可以在此融合非关键帧提高密度）
            # 这里选择关键帧融合即可
            elapsed = time.time() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            continue

        # 读帧
        try:
            rgbd, w, h, _ = make_rgbd(color_list[k], depth_list[k], DEPTH_TRUNC, NEAR_CLIP, DOWNSCALE)
        except AssertionError as e:
            print(f"[online] skip frame {k}: {e}")
            elapsed = time.time() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            continue

        # 生成点云
        pc_prev = o3d.geometry.PointCloud.create_from_rgbd_image(processed_rgbds[-1], intrinsic)
        pc_prev = pc_prev.voxel_down_sample(0.012)
        pc_cur  = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pc_cur  = pc_cur.voxel_down_sample(0.012)

        if len(pc_prev.points) < MIN_PTS_FOR_ICP or len(pc_cur.points) < MIN_PTS_FOR_ICP:
            print(f"[odom] frame {k}: too few points after downsample -> drop")
            elapsed = time.time() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            continue

        # 预筛重叠率
        ov = overlap_ratio(pc_prev.voxel_down_sample(0.03),
                           pc_cur.voxel_down_sample(0.03),
                           np.eye(4), radius=0.08, sample_n=1500)

        # 初值（可选 FPFH）
        initT = np.eye(4)
        if USE_FPFH_INIT and ov < 0.15:  # 若重叠率低一点，尝试粗对齐
            initT = fpfh_init_transform(pc_prev, pc_cur)

        # 相邻配准（彩色ICP，多尺度，失败自动降级）
        reg, info = colored_icp_multiscale(
            pc_prev, pc_cur, init=initT, fine_corr=ICP_FINE_DIST,
            use_geom_fallback=True,
            rgbd_init=(processed_rgbds[-1], rgbd), intrinsic=intrinsic
        )

        if reg is None:
            print(f"[odom] frame {k} registration failed -> drop")
            # 这里选择丢帧（也可占位一个节点但不融合）
            elapsed = time.time() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            continue

        Tij = reg.transformation
        if (reg.fitness < ICP_MIN_FIT_ODO or reg.inlier_rmse > ICP_MAX_RMSE_ODO
            or not _reasonable(Tij, ODOM_MAX_T, ODOM_MAX_R_DEG)):
            print(f"[odom] frame {k} rejected: fit={reg.fitness:.3f} rmse={reg.inlier_rmse:.3f}")
            elapsed = time.time() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            continue

        # 累计里程计 & 位姿图增边
        trans_odometry = Tij @ trans_odometry
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans_odometry)))
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            len(processed_rgbds)-1, len(processed_rgbds), Tij, info, uncertain=False
        ))

        processed_rgbds.append(rgbd)
        kept_idx.append(k)

        # 在线融合当前帧（使用当前累计位姿）
        Tcw_now = np.linalg.inv(pg.nodes[-1].pose)
        tsdf.integrate(rgbd, intrinsic, Tcw_now)

        if (len(processed_rgbds)-1) % 20 == 0:
            print(f"[online] integrated {len(processed_rgbds)-1} / ~{total//max(1,KEYFRAME_STEP)} keyframes")

        # 尝试在线回环
        try_add_loops()

        # 周期性全局优化 + 可选重融
        if (len(processed_rgbds)-1) % ONLINE_OPT_EVERY_N == 0 and len(processed_rgbds) > 3:
            new_tsdf = do_opt_and_reintegrate()
            if isinstance(new_tsdf, o3d.pipelines.integration.ScalableTSDFVolume):
                tsdf = new_tsdf
                # 保存一个中间网格
                mesh_mid = tsdf.extract_triangle_mesh(); mesh_mid.compute_vertex_normals()
                o3d.io.write_triangle_mesh(str(OUT_DIR/f"mesh_mid_{len(processed_rgbds):04d}.ply"), mesh_mid)
                print(f"[online] reintegrated {len(processed_rgbds)} keyframes after optimization")

        # 节流到目标帧率
        elapsed = time.time() - t0
        if elapsed < frame_period:
            time.sleep(frame_period - elapsed)

    # —— 结束：最终一次优化 & 重融，落盘 —— #
    new_tsdf = do_opt_and_reintegrate()
    if isinstance(new_tsdf, o3d.pipelines.integration.ScalableTSDFVolume):
        tsdf = new_tsdf

    mesh = tsdf.extract_triangle_mesh(); mesh.compute_vertex_normals()
    pcd  = tsdf.extract_point_cloud()
    o3d.io.write_triangle_mesh(str(OUT_DIR/"mesh_tsdf_online_final.ply"), mesh)
    o3d.io.write_point_cloud   (str(OUT_DIR/"pcd_tsdf_online_final.ply"),  pcd)
    print(f"[done-online] keyframes={len(kept_idx)}  mesh_verts={len(mesh.vertices)}  pcd_points={len(pcd.points)}")

# ============== 入口 ==============
if __name__ == "__main__":
    main_streaming()
