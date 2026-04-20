import os
import cv2
import numpy as np
import pyzed.sl as sl

# 动态链接库加载 (根据你的环境保留)
os.add_dll_directory(r"C:\Program Files (x86)\ZED SDK\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

# ====== 配置 ======
svo_path = "d:/Projects/Underwater/Datasets/ZED/HD720_SN36731600_15-11-45.svo2"
output_dir = "d:/Projects/Underwater/Datasets/ZED/extracted_svo/HD720_SN36731600_15-11-45"

# 创建细分目录
rgb_left_dir = os.path.join(output_dir, "rgb_left")
rgb_right_dir = os.path.join(output_dir, "rgb_right")
depth_dir = os.path.join(output_dir, "depth_16bit")
os.makedirs(rgb_left_dir, exist_ok=True)
os.makedirs(rgb_right_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

gt_file = os.path.join(output_dir, "groundtruth.txt")

# ====== 初始化 ======
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.set_from_svo_file(svo_path)
init_params.svo_real_time_mode = False
init_params.coordinate_units = sl.UNIT.METER
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
# 建议加上深度模式，以防后续又需要测试ZED原厂深度
init_params.depth_mode = sl.DEPTH_MODE.ULTRA 

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"打开SVO失败: {status}")
    exit(1)

# ====== 核心修复：开启 Positional Tracking ======
tracking_params = sl.PositionalTrackingParameters()
tracking_status = zed.enable_positional_tracking(tracking_params)
if tracking_status != sl.ERROR_CODE.SUCCESS:
    print(f"开启位置追踪失败: {tracking_status}")
    exit(1)

runtime = sl.RuntimeParameters()
image_left = sl.Mat()
image_right = sl.Mat()
depth = sl.Mat()
pose = sl.Pose()

# ====== 提取循环 ======
frame_id = 0

with open(gt_file, "w") as f:
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            # --- 双目 RGB 提取 ---
            # zed.retrieve_image(image_left, sl.VIEW.LEFT)
            # zed.retrieve_image(image_right, sl.VIEW.RIGHT)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)
            
            rgb_l = cv2.cvtColor(image_left.get_data(), cv2.COLOR_RGBA2RGB)
            rgb_r = cv2.cvtColor(image_right.get_data(), cv2.COLOR_RGBA2RGB)
            
            cv2.imwrite(f"{rgb_left_dir}/{frame_id:06d}.png", rgb_l)
            cv2.imwrite(f"{rgb_right_dir}/{frame_id:06d}.png", rgb_r)

            # --- Depth 提取 (带 NaN 处理) ---
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_map = depth.get_data()
            
            # 将无效值(NaN/Inf)替换为 0，防止转换为 uint16 时溢出或报错
            depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = (depth_map * 1000).astype(np.uint16)
            cv2.imwrite(f"{depth_dir}/{frame_id:06d}.png", depth_mm)

            # --- Pose (Ground Truth) ---
            tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            
            # 只有当追踪状态为 OK 时，才记录有效位姿
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                t = pose.get_translation().get()
                q = pose.get_orientation().get()
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()
                
                # 格式: timestamp tx ty tz qx qy qz qw
                f.write(f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

            frame_id += 1
            if frame_id % 100 == 0:
                print(f"已处理 {frame_id} 帧...")
        else:
            break

# 收尾
zed.disable_positional_tracking()
zed.close()
print("🎉 导出全部完成！")