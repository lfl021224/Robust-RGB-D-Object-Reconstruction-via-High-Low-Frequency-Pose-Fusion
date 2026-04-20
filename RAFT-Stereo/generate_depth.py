import os
import glob
import numpy as np
import cv2

# ====== 配置输入输出路径 ======
# 指向你刚刚 RAFT-Stereo 输出的目录
input_dir = "d:/Projects/Underwater/RAFT-Stereo/zed_depth_output"

# 创建两个输出文件夹
output_dir_16bit = "d:/Projects/Underwater/RAFT-Stereo/zed_depth_maps_16bit"
output_dir_color = "d:/Projects/Underwater/RAFT-Stereo/zed_depth_maps_color"
os.makedirs(output_dir_16bit, exist_ok=True)
os.makedirs(output_dir_color, exist_ok=True)

# ====== 水下专属相机参数 ======
# 直接使用你截图中的水下参数
f_x = 955.7350        # 水下等效焦距 (像素)
baseline = 0.11996    # 基线长度 TX (米)

# ====== 开始批量转换 ======
npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
print(f"Find the parallax matrix file {len(npy_files)} and start converting it into a physical depth map...")

for npy_path in npy_files:
    filename = os.path.basename(npy_path).replace(".npy", "")
    
    # 1. 加载视差矩阵 (格式: float32)
    disparity = np.load(npy_path)
    
    # 2. 安全处理：避免除以 0 或无效的负视差
    # 设定极小值阈值，滤除背景噪点
    valid_mask = disparity > 0.1 
    
    # 初始化深度矩阵 (全 0 代表无效区域)
    depth_m = np.zeros_like(disparity, dtype=np.float32)
    
    # 3. 核心计算：转换为真实物理深度 (单位：米)
    depth_m[valid_mask] = (f_x * baseline) / disparity[valid_mask]
    
    # 4. 截断异常值：限制水下深度最大有效范围 (比如限制在 15 米以内)
    # 防止因为极小视差导致算出几百米的飞点
    max_depth = 15.0
    depth_m = np.clip(depth_m, 0, max_depth)
    
    # ==========================================
    # 导出格式一：标准 16-bit 深度图 (用于 3D 建模)
    # 约定俗成：将米乘以 1000 转换为毫米，存入 uint16
    # ==========================================
    depth_mm = (depth_m * 1000).astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir_16bit, f"{filename}.png"), depth_mm)
    
    # ==========================================
    # 导出格式二：伪彩色深度图 (仅用于肉眼检查)
    # ==========================================
    # 设定你水下物体的常见距离范围，用于拉伸色彩对比度 (例如 0.5米 到 5米)
    min_d, max_d = 0.5, 5.0 
    depth_norm = np.clip((depth_m - min_d) / (max_d - min_d), 0, 1)
    
    # 将 0-1 映射为 0-255 的 8-bit 图像
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # 应用 COLORMAP_JET 伪彩色 (暖色=近，冷色=远)
    # 255 - depth_uint8 是为了让近处发红，远处变蓝
    depth_color = cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_JET)
    
    # 将没有深度的无效区域(原本为0的部分)涂黑
    depth_color[depth_m == 0] = [0, 0, 0]
    
    cv2.imwrite(os.path.join(output_dir_color, f"{filename}_color.png"), depth_color)

print("Finish!")
print(f"建模用 16-bit PNG 已保存至: {output_dir_16bit}")
print(f"查看用 彩色 PNG 已保存至: {output_dir_color}")