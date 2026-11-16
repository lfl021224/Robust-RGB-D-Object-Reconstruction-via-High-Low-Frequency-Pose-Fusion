# RGB-D High-Low Frequency Pose Fusion

面向弱纹理、平面退化及高反光物体的 RGB-D 三维重建研究代码。通过帧间 ICP 估计局部运动，在 SE(3) 上融合外部位姿先验，并结合深度预处理与 TSDF 积分生成三维结果。

对应论文：**Robust RGB-D Object Reconstruction in Degenerate Scenes via High-Low Frequency Pose Fusion for Industrial Inspection**。

[仓库内论文 PDF](Essay/Robust_RGB_D_Object_Reconstruction_in_Degenerate_Scenes_via_High_Low_Frequency_Pose_Fusion.pdf)

本仓库保留不同场景的研究脚本，尚未统一为命令行软件包。根目录 `main.py` 是 KITTI 双目 SGBM 示例，不是论文方法入口。后续 dissertation 工程中的 MARINE、通用位姿图和网格后处理不属于这里的原始论文流程。

## 方法

```text
RGB + Depth → 时间关联、内参和单位核对
            → 自适应深度范围与双边滤波
            → ORB-PnP 初始化与多尺度帧间 ICP
外部位姿先验 → SE(3) 衰减、限幅融合
            → 指定位姿源的 TSDF 积分
            → 点云导出、可视化与轨迹评估
```

- **深度预处理**：根据深度分位数估计截断范围，并通过时间平滑降低范围抖动；双边滤波处理局部噪声。ROI、补洞、背景平面移除与遮挡处理按配置启用。
- **局部运动**：主要入口尝试 ORB-PnP 初值与多尺度 Colored ICP，失败时回退至鲁棒 point-to-plane ICP；按配准内点率、残差和运动幅度筛选更新。当前主循环是帧间配准，不是完整的局部地图 SLAM。
- **先验融合**：计算 `xi = Log(T_prior @ inverse(T_est))`，再执行 `T_fused = Exp(clamp(alpha * xi)) @ T_est`。
- **建模**：Open3D TSDF 积分、实时预览与点云导出，后端取决于本机 Open3D/CUDA 能力。

“高低频”表示局部运动更新与先验校正的分工，并非显式频谱分解。融合效果依赖先验质量；衰减和限幅不能保证消除系统性先验漂移。本仓库不是已验证可在无可靠先验条件下完成任意场景闭环的 SLAM 系统。

## 目录与入口

| 路径 | 用途 |
| --- | --- |
| `High-Low Frequency Pose Fusion/GT+ICP_external_prior_fusion_tsdf_live.py` | 主要 TUM RGB-D 融合重建入口 |
| `High-Low Frequency Pose Fusion/gt_icp_tsdf_tum_sphere_fix.py` | 球体场景实验变体 |
| `High-Low Frequency Pose Fusion/gt_icp_tsdf_tum_flowerbouquet_fix_v20.py` | 花束/薄结构实验变体；实际输入仍由配置决定 |
| `High-Low Frequency Pose Fusion/redwood_batch_tsdf.py` | Redwood 批量 TSDF 脚本 |
| `High-Low Frequency Pose Fusion/TSDF_live_demo.py` | TSDF 演示 |
| `High-Low Frequency Pose Fusion/Results for Essay/` | 本地论文图件、结果及代码快照 |
| `Essay/` | 当前、旧版及压缩版论文 |
| `extract_svo.py` | ZED SVO/SVO2 导出实验工具 |
| `main.py` | KITTI 双目 SGBM 与点云示例 |
| `DepthSwinFPN_main.py` | 独立监督深度训练实验，非融合方法必需组件 |
| `RAFT-Stereo/` | 双目深度相关第三方代码 |
| `view_outputs.py`, `unzipbag.py` | 查看结果及数据辅助工具 |
| `Datasets/`, `depth_checkpoints/`, `Code Backup/` | 本地数据、权重与历史备份，被 Git 忽略 |

## 环境

主要融合脚本依赖 Python、NumPy、OpenCV、Open3D 和 Matplotlib。`requirements.txt` 是包含 Windows/ZED 依赖的环境快照，不是跨平台最小依赖清单。

在独立 Python 环境中安装核心依赖：

```powershell
python -m pip install numpy==2.2.6 opencv-python==4.10.0.84 open3d==0.19.0 matplotlib==3.10.6
```

版本来自仓库记录；本次 README 整理未重新验证环境安装或运行性能。

- 普通 OpenCV/Open3D wheel 不保证 CUDA 支持。代码包含探测与回退，安装成功不代表启用 GPU 加速。
- ZED 导出另需匹配的 ZED SDK、Python API、CUDA 及本机 DLL 路径；TUM 融合不需要 ZED SDK。
- 深度训练和 RAFT 另需 PyTorch 等依赖及权重，不包含在上述核心安装命令中。
- 当前脚本含 OpenCV 窗口和交互 ROI，按桌面程序运行；关闭 Open3D 实时预览不等于完整支持无显示服务器。

## 输入数据

主要入口采用 TUM 风格目录：

```text
sequence/
├─ rgb/                 # 时间戳命名的 RGB 图像
├─ depth/               # 对齐深度
├─ associations.txt     # 可选，推荐显式关联
└─ groundtruth.txt      # 可选外部位姿流
```

关联文件每行：

```text
t_rgb rgb/t_rgb.png t_depth depth/t_depth.png
```

外部位姿每行：

```text
timestamp tx ty tz qx qy qz qw
```

无关联文件时按文件名时间戳最近邻配对，容差 0.050 秒。位姿查询也依赖图像文件名时间戳，不能将连续帧编号直接当作时间。

主要入口将 `uint16` 深度除以 **5000** 转为米，这是 TUM 约定。毫米深度应除以 **1000**，不能直接套用。内参按路径中的 Freiburg 标识选择；其他相机需配置实际内参，不能依赖默认回退值。

`groundtruth.txt` 是接口名称：TUM 中可以是真值，ZED 数据中则通常只是 SDK 估计轨迹，不能混淆二者的准确性与评估用途。

## 运行

### 1. 修改顶部配置

打开 `High-Low Frequency Pose Fusion/GT+ICP_external_prior_fusion_tsdf_live.py`，至少设置：

```python
SEQ_DIR = Path(r"D:\datasets\rgbd_dataset_freiburg2_metallic_sphere2")
OUT_DIR = Path(r"D:\outputs\rgbd_sphere_fused")
TSDF_POSE_SOURCE = "fused"
```

同时核对 ROI、深度范围、内参、`VOXEL_LEN`、时间关联和融合参数。代码仍含旧机器绝对路径，脚本文件名不保证对应当前配置的数据集。

### 2. 选择积分位姿

| `TSDF_POSE_SOURCE` | 行为 |
| --- | --- |
| `"gt"` | 直接用对齐后的外部位姿积分；查询不到先验则跳过该帧积分 |
| `"icp"` / `"odo"` | 用累计局部里程计轨迹积分 |
| `"fused"` | 用先验校正后的融合轨迹积分；当前实现由其他字符串的分支接收 |

**当前主要脚本默认是 `"gt"`，不是 `"fused"`。** 评估融合重建时需显式修改；直接先验建模不能标记为融合结果。无先验时局部跟踪可能运行，但没有先验纠漂能力，也不应选择要求先验的 `"gt"` 积分模式。

### 3. 启动

在仓库根目录运行：

```powershell
python ".\High-Low Frequency Pose Fusion\GT+ICP_external_prior_fusion_tsdf_live.py"
```

入口使用脚本内配置，没有统一参数化 CLI。默认进行一次交互 ROI 选择；不使用时修改 `USE_INTERACTIVE_ROI_ONCE` 及跟踪/积分 ROI 配置。

### 4. 输出

主要入口在 `OUT_DIR` 下保存 `pcd_final_*.ply` 点云，按配置输出 `snapshots/`。有可用参考轨迹时输出 ATE 信息与 `ate_error_plot_*.png`，异常写入 `live_crash.log`。最终导出是点云，不能仅凭内部使用 TSDF 就称为已导出的三角网格。

## 论文实验与复现边界

以下为文稿报告值，本次文档整理未重新运行：

| 序列 | ICP ATE RMSE | 融合 ATE RMSE |
| --- | ---: | ---: |
| `freiburg2_metallic_sphere2` | 2.9777 m | 0.1267 m |
| `freiburg3_cabinet` | 0.8300 m | 0.0126 m |

花束是薄结构挑战展示，不在该定量表中。

当前工作脚本与论文设置有差异：

| 项目 | 论文 | 主要脚本当前配置/实现 |
| --- | --- | --- |
| 融合增益 | 0.10 | 0.25 |
| 单步旋转校正上限 | 1° | 2° |
| 单步平移校正上限 | 0.01 m | 0.03 m |
| TSDF 位姿源 | 融合位姿 | 默认 `"gt"` |
| ATE 对齐 | 文稿为 Sim(3) | `compute_ate_rmse` 只估计旋转和平移，不估计尺度 |

当前默认运行不是论文逐项复现配置。应记录脚本版本、先验来源、ROI、深度处理、时间关联、接纳帧集、积分位姿源和 ATE 对齐方式；比较 ICP 与融合时保持数据与深度处理一致。

若同一参考轨迹既用于融合又用于 ATE 评估，指标反映的是使用该先验条件下的轨迹接近程度，不能证明无先验里程计的独立精度。论文运行时间为指定平台报告值，不是所有环境的性能保证。

## ZED 扩展边界

`extract_svo.py` 导出 `rgb_left/`、`rgb_right/`、`depth_16bit/` 及名为 `groundtruth.txt` 的 SDK 位姿，与上述 TUM 入口不是直接兼容的链路：

- SDK 深度为毫米 PNG，TUM 默认按 1/5000 米解码。
- 图像按帧编号命名，位姿只在跟踪 OK 时写入，图像和位姿不能按行号直接对应。
- 提取使用 `RIGHT_HANDED_Z_UP`，需核对相机光学坐标与世界坐标变换。
- 水下深度偏差和 SDK 轨迹漂移不能由衰减融合自动消除，需在后续工程独立验证。

## 数据与许可

数据、权重及部分本地输出被 `.gitignore` 排除，克隆代码不代表获得全部实验材料。RAFT-Stereo 等第三方代码遵循各自许可；本 README 不新增或授予开源许可。
