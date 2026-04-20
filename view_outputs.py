# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import open3d as o3d

# ===== 输出目录 =====
OUT_DIR   = Path(r"D:\Projects\Underwater\Outputs\TUM_RGBD\rgbd_dataset_freiburg2_metallic_sphere2")
MESH_PATH = OUT_DIR / "mesh_tsdf.ply"
PCD_PATH  = OUT_DIR / "pcd_final_20260331_170742.ply"
F0_PATH   = OUT_DIR / "frame0.ply"
# ====================

MAX_FACES_TO_VIEW  = 800_000
MAX_POINTS_TO_VIEW = 2_000_000
SNAP_W, SNAP_H     = 1600, 1200

def print_stats(geom):
    if isinstance(geom, o3d.geometry.TriangleMesh):
        bb = geom.get_axis_aligned_bounding_box()
        print(f"[MESH] verts={len(geom.vertices)}, faces={len(geom.triangles)}, extent(m)={bb.get_extent()}")
    elif isinstance(geom, o3d.geometry.PointCloud):
        bb = geom.get_axis_aligned_bounding_box()
        print(f"[PCD ] points={len(geom.points)}, extent(m)={bb.get_extent()}")

def simplify_for_view(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    f = len(mesh.triangles)
    if f > MAX_FACES_TO_VIEW:
        target = MAX_FACES_TO_VIEW
        mesh = mesh.simplify_quadric_decimation(target)
    mesh.compute_vertex_normals()
    return mesh

def downsample_for_view(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    n = len(pcd.points)
    if n > MAX_POINTS_TO_VIEW:
        extent = pcd.get_axis_aligned_bounding_box().get_extent()
        s = max(extent) / 200.0
        pcd = pcd.voxel_down_sample(max(0.005, float(s)))
    return pcd

def try_gui_view(geom, title):
    # 若是 headless 构建，这里会抛异常
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    o3d.visualization.draw_geometries([geom, axes], window_name=title)

def snapshot_matplotlib_from_points(points, colors, out_png: Path, elev=20, azim=45):
    import matplotlib
    matplotlib.use("Agg")  # 纯离屏
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    x, y, z = points[:,0], points[:,1], points[:,2]
    # 尺度归一：保持等比例
    m = np.column_stack([x, y, z]).mean(axis=0)
    s = np.max(np.ptp(np.column_stack([x, y, z]), axis=0)) * 0.55
    fig = plt.figure(figsize=(SNAP_W/100, SNAP_H/100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(x, y, z, s=0.1, c=colors, linewidths=0, alpha=1.0)
    ax.set_xlim(m[0]-s, m[0]+s); ax.set_ylim(m[1]-s, m[1]+s); ax.set_zlim(m[2]-s, m[2]+s)
    ax.set_axis_off()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("[SNAPSHOT] ->", out_png)

def headless_snapshot_mesh(mesh: o3d.geometry.TriangleMesh, out_png: Path):
    # 将网格均匀采样为点（带颜色），再用 matplotlib 画
    n_target = min(800_000, max(200_000, len(mesh.vertices)))
    pts = mesh.sample_points_uniformly(n_target)
    P = np.asarray(pts.points)
    if pts.has_colors():
        C = np.asarray(pts.colors)
    elif mesh.has_vertex_colors():
        # 直接用顶点色
        C = np.asarray(mesh.vertex_colors)
        if len(C) != len(P):  # 尺寸不匹配时用灰色
            C = np.tile(np.array([[0.5,0.5,0.5]]), (len(P),1))
    else:
        C = np.tile(np.array([[0.6,0.6,0.6]]), (len(P),1))
    snapshot_matplotlib_from_points(P, C, out_png)

def headless_snapshot_pcd(pcd: o3d.geometry.PointCloud, out_png: Path):
    pcd = downsample_for_view(pcd)
    P = np.asarray(pcd.points)
    if pcd.has_colors():
        C = np.asarray(pcd.colors)
    else:
        C = np.tile(np.array([[0.4,0.4,0.9]]), (len(P),1))
    snapshot_matplotlib_from_points(P, C, out_png)

def export_variants(mesh: o3d.geometry.TriangleMesh, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()  # 写 STL 需要三角法线
    # 简化 OBJ
    target = min(MAX_FACES_TO_VIEW, max(50_000, len(mesh.vertices)//2))
    mesh_dec = mesh.simplify_quadric_decimation(target)
    mesh_dec.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_dir / "mesh_tsdf_dec.obj"), mesh_dec)
    # 原始 STL（或可写 mesh_tsdf_dec.stl）
    o3d.io.write_triangle_mesh(str(out_dir / "mesh_tsdf.stl"), mesh)
    print("[EXPORT] ->", out_dir / "mesh_tsdf_dec.obj")
    print("[EXPORT] ->", out_dir / "mesh_tsdf.stl")

def main():
    if MESH_PATH.exists():
        mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))
        print_stats(mesh)

        # 色彩夹到[0,1]
        if mesh.has_vertex_colors():
            cols = np.asarray(mesh.vertex_colors)
            cols = np.clip(cols, 0.0, 1.0)
            mesh.vertex_colors = o3d.utility.Vector3dVector(cols)

        export_variants(mesh, OUT_DIR)

        # 尝试 GUI；失败则用 matplotlib 快照
        try:
            try_gui_view(simplify_for_view(mesh), "TSDF Mesh (view)",)
        except Exception as e:
            print(f"[VIEW-FALLBACK] {e}")
            headless_snapshot_mesh(mesh, OUT_DIR / "mesh_view.png")
    else:
        print(f"[WARN] 未找到 {MESH_PATH}")

    if PCD_PATH.exists():
        pcd = o3d.io.read_point_cloud(str(PCD_PATH))
        print_stats(pcd)
        try:
            try_gui_view(downsample_for_view(pcd), "TSDF Point Cloud (view)")
        except Exception as e:
            print(f"[VIEW-FALLBACK] {e}")
            headless_snapshot_pcd(pcd, OUT_DIR / "pcd_view.png")
    else:
        print(f"[WARN] 未找到 {PCD_PATH}")

    if F0_PATH.exists():
        pcd0 = o3d.io.read_point_cloud(str(F0_PATH))
        print_stats(pcd0)
        try:
            try_gui_view(downsample_for_view(pcd0), "Frame0 Point Cloud")
        except Exception as e:
            print(f"[VIEW-FALLBACK] {e}")
            headless_snapshot_pcd(pcd0, OUT_DIR / "frame0_view.png")
    else:
        print(f"[WARN] 未找到 {F0_PATH}")

if __name__ == "__main__":
    main()
