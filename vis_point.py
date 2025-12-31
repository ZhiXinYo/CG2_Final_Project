# import numpy as np
# import matplotlib.pyplot as plt

# def generate_sads_comparison_v2(bin_path, out_file='sads_comparison_final.png'):
#     # 1. 加载 8 维点云 [x, y, z, r, s_bg, s_ped, s_cyc, s_car]
#     points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 8)
    
#     # 2. 采样参数 (设狠一点，为了PPT对比效果)
#     dist_thr = 30.0
#     keep_ratio_bg = 0.05 # 只留 5% 的近处背景，对比更强烈
    
#     dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
#     # 【核心修正】：使用 argmax 判定。索引 4 是背景，5,6,7 是前景
#     # 只有当 5,6,7 里的最大值比 4 大，才叫前景
#     scores = points[:, 4:] # [s0, s1, s2, s3]
#     is_foreground = np.argmax(scores, axis=1) > 0 
    
#     # 逻辑：(远处) 或 (是前景) -> 100% 保护
#     is_protected = (dist >= dist_thr) | is_foreground
#     can_sample = ~is_protected
    
#     # 3. 执行采样
#     keep_indices_prot = np.where(is_protected)[0]
#     sample_candidates = np.where(can_sample)[0]
    
#     num_keep = int(len(sample_candidates) * keep_ratio_bg)
#     rand_idx = np.random.choice(sample_candidates, num_keep, replace=False)
    
#     final_indices = np.concatenate([keep_indices_prot, rand_idx])
#     points_sads = points[final_indices]

#     # 4. 绘图
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
    
#     # 左图：原始 (颜色代表高度，显示密集感)
#     ax1.scatter(points[:, 1], points[:, 0], s=0.2, c=points[:, 2], cmap='Blues_r', alpha=0.8)
#     ax1.set_title("Original LiDAR (Redundant)", color='white', fontsize=20, pad=20)
    
#     # 右图：SADS (前景红色，背景灰色)
#     # 重新计算保留点的颜色
#     sads_fg_mask = is_foreground[final_indices]
#     c_list = np.where(sads_fg_mask, '#FF0000', '#444444') # 前景亮红，背景暗灰
    
#     ax2.scatter(points_sads[:, 1], points_sads[:, 0], s=0.2, c=c_list, alpha=0.9)
#     ax2.set_title("SADS Optimized", color='white', fontsize=20, pad=20)

#     # 统一设置
#     for ax in [ax1, ax2]:
#         ax.set_xlim([-30, 30]) # 缩窄范围看细节
#         ax.set_ylim([0, 60])
#         ax.axis('off')

#     plt.tight_layout()
#     plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='black')
#     print(f"✅ 修正后的对比图已保存: {out_file}")

# # 运行
# generate_sads_comparison_v2('data/kitti_painted/training/velodyne_painted_1225/000134.bin')


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def get_projection_matrix(calib_path):
    """读取KITTI标定矩阵"""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
    R0 = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
    V2C = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)
    R0_homo = np.eye(4); R0_homo[:3, :3] = R0
    V2C_homo = np.eye(4); V2C_homo[:3, :4] = V2C
    return P2 @ R0_homo @ V2C_homo

def plot_projected_points(ax, img, points, is_foreground, title):
    """将点云投影并绘制在图像上"""
    ax.imshow(img)
    # 颜色：前景亮红，背景淡蓝
    colors = np.where(is_foreground, '#FF0000', '#0000ff')
    sizes = np.where(is_foreground, 2.0, 2.0) # 前景点稍微大一点
    
    ax.scatter(points[:, 0], points[:, 1], s=sizes, c=colors, edgecolors='none', alpha=0.6)
    ax.set_title(title, color='white', fontsize=18)
    ax.axis('off')

def generate_sads_2d_projection(bin_path, img_path, calib_path, out_file='sads_2d_comparison.png'):
    # 1. 加载数据
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 8)
    img = Image.open(img_path)
    W, H = img.size
    lidar2img = get_projection_matrix(calib_path)

    # 2. SADS 逻辑
    dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    scores = points[:, 4:]
    is_fg = np.argmax(scores, axis=1) > 0
    
    is_protected = (dist >= 30.0) | is_fg
    can_sample = ~is_protected

    # 执行采样（设为 0.15 比例以获得最强视觉对比）
    num_keep = int(np.sum(can_sample) * 0.25)
    sample_indices = np.where(can_sample)[0]
    rand_idx = np.random.choice(sample_indices, num_keep, replace=False)
    final_indices = np.concatenate([np.where(is_protected)[0], rand_idx])
    
    # 3. 投影坐标计算
    def project(pts):
        pts_homo = np.column_stack([pts[:, :3], np.ones(len(pts))])
        img_pts_homo = (lidar2img @ pts_homo.T).T
        # 过滤掉相机背后的点
        mask = img_pts_homo[:, 2] > 0
        img_pts = img_pts_homo[mask, :2] / img_pts_homo[mask, 2:3]
        # 过滤掉图像范围外的点
        in_fov = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & \
                 (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
        return img_pts[in_fov], mask, in_fov

    # 计算原始点云投影
    img_pts_orig, m1, m2 = project(points)
    fg_orig = is_fg[m1][m2]

    # 计算SADS后点云投影
    img_pts_sads, m3, m4 = project(points[final_indices])
    fg_sads = is_fg[final_indices][m3][m4]

    # 4. 分别绘图并保存（无边框、无标题、无背景）

    # 原始点云投影
    fig1, ax1 = plt.subplots(figsize=(img.size[0] / 100, img.size[1] / 100))
    ax1.imshow(img)
    plot_projected_points(
        ax1, img, img_pts_orig, fg_orig, title=""
    )
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("original_projection.png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

    # SADS 后点云投影
    fig2, ax2 = plt.subplots(figsize=(img.size[0] / 100, img.size[1] / 100))
    ax2.imshow(img)
    plot_projected_points(
        ax2, img, img_pts_sads, fg_sads, title=""
    )
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("sads_projection.png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

    print("✅ 已保存无边框、无标题、像素对齐的投影图")

# 路径设置示例

generate_sads_2d_projection('data/kitti_painted/training/velodyne_painted_1225/000134.bin', 
                            'data/kitti_painted/training/image_2/000134.png',
                            'data/kitti_painted/training/calib/000134.txt')