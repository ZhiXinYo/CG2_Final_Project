import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Car: Orange-Red, Pedestrian: Blue, Cyclist: Green
COLOR_MAP = {
    0: (128, 128, 128), # Background (Grey)
    1: (255, 0, 0),     # Pedestrian (Blue)
    2: (0, 255, 0),     # Cyclist (Green)
    3: (0, 165, 255)    # Car (Orange)
}

class KittiCalibration:
    """ 读取标定文件，用于把 3D 点投影回 2D 图像 """
    def __init__(self, calib_file):
        self.calib = self.read_calib_file(calib_file)
        self.P2 = self.calib['P2'].reshape(3, 4)
        self.R0_rect = self.calib['R0_rect'].reshape(3, 3)
        self.Tr_velo_to_cam = self.calib['Tr_velo_to_cam'].reshape(3, 4)

        # 补齐矩阵维度到 4x4
        self.R0_rect_4x4 = np.eye(4)
        self.R0_rect_4x4[:3, :3] = self.R0_rect
        
        self.Tr_velo_to_cam_4x4 = np.eye(4)
        self.Tr_velo_to_cam_4x4[:3, :4] = self.Tr_velo_to_cam

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

def project_velo_to_image(pts_3d_velo, calib):
    '''
    将雷达坐标系的点投影到图像像素坐标系
    '''
    # 1. 转为齐次坐标
    pts_3d_velo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))
    
    # 2. 变换: Velo -> Cam -> Rect -> Image
    # P_velo_to_img = P2 * R0_rect * Tr_velo_to_cam
    P_velo_to_img = calib.P2 @ calib.R0_rect_4x4 @ calib.Tr_velo_to_cam_4x4
    
    pts_2d_hom = P_velo_to_img @ pts_3d_velo.T
    pts_2d_hom = pts_2d_hom.T
    
    # 3. 归一化 (u = x/z, v = y/z)
    mask = pts_2d_hom[:, 2] > 0 # 只保留前方的点
    pts_2d_hom = pts_2d_hom[mask]
    pts_3d_velo = pts_3d_velo[mask] # 对应的原始点也要筛选
    
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
    
    return pts_2d, pts_3d_velo, mask

def visualize(idx):
    # 1. 路径设置
    bin_file = f"data/kitti/training/velodyne_painted_stereo/{idx}.bin"
    img_file = f"data/kitti/training/image_2/{idx}.png"
    calib_file = f"data/kitti/training/calib/{idx}.txt"
    
    if not os.path.exists(bin_file):
        print(f"Error: {bin_file} not found. Try running painting script first.")
        return

    # 2. 读取数据
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 8)
    img = cv2.imread(img_file)
    calib = KittiCalibration(calib_file)
    
    # 3. 投影
    xyz = points[:, :3]
    scores = points[:, 4:] # paint scores
    
    # 投影到 2D
    uv, xyz_valid, valid_indices_mask = project_velo_to_image(xyz, calib)
    
    # 对应的 scores 也要筛选
    scores_valid = scores[valid_indices_mask]
    
    # 4. 在图像上画点
    H, W = img.shape[:2]
    
    # 为了让可视化好看，我们按距离排序，画远的再画近的，防止遮挡
    depth = xyz_valid[:, 0]
    order = np.argsort(depth)[::-1]
    
    uv = uv[order]
    scores_valid = scores_valid[order]
    
    canvas = img.copy()
    
    print(f"Drawing {len(uv)} points on image...")
    
    for i in range(len(uv)):
        u, v = int(uv[i, 0]), int(uv[i, 1])
        
        # 过滤画面外的点
        if u < 0 or u >= W or v < 0 or v >= H:
            continue
            
        # 获取该点的类别
        score = scores_valid[i]
        label = np.argmax(score)
        
        # 只画前景 (Car, Ped, Cyc)，背景(0)不画或者画很细
        if label == 0:
            continue # 跳过背景，让图更干净，像论文里那样只显示物体
            
        color = COLOR_MAP[label]
        
        # 画圆点
        cv2.circle(canvas, (u, v), 1, color, -1)

    # 5. 保存结果
    save_name = f"./visualization/vis_projected_{idx}.png"
    cv2.imwrite(save_name, canvas)
    print(f"Visualization saved to: {save_name}")

if __name__ == "__main__":
    # 依然看第 8 帧，因为这帧车多
    visualize("000000")