import torch
import numpy as np
import os
from tqdm import tqdm
from mmseg.apis import inference_model, init_model
import mmcv

# ================= 配置区域 =================
DEVICE = "cuda:7"
DATA_ROOT = "data/kitti/training/"  # data/nuscenes/samples/
SEG_CONFIG = 'configs_seg/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
SEG_CHECKPOINT = 'checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'

# 是否启用双目 Painting (如果设为 False，就只用 image_2)
USE_STEREO = True 
SAVE_PATH = os.path.join(DATA_ROOT, "velodyne_painted_stereo" if USE_STEREO else "velodyne_painted_mono")
# ===========================================

class KittiCalibration:
    def __init__(self, calib_file):
        self.calib = self.read_calib_file(calib_file)
        
        # 解析 P2 (左相机) 和 P3 (右相机)
        self.P2 = np.zeros((4, 4))
        self.P2[:3, :4] = self.calib['P2'].reshape(3, 4)
        self.P2[3, 3] = 1.0

        self.P3 = np.zeros((4, 4))
        self.P3[:3, :4] = self.calib['P3'].reshape(3, 4)
        self.P3[3, 3] = 1.0

        # R0_rect
        self.R0_rect = np.zeros((4, 4))
        self.R0_rect[:3, :3] = self.calib['R0_rect'].reshape(3, 3)
        self.R0_rect[3, 3] = 1.0

        # Tr_velo_to_cam
        self.Tr_velo_to_cam = np.zeros((4, 4))
        self.Tr_velo_to_cam[:3, :4] = self.calib['Tr_velo_to_cam'].reshape(3, 4)
        self.Tr_velo_to_cam[3, 3] = 1.0

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

class Painter:
    def __init__(self):
        self.root_split_path = DATA_ROOT
        self.save_path = SAVE_PATH
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        print(f'Loading Segmentation Model...')
        self.model = init_model(SEG_CONFIG, SEG_CHECKPOINT, device=DEVICE)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne', f'{idx}.bin')
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, camera_folder):
        """
        camera_folder: 'image_2' or 'image_3'
        """
        img_path = os.path.join(self.root_split_path, camera_folder, f'{idx}.png')
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found!")
            return None

        result = inference_model(self.model, img_path)
        
        # 兼容性处理
        if hasattr(result, 'pred_sem_seg'):
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
        else:
            pred_mask = result[0]

        H, W = pred_mask.shape
        # 定义 4 类：[Background, Pedestrian, Cyclist, Car]
        output_score = np.zeros((H, W, 4), dtype=np.float32)
        
        # 默认背景分 1.0
        output_score[:, :, 0] = 1.0 
        
        # Car (Cityscapes: 13-16)
        car_mask = np.isin(pred_mask, [13, 14, 15, 16])
        output_score[car_mask, 0] = 0
        output_score[car_mask, 3] = 1.0
        
        # Pedestrian (Cityscapes: 11, 12)
        ped_mask = np.isin(pred_mask, [11, 12])
        output_score[ped_mask, 0] = 0
        output_score[ped_mask, 1] = 1.0
        
        # Cyclist (Cityscapes: 17, 18)
        cyc_mask = np.isin(pred_mask, [17, 18])
        output_score[cyc_mask, 0] = 0
        output_score[cyc_mask, 2] = 1.0
        
        return output_score

    def get_calib(self, idx):
        calib_file = os.path.join(self.root_split_path, 'calib', f'{idx}.txt')
        return KittiCalibration(calib_file)

    def project_lidar(self, points, calib, projection_matrix, img_h, img_w):
        """
        通用投影函数
        projection_matrix: P2 or P3
        """
        pts_3d_velo = points.copy()
        pts_3d_velo[:, 3] = 1.0
        
        # P_velo_to_img = P_cam * R0_rect * Tr_velo_to_cam
        P_velo_to_img = projection_matrix @ calib.R0_rect @ calib.Tr_velo_to_cam
        
        pts_2d_hom = P_velo_to_img @ pts_3d_velo.T 
        pts_2d_hom = pts_2d_hom.T 
        
        # 深度掩码
        depth_mask = pts_2d_hom[:, 2] > 0
        
        # 归一化
        pts_2d = np.zeros((points.shape[0], 2))
        pts_2d[depth_mask] = pts_2d_hom[depth_mask, :2] / pts_2d_hom[depth_mask, 2:3]
        
        u = np.round(pts_2d[:, 0]).astype(int)
        v = np.round(pts_2d[:, 1]).astype(int)
        
        # 图像范围掩码
        img_mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & depth_mask
        
        return u, v, img_mask

    def run(self):
        image_dir = os.path.join(self.root_split_path, "image_2")
        files = sorted(os.listdir(image_dir))
        file_ids = [f.split('.')[0] for f in files if f.endswith('.png')]

        print(f"Start painting {len(file_ids)} frames. Mode: {'Stereo' if USE_STEREO else 'Mono'}")
        
        for idx in tqdm(file_ids):
            points = self.get_lidar(idx) # (N, 4)
            calib = self.get_calib(idx)
            
            # 初始化 Paint Features (N, 4)，全0
            paint_features = np.zeros((points.shape[0], 4), dtype=np.float32)
            
            # ------ 处理左目 (Image 2) ------
            scores_l = self.get_score(idx, "image_2")
            if scores_l is not None:
                h_l, w_l, c = scores_l.shape
                u_l, v_l, mask_l = self.project_lidar(points, calib, calib.P2, h_l, w_l)
                
                # 累加分数 (Sampling)
                # 只有投影在图像内的点才有分
                paint_features[mask_l] += scores_l[v_l[mask_l], u_l[mask_l]]
            
            # ------ 处理右目 (Image 3) ------
            mask_r = np.zeros(points.shape[0], dtype=bool)
            if USE_STEREO:
                scores_r = self.get_score(idx, "image_3")
                if scores_r is not None:
                    h_r, w_r, c = scores_r.shape
                    u_r, v_r, mask_r = self.project_lidar(points, calib, calib.P3, h_r, w_r)
                    
                    # 累加分数
                    paint_features[mask_r] += scores_r[v_r[mask_r], u_r[mask_r]]
            
            # 融合
            overlap_mask = mask_l & mask_r
            if USE_STEREO:
                # 重叠区域取平均
                paint_features[overlap_mask] *= 0.5
            
            all_outside_mask = ~(mask_l | mask_r)
            paint_features[all_outside_mask, 0] = 1.0 # 强制设为背景
            
            painted_points = np.hstack([points, paint_features])
            save_file = os.path.join(self.save_path, f"{idx}.bin")
            painted_points.astype(np.float32).tofile(save_file)

if __name__ == '__main__':
    painter = Painter()
    painter.run()