import os
import numpy as np
import torch
from tqdm import tqdm
import mmcv
from PIL import Image

# MMDetection3D / MMSegmentation 1.x APIs
from mmseg.apis import init_model, inference_model

# -------------------------------------------------------------------------
# KITTI Calibration (完全复刻原仓库 calibration_kitti.py 逻辑)
# -------------------------------------------------------------------------
def get_calib_from_file(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    obj = {}
    for line in lines:
        if line == '\n': continue
        key, value = line.split(':', 1)
        obj[key] = np.array([float(x) for x in value.split()])
    
    calib = {}
    calib['P2'] = np.concatenate([obj['P2'].reshape(3, 4), np.array([[0., 0., 0., 1.]])], axis=0)
    calib['P3'] = np.concatenate([obj['P3'].reshape(3, 4), np.array([[0., 0., 0., 1.]])], axis=0)
    calib['R0_rect'] = np.zeros([4, 4], dtype=obj['R0_rect'].dtype)
    calib['R0_rect'][3, 3] = 1.
    calib['R0_rect'][:3, :3] = obj['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = np.concatenate([obj['Tr_velo_to_cam'].reshape(3, 4), np.array([[0., 0., 0., 1.]])], axis=0)
    return calib

# -------------------------------------------------------------------------
# Painter 类 (1:1 复刻原仓库逻辑)
# -------------------------------------------------------------------------
class Painter:
    def __init__(self, config, checkpoint, device='cuda:3'):
        self.root_path = 'data/kitti_painted/training/'
        self.save_path = 'data/kitti_painted/training/velodyne_painted_1225/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # 初始化分割模型
        print(f'Initializing Segmentation Network...')
        self.model = init_model(config, checkpoint, device=device)
        
        # 强制开启 logits 输出 (适配 mmseg 1.x)
        if hasattr(self.model, 'cfg'):
            self.model.cfg.model.test_cfg.output_logits = True

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_path, 'velodyne', f'{idx}.bin')
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, camera_folder):
        filename = os.path.join(self.root_path, camera_folder, f'{idx}.png')
        result = inference_model(self.model, filename)
        
        # 获取 logits 并应用 Softmax
        logits = result.seg_logits.data
        probs = torch.softmax(logits, dim=0) # [19, H, W]
        output_permute = probs.permute(1, 2, 0) # [H, W, 19]

        # 1:1 类别重组逻辑 (Cityscapes -> 5 classes)
        # 0: background, 1: bicycle, 2: car, 3: person, 4: rider
        output_reassign = torch.zeros(output_permute.size(0), output_permute.size(1), 5, device=probs.device)
        output_reassign[:, :, 0], _ = torch.max(output_permute[:, :, :11], dim=2) # 0-10: bg
        output_reassign[:, :, 1], _ = torch.max(output_permute[:, :, [17, 18]], dim=2) # 17,18: bike
        output_reassign[:, :, 2], _ = torch.max(output_permute[:, :, [13, 14, 15, 16]], dim=2) # 13-16: vehicle
        output_reassign[:, :, 3] = output_permute[:, :, 11] # 11: person
        output_reassign[:, :, 4] = output_permute[:, :, 12] # 12: rider
        
        # 再次对重组后的 5 类做 softmax
        sf = torch.nn.Softmax(dim=2)
        return sf(output_reassign).cpu().numpy()

    def create_cyclist(self, augmented_lidar):
        """ 完全复刻原仓库 create_cyclist 逻辑 """
        # augmented_lidar: [N, 9] -> [x, y, z, r, bg, bike, car, person, rider]
        rider_idx = np.where(augmented_lidar[:, 8] >= 0.3)[0] 
        rider_points = augmented_lidar[rider_idx]
        bike_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
        bike_total = (np.argmax(augmented_lidar[:, -5:], axis=1) == 1)
        
        for i in range(rider_idx.shape[0]):
            # 空间查询：骑手附近 1 米内的自行车
            bike_mask = (np.linalg.norm(augmented_lidar[:, :3] - rider_points[i, :3], axis=1) < 1) & bike_total
            bike_mask_total |= bike_mask
        
        # 将被识别为 Cyclist 的点进行分数合并/置换
        augmented_lidar[bike_mask_total, 8] = augmented_lidar[bike_mask_total, 5]
        augmented_lidar[bike_total ^ bike_mask_total, 4] = augmented_lidar[bike_total ^ bike_mask_total, 5]
        
        # 返回 8 维数据: [x, y, z, r, bg, bicycle_merged, car, person]
        # 对应原代码 indices: [0, 1, 2, 3, 4, 8, 6, 7]
        return augmented_lidar[:, [0, 1, 2, 3, 4, 8, 6, 7]]

    def augment_lidar_class_scores_both(self, class_scores_r, class_scores_l, lidar_raw, projection_mats):
        """ 完全复刻原仓库双目投影融合逻辑 """
        lidar_velo_coords = lidar_raw.copy()
        reflectances = lidar_velo_coords[:, -1].copy()
        lidar_velo_coords[:, -1] = 1 # 齐次坐标
        
        # 转到相机坐标系
        lidar_cam_coords = projection_mats['Tr_velo_to_cam'].dot(lidar_velo_coords.transpose()).transpose()

        def project_to_mask(P_mat, scores_shape):
            # R0_rect 修正并投影到图像平面
            pts_img = P_mat.dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose())).transpose()
            pts_img = pts_img[:, :2] / pts_img[:, 2:3]
            
            # 边界过滤
            val_flag = (pts_img[:, 0] > 0) & (pts_img[:, 0] < scores_shape[1]) & \
                       (pts_img[:, 1] > 0) & (pts_img[:, 1] < scores_shape[0])
            return pts_img, val_flag

        # 分别计算左右图投影
        pts_img_l, mask_l = project_to_mask(projection_mats['P2'], class_scores_l.shape)
        pts_img_r, mask_r = project_to_mask(projection_mats['P3'], class_scores_r.shape)

        mask_both = mask_l & mask_r
        mask_any = mask_l | mask_r

        # 初始化 9 维增强点云 [4 + 5]
        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], 5))), axis=1)

        # 融合分数
        idx_l = pts_img_l[mask_l].astype(int)
        augmented_lidar[mask_l, -5:] += class_scores_l[idx_l[:, 1], idx_l[:, 0]]
        
        idx_r = pts_img_r[mask_r].astype(int)
        augmented_lidar[mask_r, -5:] += class_scores_r[idx_r[:, 1], idx_r[:, 0]]

        # 双目共有部分取均值
        augmented_lidar[mask_both, -5:] *= 0.5
        
        # 只保留在图像范围内的点 (原代码逻辑)
        augmented_lidar = augmented_lidar[mask_any]
        
        # 处理 Cyclist 并缩减到 8 维
        return self.create_cyclist(augmented_lidar)

    def run(self, num_samples=7481):
        for i in tqdm(range(num_samples)):
            idx = "%06d" % i
            
            points = self.get_lidar(idx)
            scores_l = self.get_score(idx, "image_2/")
            scores_r = self.get_score(idx, "image_3/")
            calib = get_calib_from_file(os.path.join(self.root_path, 'calib', f'{idx}.txt'))
            
            # 核心融合
            painted_points = self.augment_lidar_class_scores_both(scores_r, scores_l, points, calib)
            
            # 保存为 bin (MMDet3D 友好格式)
            painted_points.astype(np.float32).tofile(os.path.join(self.save_path, f'{idx}.bin'))

if __name__ == '__main__':
    SEG_CONFIG = 'configs_seg/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
    SEG_CHECKPOINT = 'checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
    
    painter = Painter(SEG_CONFIG, SEG_CHECKPOINT)
    painter.run()