import torch
import numpy as np
import os
import mmengine
from tqdm import tqdm
from mmseg.apis import inference_model, init_model

# ================= 配置区域 =================
DEVICE = "cuda:7"
DATA_ROOT = "data/nuscenes/"  # 确保此目录下有 samples 文件夹
# nuScenes-mini 经过 MMDet3D 预处理后的 info 文件
INFO_PATH = os.path.join(DATA_ROOT, "nuscenes_infos_train.pkl")

SEG_CONFIG = 'configs_seg/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
SEG_CHECKPOINT = 'checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'

# 保存路径
SAVE_PATH = os.path.join(DATA_ROOT, "samples/LIDAR_TOP_PAINTED")
CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


# ===========================================

class Painter:
    def __init__(self):
        print(f"Loading info file from {INFO_PATH}...")
        # 适配 MMDet3D v1.1+ 的 data_list 结构
        data = mmengine.load(INFO_PATH)
        self.infos = data['data_list']

        print(f"Total frames: {len(self.infos)}")
        self.model = init_model(SEG_CONFIG, SEG_CHECKPOINT, device=DEVICE)
        os.makedirs(SAVE_PATH, exist_ok=True)

    def get_score(self, img_path):
        """语义分割并映射到 4 类评分向量"""
        result = inference_model(self.model, img_path)
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

        h, w = pred_mask.shape
        scores = np.zeros((h, w, 4), dtype=np.float32)
        scores[:, :, 0] = 1.0  # 默认背景

        # 映射 Cityscapes 类别到 Car, Pedestrian, Cyclist
        scores[np.isin(pred_mask, [13, 14, 15, 16]), 3] = 1.0  # Car
        scores[np.isin(pred_mask, [13, 14, 15, 16]), 0] = 0
        scores[np.isin(pred_mask, [11, 12]), 1] = 1.0  # Pedestrian
        scores[np.isin(pred_mask, [11, 12]), 0] = 0
        scores[np.isin(pred_mask, [17, 18]), 2] = 1.0  # Cyclist
        scores[np.isin(pred_mask, [17, 18]), 0] = 0
        return scores

    def run(self):
        for info in tqdm(self.infos):
            # 1. 提取点云路径 (对应图片中的 info['lidar_points']['lidar_path'])
            lidar_rel_path = info['lidar_points']['lidar_path']
            # 手动补全 nuScenes-mini 的 samples 路径
            lidar_path = os.path.join(DATA_ROOT, "samples/LIDAR_TOP", os.path.basename(lidar_rel_path))

            if not os.path.exists(lidar_path):
                continue

            # 加载点云 (nuScenes 默认 5 维，取前 4 维)
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
            paint_feats = np.zeros((points.shape[0], 4), dtype=np.float32)
            paint_feats[:, 0] = 1.0

            # 2. 遍历图像字典 (对应图片中的 info['images'])
            for cam_type, cam_info in info['images'].items():
                if cam_type not in CAMERA_NAMES:
                    continue

                # 修正图片路径
                img_rel_path = cam_info['img_path']
                img_path = os.path.join(DATA_ROOT, "samples", cam_type, os.path.basename(img_rel_path))

                if not os.path.exists(img_path):
                    continue

                img_score = self.get_score(img_path)
                h, w, _ = img_score.shape

                # --- 核心投影逻辑 ---
                pts_lidar = points[:, :3]  # (N, 3)

                # A. 提取 Lidar 到 相机 的变换矩阵 (对应图片中的 lidar2cam, 4x4)
                lidar2cam = np.array(cam_info['lidar2cam'])

                # B. 提取 相机内参 (对应图片中的 cam2img, 3x3)
                cam2img = np.array(cam_info['cam2img'])

                # 将点云转换为齐次坐标 (N, 4)
                pts_extend = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=-1)

                # 投影到相机坐标系
                pts_cam = (pts_extend @ lidar2cam.T)[:, :3]

                # 投影到像素坐标系 (N, 3)
                pts_img = pts_cam @ cam2img.T

                # 归一化处理
                depth = pts_img[:, 2]
                mask_depth = depth > 0.1  # 过滤掉相机背后的点

                u = np.zeros_like(depth, dtype=int)
                v = np.zeros_like(depth, dtype=int)
                u[mask_depth] = np.round(pts_img[mask_depth, 0] / depth[mask_depth]).astype(int)
                v[mask_depth] = np.round(pts_img[mask_depth, 1] / depth[mask_depth]).astype(int)

                # 边界检查
                final_mask = mask_depth & (u >= 0) & (u < w) & (v >= 0) & (v < h)

                if final_mask.any():
                    paint_feats[final_mask] = img_score[v[final_mask], u[final_mask]]

            # 3. 拼接保存
            final_points = np.concatenate([points, paint_feats], axis=-1)
            save_name = os.path.basename(lidar_path)
            final_points.astype(np.float32).tofile(os.path.join(SAVE_PATH, save_name))


# 修改配置区域，改为列表循环处理
INFO_FILES = ["nuscenes_infos_train.pkl", "nuscenes_infos_val.pkl"]

# 在 main 中循环
if __name__ == '__main__':
    for info_name in INFO_FILES:
        # 更新 INFO_PATH
        INFO_PATH = os.path.join(DATA_ROOT, info_name)
        painter = Painter() # 重新初始化以加载不同的 info
        painter.run()