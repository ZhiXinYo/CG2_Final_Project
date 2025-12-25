import pickle

import torch
import numpy as np
import os

os.environ["HF_HUB_OFFLINE"] = "1"
# import mmengine
from PIL import Image
from tqdm import tqdm
# from mmseg.apis import inference_model, init_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ================= 配置区域 =================
DEVICE = "cuda:7"
BATCH = 3
DATA_ROOT = "/home/bliu/cg/data/nuscenes/"  # 确保此目录下有 samples 文件夹
# nuScenes-mini 经过 MMDet3D 预处理后的 info 文件
INFO_PATH = os.path.join(DATA_ROOT, "nuscenes_infos_train.pkl")

SEG_SAVE_NAME = "GS2_SEG"
PAINED_SAVE_NAME = "LIDAR_TOP_PAINTED"

# 保存路径
SAMPLES_SAVE_PATH = os.path.join(DATA_ROOT, "samples", PAINED_SAVE_NAME)
SWEEPS_SAVE_PATH = os.path.join(DATA_ROOT, "sweeps", PAINED_SAVE_NAME)
CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

NUSCENES_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic cone', 'barrier'
]
TEXT_PROMPT = ". ".join(NUSCENES_CLASSES) + "."


# ===========================================

class GS2:
    def __init__(self,
                 model_id="IDEA-Research/grounding-dino-tiny",
                 sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
                 model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                 seg_save_name=SEG_SAVE_NAME,
                 painted_save_name=PAINED_SAVE_NAME
                 ):
        # 1. Grounding DINO
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

        # 2. SAM 2
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        self.predictor = SAM2ImagePredictor(sam2_model)

        self.samples_path = os.path.join(DATA_ROOT, "samples")

        self.seg_save_name = seg_save_name
        self.painted_save_name = painted_save_name

    def predict(self, img_path):
        image_pil = Image.open(img_path)
        image_np = np.array(image_pil.convert("RGB"))
        h, w = image_np.shape[:2]

        # 初始化 11 维评分图 (0: Background, 1-10: NuScenes Classes)
        # 默认背景通道为 1.0
        scores = np.zeros((h, w, 11), dtype=np.float32)
        scores[:, :, 0] = 1.0

        # --- Step 1: Grounding DINO 检测框 ---
        inputs = self.processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.4, text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        if len(results["boxes"]) > 0:
            # --- Step 2: SAM 2 掩码预测 ---
            self.predictor.set_image(image_np)
            masks, _, _ = self.predictor.predict(
                box=results["boxes"].cpu().numpy(),
                multimask_output=False
            )
            # --- 修复后的维度处理逻辑 ---
            # 目标是确保 masks 最终形状为 [N, H, W]
            if masks.ndim == 4:
                # 如果是 [N, 1, H, W]，则去掉轴 1
                masks = masks.squeeze(1)
            # ---------------------------

            # --- Step 3: 映射到 11 维（保留分类分数） ---
            labels = results["labels"]
            confidences = results["scores"]  # 提取 DINO 的置信度分数

            for i, label in enumerate(labels):
                mask = masks[i].astype(bool)
                score = confidences[i].item()  # 获取该目标的具体分数（例如 0.85）

                try:
                    # 匹配 nuScenes 类别
                    idx = next(i for i, cls in enumerate(NUSCENES_CLASSES) if cls in label.lower())
                    channel_idx = idx + 1

                    # 1. 填充该通道为实际的置信度分数，而不是固定的 1.0
                    current_bg_scores = scores[mask, channel_idx]
                    scores[mask, channel_idx] = np.maximum(current_bg_scores, score)

                except StopIteration:
                    continue

        # --- 关键的一步：后处理背景和归一化 ---

        # 2. 重新计算背景分：背景分 = 1.0 - 所有物体通道中的最大值
        # 这样保证了如果有物体，背景分一定会被压低
        max_obj_scores = np.max(scores[:, :, 1:], axis=2)
        scores[:, :, 0] = np.maximum(0.0, 1.0 - max_obj_scores)

        # 3. (高级技巧) 归一化：确保每个像素 11 维之和严格等于 1.0
        # 这能让 3D 检测器更容易学习（输入特征的量纲一致）
        sum_norm = np.sum(scores, axis=2, keepdims=True)
        scores /= (sum_norm + 1e-6)  # 防止除零

        return scores, image_np, results, masks if len(results["boxes"]) > 0 else None

    def predict_batch(self, img_paths):
        # --- A. 准备数据 ---
        images_pil = [Image.open(p) for p in img_paths]
        images_np = [np.array(p.convert("RGB")) for p in images_pil]

        # --- B. Grounding DINO 批量检测 ---
        inputs = self.processor(images=images_pil, text=[TEXT_PROMPT] * len(img_paths), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        batch_results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.4, text_threshold=0.3,
            target_sizes=[p.size[::-1] for p in images_pil]
        )

        # --- C. SAM 2 真正批处理逻辑 ---
        # 1. 一次性对 6 张图进行特征提取 (Feature Encoding)
        # 构造图片 Tensor [6, 3, H, W] 并编码
        # 注意：这里需要确保所有图片尺寸一致（nuScenes 都是 900x1600）
        self.predictor.set_image_batch(images_np)

        batch_scores = []

        # 2. 遍历每张图的结果并生成掩码
        # 虽然这里有循环，但由于 set_image_batch 已经把特征算好了，predict 会非常快
        for i in range(len(img_paths)):
            h, w = images_np[i].shape[:2]
            scores = np.zeros((h, w, 11), dtype=np.float32)
            scores[:, :, 0] = 1.0  # 默认背景

            res = batch_results[i]

            if len(res["boxes"]) > 0:
                # 使用已经在显存里的第 i 张图的特征
                # predictor.predict 会自动关联 set_image_batch 里的索引
                masks, _, _ = self.predictor.predict(
                    box=res["boxes"].cpu().numpy(),
                    multimask_output=False,
                    # 注意：如果 SAM2 版本支持，可以直接在这里通过索引调用
                )

                if masks.ndim == 4: masks = masks.squeeze(1)

                # --- D. 映射分数 (与之前逻辑一致) ---
                for j, label in enumerate(res["labels"]):
                    try:
                        idx = next(k for k, cls in enumerate(NUSCENES_CLASSES) if cls in label.lower())
                        score_val = res["scores"][j].item()
                        scores[masks[j].astype(bool), idx + 1] = score_val
                    except StopIteration:
                        continue

            # --- E. 归一化 (与之前逻辑一致) ---
            max_obj = np.max(scores[:, :, 1:], axis=2)
            scores[:, :, 0] = np.maximum(0.0, 1.0 - max_obj)
            scores /= (np.sum(scores, axis=2, keepdims=True) + 1e-6)

            batch_scores.append(scores)

        return batch_scores

    def predict_all(self, infos_name):
        with open(os.path.join(DATA_ROOT, infos_name), 'rb') as f:
            data = pickle.load(f)
        infos = data['data_list']
        save_paths = {cam_name: os.path.join(self.samples_path, self.seg_save_name, cam_name) for cam_name in CAMERA_NAMES}
        for save_path in save_paths.values():
            os.makedirs(save_path, exist_ok=True)
        for info in tqdm(infos):
            for k_name, img_file in info["images"].items():
                img_path = img_file["img_path"]
                img_path = os.path.join(self.samples_path, k_name, img_path)
                scores, _, _, _ = self.predict(img_path)
                npy_name = os.path.join(save_paths[k_name], os.path.basename(img_path).split('.')[0] + ".npy")
                np.save(npy_name, scores.astype(np.float32))


class Painter:
    def __init__(self, save_path=SAMPLES_SAVE_PATH):
        self.seg_npy_path = os.path.join(DATA_ROOT, "samples", SEG_SAVE_NAME)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.gs2 = GS2()

    def run(self, infos_name):
        with open(os.path.join(DATA_ROOT, infos_name), 'rb') as f:
            data = pickle.load(f)
        infos = data['data_list']
        for info in tqdm(infos):
            # 1. 提取点云路径 (对应图片中的 info['lidar_points']['lidar_path'])
            lidar_rel_path = info['lidar_points']['lidar_path']
            # 手动补全 nuScenes-mini 的 samples 路径
            lidar_path = os.path.join(DATA_ROOT, "samples/LIDAR_TOP", os.path.basename(lidar_rel_path))

            # 加载点云 (nuScenes 默认 5 维，取前 5 维)
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            paint_feats = np.zeros((points.shape[0], len(NUSCENES_CLASSES) + 1), dtype=np.float32)

            # --- 准备阶段 ---
            img_paths = []
            cam_types = []
            for cam_type, cam_info in info['images'].items():
                img_rel_path = cam_info['img_path']
                img_path = os.path.join(DATA_ROOT, "samples", cam_type, os.path.basename(img_rel_path))
                img_paths.append(img_path)
                cam_types.append(cam_type)

            # ---------------------------------------------------------
            # 阶段一：分批次批量分割 (GPU 密集型)
            # ---------------------------------------------------------
            all_seg_scores = []

            # 将 6 张图分成多组进行推理
            for i in range(0, len(img_paths), BATCH):
                # 截取当前批次的路径 (例如 0:3, 3:6)
                batch_img_paths = img_paths[i: i + BATCH]

                # 调用你的批量推理函数
                batch_results = self.gs2.predict_batch(batch_img_paths)

                # 将这一批的结果添加到汇总列表中
                all_seg_scores.extend(batch_results)

            # ---------------------------------------------------------
            # 阶段二：逐相机投影染色 (CPU 数学运算)
            # ---------------------------------------------------------
            for i, cam_type in enumerate(cam_types):
                seg_score = all_seg_scores[i]
                cam_info = info['images'][cam_type]
                h, w, _ = seg_score.shape

                # --- 投影数学计算 (保持不变) ---
                lidar2cam = np.array(cam_info['lidar2cam'])
                cam2img = np.array(cam_info['cam2img'])

                pts_lidar = points[:, :3]
                pts_extend = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=-1)
                pts_cam = (pts_extend @ lidar2cam.T)[:, :3]
                pts_img = pts_cam @ cam2img.T

                depth = pts_img[:, 2]
                mask_depth = depth > 0.1

                u = np.zeros_like(depth, dtype=int)
                v = np.zeros_like(depth, dtype=int)
                u[mask_depth] = np.round(pts_img[mask_depth, 0] / depth[mask_depth]).astype(int)
                v[mask_depth] = np.round(pts_img[mask_depth, 1] / depth[mask_depth]).astype(int)

                final_mask = mask_depth & (u >= 0) & (u < w) & (v >= 0) & (v < h)

                if final_mask.any():
                    # --- 竞争更新逻辑 (保持不变) ---
                    new_scores = seg_score[v[final_mask], u[final_mask]]
                    old_scores = paint_feats[final_mask]

                    new_conf = np.max(new_scores[:, 1:], axis=1)
                    old_conf = np.max(old_scores[:, 1:], axis=1)
                    better_mask = new_conf > old_conf

                    if better_mask.any():
                        update_indices = np.where(final_mask)[0][better_mask]
                        paint_feats[update_indices] = new_scores[better_mask]

            # 3. 最终拼接保存
            final_points = np.concatenate([points, paint_feats], axis=-1)
            save_name = os.path.basename(lidar_path)
            final_points.astype(np.float32).tofile(os.path.join(self.save_path, save_name))

            # # 2. 遍历图像字典 (对应图片中的 info['images'])
            # for cam_type, cam_info in info['images'].items():
            #     # 修正图片路径
            #     img_rel_path = cam_info['img_path']
            #     img_path = os.path.join(DATA_ROOT, "samples", cam_type, os.path.basename(img_rel_path))
            #     seg_score, _, _, _ = self.gs2.predict(img_path)
            #
            #     h, w, c = seg_score.shape
            #
            #     # --- 核心投影逻辑 ---
            #     pts_lidar = points[:, :3]  # (N, 3)
            #
            #     # A. 提取 Lidar 到 相机 的变换矩阵 (对应图片中的 lidar2cam, 4x4)
            #     lidar2cam = np.array(cam_info['lidar2cam'])
            #
            #     # B. 提取 相机内参 (对应图片中的 cam2img, 3x3)
            #     cam2img = np.array(cam_info['cam2img'])
            #
            #     # 将点云转换为齐次坐标 (N, 4)
            #     pts_extend = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=-1)
            #
            #     # 投影到相机坐标系
            #     pts_cam = (pts_extend @ lidar2cam.T)[:, :3]
            #
            #     # 投影到像素坐标系 (N, 3)
            #     pts_img = pts_cam @ cam2img.T
            #
            #     # 归一化处理
            #     depth = pts_img[:, 2]
            #     mask_depth = depth > 0.1  # 过滤掉相机背后的点
            #
            #     u = np.zeros_like(depth, dtype=int)
            #     v = np.zeros_like(depth, dtype=int)
            #     u[mask_depth] = np.round(pts_img[mask_depth, 0] / depth[mask_depth]).astype(int)
            #     v[mask_depth] = np.round(pts_img[mask_depth, 1] / depth[mask_depth]).astype(int)
            #
            #     # 边界检查
            #     final_mask = mask_depth & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            #
            #     if final_mask.any():
            #         # 1. 提取当前相机预测的新分数 (N_mask, 11)
            #         new_scores = seg_score[v[final_mask], u[final_mask]]
            #
            #         # 2. 提取该点已经保存的旧分数 (N_mask, 11)
            #         old_scores = paint_feats[final_mask]
            #
            #         # 3. 计算新旧分数的最大置信度 (N_mask,)
            #         # 注意：通常我们不看背景通道（index 0），只看物体通道的最大置信度
            #         new_conf = np.max(new_scores[:, 1:], axis=1)
            #         old_conf = np.max(old_scores[:, 1:], axis=1)
            #
            #         # 4. 找到新置信度更高点的索引
            #         better_mask = new_conf > old_conf
            #
            #         # 5. 只在更好的位置更新全部通道
            #         if better_mask.any():
            #             # 注意这里的索引技巧：final_mask 选出投影区域，better_mask 选出其中更优的点
            #             update_indices = np.where(final_mask)[0][better_mask]
            #             paint_feats[update_indices] = new_scores[better_mask]
            #
            # # 3. 拼接保存
            # final_points = np.concatenate([points, paint_feats], axis=-1)
            # save_name = os.path.basename(lidar_path)
            # final_points.astype(np.float32).tofile(os.path.join(self.save_path, save_name))


# 修改配置区域，改为列表循环处理
INFOS_FILES = ["nuscenes_infos_train.pkl", "nuscenes_infos_val.pkl"]

# 在 main 中循环
if __name__ == '__main__':
    # for infos_name in INFOS_FILES:
    #     gs2 = GS2()
    #     gs2.predict_all(infos_name)

    for infos_name in INFOS_FILES:
        painter = Painter()
        painter.run(infos_name)

    sweeps_lidars_path = os.path.join(DATA_ROOT, "sweeps/LIDAR_TOP")
    os.makedirs(SWEEPS_SAVE_PATH, exist_ok=True)
    for lidar_file in tqdm(os.listdir(sweeps_lidars_path)):
        # 1. 加载原始点云 (N, 5)
        file_path = os.path.join(sweeps_lidars_path, lidar_file)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)

        # 2. 创建 11 维的空特征 (N, 11)
        # 默认初始化为全 0
        paint_feats = np.zeros((points.shape[0], len(NUSCENES_CLASSES) + 1), dtype=np.float32)

        # # 3. 将第一位（背景类）设为 1.0
        # # 这样模型知道这些点在语义上属于“背景/未知”
        # paint_feats[:, 0] = 1.0

        # 4. 拼接 (N, 5) + (N, 11) -> (N, 16)
        final_points = np.concatenate([points, paint_feats], axis=-1)

        # 5. 保存为二进制文件
        save_file_path = os.path.join(SWEEPS_SAVE_PATH, os.path.basename(lidar_file))
        final_points.astype(np.float32).tofile(save_file_path)

