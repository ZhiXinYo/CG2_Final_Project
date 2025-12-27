import copy
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from mmengine import MessageHub

from mmdet3d.models.detectors import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


@MODELS.register_module()
class YoloPointPainter(MVXTwoStageDetector):
    def __init__(self, img_bbox_head=None, **kwargs):
        super(YoloPointPainter, self).__init__(**kwargs)
        self.img_bbox_head = MODELS.build(img_bbox_head)

    def extract_img_feat(self, img, input_metas):
        """提取图像特征并处理 5D Tensor (B, N, C, H, W)."""
        if self.with_img_backbone and img is not None:
            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            img_feats = self.img_backbone(img)
            if self.with_img_neck:
                img_feats = self.img_neck(img_feats)
            return img_feats
        return None

    def painting(self, batch_input_metas, points, head_outputs):
        # ==================== [DEBUG_START: 重复检测与进度记录] ====================
        # if not hasattr(self, 'seen_tokens'):
        #     self.seen_tokens = set()
        #
        # message_hub = MessageHub.get_current_instance()
        # cur_epoch = message_hub.get_info('epoch')
        # cur_iter = message_hub.get_info('iter')
        #
        # if not hasattr(self, 'last_epoch_tracker'):
        #     self.last_epoch_tracker = cur_epoch
        # if cur_epoch != self.last_epoch_tracker:
        #     self.seen_tokens.clear()
        #     self.last_epoch_tracker = cur_epoch
        #
        # print(f"\n>>> DEBUG_LOG: Epoch [{cur_epoch}] | Global Iter [{cur_iter}]")
        #
        # for meta in batch_input_metas:
        #     s_id = meta.get('token', meta.get('sample_idx', 'unknown'))
        #     if s_id in self.seen_tokens:
        #         print(f"  [!] 重复检测警告: 样本 {s_id} 已在该 Epoch 出现过")
        #     else:
        #         self.seen_tokens.add(s_id)
        # ==================== [DEBUG_END] ====================

        # --- 提取 2D 预测图 ---
        cls_logits = head_outputs[0]
        cls_scores = [torch.sigmoid(l).float() for l in cls_logits]
        resized_scores = [F.interpolate(s, size=cls_scores[0].shape[-2:], mode='nearest') for s in cls_scores]
        final_cls_map = torch.max(torch.stack(resized_scores), dim=0)[0]

        B = len(points)
        num_cams = final_cls_map.shape[0] // B
        num_classes = final_cls_map.shape[1]
        p_list = []

        for i in range(B):
            img_meta = batch_input_metas[i]
            device = points[i].device
            num_pts = points[i].shape[0]

            # 尺寸定义
            img_h, img_w = 576, 1024
            if 'img_shape' in img_meta:
                raw_shape = img_meta['img_shape']
                if isinstance(raw_shape, (list, tuple)) and isinstance(raw_shape[0], (list, tuple)):
                    img_h, img_w = raw_shape[0][:2]
                else:
                    img_h, img_w = raw_shape[:2]

            # --- 1. 【修复关键】首先定义并克隆点云 ---
            cur_points = points[i][:, :3].clone().float()

            # ==================== [DEBUG_START: 逆转翻转增强] ====================
            # 翻转逻辑必须在 cur_points 定义后，且建议在平移/旋转之前逆转
            if img_meta.get('pcd_horizontal_flip', False):
                # 水平翻转在 LiDAR 系通常是 Y 轴取反
                cur_points[:, 1] = -cur_points[:, 1]
            if img_meta.get('pcd_vertical_flip', False):
                # 垂直翻转在 LiDAR 系通常是 X 轴取反
                cur_points[:, 0] = -cur_points[:, 0]
            # ==================== [DEBUG_END] ====================

            # --- 2. 逆转其他 3D 数据增强 ---
            if 'pcd_trans' in img_meta:
                trans = cur_points.new_tensor(img_meta['pcd_trans']).to(device)
                cur_points -= trans
            if 'pcd_scale_factor' in img_meta:
                scale = img_meta['pcd_scale_factor']
                cur_points /= scale
            if 'pcd_rotation' in img_meta:
                rot_mat_T = cur_points.new_tensor(img_meta['pcd_rotation']).to(device)
                # 旋转矩阵是正交阵，其逆变换是转置
                cur_points = torch.matmul(cur_points, rot_mat_T.t())

            # --- 3. 投影计算 ---
            K = cur_points.new_tensor(img_meta['cam2img']).to(device).float()
            T_ext = cur_points.new_tensor(img_meta['lidar2cam']).to(device).float()
            lidar2img = K @ T_ext[:, :3, :4]

            pts_homo = torch.cat([cur_points, cur_points.new_ones((num_pts, 1))], dim=1)
            pts_img_homo = torch.matmul(lidar2img, pts_homo.t())

            z = pts_img_homo[:, 2:3, :]
            mask_z = (z > 0.5)
            safe_z = torch.where(mask_z, z, torch.ones_like(z))
            uv = pts_img_homo[:, :2, :] / safe_z
            uv_permuted = uv.permute(0, 2, 1)

            # ==================== [DEBUG_START: 可视化] ====================
            # self._visualize_projection(img_meta, uv_permuted, mask_z.squeeze(1), img_h, img_w, i)
            # ==================== [DEBUG_END] ====================

            # --- 4. 采样逻辑 ---
            grid_uv = (uv_permuted / uv_permuted.new_tensor([img_w, img_h]).to(device)) * 2.0 - 1.0
            valid_mask = (mask_z.squeeze(1)) & \
                         (grid_uv[..., 0] >= -1.0) & (grid_uv[..., 0] <= 1.0) & \
                         (grid_uv[..., 1] >= -1.0) & (grid_uv[..., 1] <= 1.0)

            grid = grid_uv.view(num_cams, 1, num_pts, 2)
            cur_cam_feats = final_cls_map[i * num_cams: (i + 1) * num_cams]
            sampled_feat = F.grid_sample(cur_cam_feats, grid, mode='nearest', padding_mode='zeros',
                                         align_corners=False).view(num_cams, num_classes, num_pts)
            sampled_feat = sampled_feat * valid_mask.view(num_cams, 1, num_pts)
            final_feat, _ = torch.max(sampled_feat, dim=0)
            p_list.append(torch.cat([points[i], final_feat.t()], dim=-1))

            # ==================== [DEBUG_START: 采样统计] ====================
            # with torch.no_grad():
            #     pts_in_img_mask = valid_mask.any(dim=0)
            #     num_pts_in_img = pts_in_img_mask.sum().item()
            #     max_2d_score = cur_cam_feats.max().item()
            #     max_sampled_score = final_feat.max().item()
            #     print(
            #         f"  [Sample {i}] In-Img: {num_pts_in_img}/{num_pts} | 2D Max: {max_2d_score:.4f} | Sample Max: {max_sampled_score:.4f}")
            # ==================== [DEBUG_END] ====================

        return p_list

    # ==================== [DEBUG_START: 可视化辅助函数] ====================
    def _visualize_projection(self, img_meta, uv_coords, depth_mask, img_h, img_w, batch_idx):
        save_dir = 'vis_painting'
        os.makedirs(save_dir, exist_ok=True)
        from mmengine.logging import MessageHub
        hub = MessageHub.get_current_instance()
        cur_iter = hub.get_info('iter')

        s_id = img_meta.get('token', img_meta.get('sample_idx', 'none'))
        s_id_short = s_id[:6] if isinstance(s_id, str) else s_id

        real_h, real_w = img_h, img_w
        if 'pad_shape' in img_meta:
            real_h, real_w = img_meta['pad_shape'][:2]

        img_paths = img_meta['img_path']
        for cam_idx in range(len(img_paths)):
            img = cv2.imread(img_paths[cam_idx])
            if img is None: continue
            img = cv2.resize(img, (int(real_w), int(real_h)))

            cam_uv = uv_coords[cam_idx]
            in_view_mask = depth_mask[cam_idx] & \
                           (cam_uv[:, 0] >= 0) & (cam_uv[:, 0] < real_w - 1) & \
                           (cam_uv[:, 1] >= 0) & (cam_uv[:, 1] < real_h - 1)

            valid_uv = cam_uv[in_view_mask].cpu().numpy()
            if len(valid_uv) > 0:
                indices = np.random.choice(len(valid_uv), min(len(valid_uv), 800), replace=False)
                for idx in indices:
                    u, v = valid_uv[idx]
                    cv2.circle(img, (int(u), int(v)), 2, (0, 0, 255), -1)

            fname = f"iter{cur_iter}_id{s_id_short}_b{batch_idx}_c{cam_idx}.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), img)

    # ==================== [DEBUG_END] ====================

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)

        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        head_outputs = self.img_bbox_head(img_feats)
        self._cache_head_outputs = head_outputs

        # ==================== [DIAGNOSE_START: 2D 逐层状态检查] ====================
        # print(f"\n" + "=" * 50 + "\n[DIAGNOSE] 2D Branch Weight/Feature Status:")

        # # 1. 检查图像输入 (确保 DataPreprocessor 没把图弄坏)
        # if imgs is not None:
        #     print(f"  - Raw Img: Mean={imgs.mean():.4f}, Var={imgs.var():.4f}")

        # 2. 分解图像提取过程
        # img_feats = None
        # if self.with_img_backbone and imgs is not None:
        #     # 执行 Backbone
        #     if imgs.dim() == 5:
        #         B, N, C, H, W = imgs.size()
        #         _imgs = imgs.view(B * N, C, H, W)
        #     else:
        #         _imgs = imgs
        #
        #     backbone_feats = self.img_backbone(_imgs)
        #     print(f"  - Backbone Out (P5): Mean={backbone_feats[-1].mean():.4f}, Var={backbone_feats[-1].var():.4f}")
        #
        #     # 执行 Neck
        #     if self.with_img_neck:
        #         img_feats = self.img_neck(backbone_feats)
        #         print(f"  - Neck Out (P5): Mean={img_feats[-1].mean():.4f}, Var={img_feats[-1].var():.4f}")
        #     else:
        #         img_feats = backbone_feats
        # ==================== [DIAGNOSE_END] ====================

        # ==================== [DIAGNOSE_START: Head 输出检查] ====================
        # cls_logits = head_outputs[0]
        # 检查分类头
        # print(f"  - Head Logits (Stride 8): Mean={cls_logits[0].mean():.4f}, Var={cls_logits[0].var():.4f}")
        # print(f"  - Head Max Logit: {cls_logits[0].max():.4f} | Min Logit: {cls_logits[0].min():.4f}")
        # print("=" * 50 + "\n")
        # ==================== [DIAGNOSE_END] ====================

        pts_feats = None
        # 2D 辅助特征涂色
        points = self.painting(batch_input_metas, points, head_outputs)

        # 3D 分支
        voxel_dict = self.data_preprocessor.voxelize(points, batch_input_metas)
        pts_feats = self.extract_pts_feat(voxel_dict, points, img_feats, batch_input_metas)

        return (img_feats, pts_feats)

    def loss_imgs(self, x, batch_data_samples, **kwargs):
        """修复元数据构造，适配 YOLOv8Head."""
        if not self.with_img_bbox or not hasattr(self, '_cache_head_outputs'):
            return dict()

        head_outputs = self._cache_head_outputs
        if head_outputs is None: return dict()

        cls_logits, bbox_preds, bbox_dist_preds = head_outputs
        all_gt_instances = []
        all_img_metas = []
        device = x[0].device
        target_dtype = x[0].dtype  # 通常是 float16 (FP16) 或 float32
        num_views = batch_data_samples[0].metainfo['num_views']
        batch_size = len(batch_data_samples)

        for sample_idx, sample_3d in enumerate(batch_data_samples):
            base_meta = sample_3d.metainfo
            # print(base_meta.keys())
            cam_instances_dict = sample_3d.get('gt_instances', {}).get('cam_instances', {})
            for cam_idx, cam_key in enumerate(cam_instances_dict.keys()):
                # --- 1. 修复嵌套的 img_shape ---
                img_shape = base_meta['img_shape'][cam_idx]
                # --- 2. 获取缩放因子 (用于将 1600 坐标映射到 1024) ---
                # 假设 scale_factor 是 [w_scale, h_scale]
                cur_scale = base_meta['scale_factor'][cam_idx]
                cur_meta = {
                    'img_shape': img_shape,
                    'ori_shape': base_meta['ori_shape'][cam_idx],
                    'pad_shape': base_meta['pad_shape'],
                    'scale_factor': cur_scale,
                    'img_path': base_meta['img_path'][cam_idx]  # <--- 必须补上这一行！
                }
                all_img_metas.append(cur_meta)
                gt_instances = InstanceData()
                instances = cam_instances_dict.get(cam_key)
                if len(instances) > 0:
                    # --- 3. 关键修复：尺度缩放 + 精度转换 ---
                    bboxes = torch.as_tensor([inst['bbox'] for inst in instances],
                                             device=device, dtype=target_dtype)

                    # 将原图尺度 (1600, 900) 乘以缩放因子 (例如 0.64) 转换到 (1024, 576)
                    # bboxes 格式为 [x1, y1, x2, y2]
                    bboxes[:, [0, 2]] *= cur_scale[0]  # scale_x
                    bboxes[:, [1, 3]] *= cur_scale[1]  # scale_y

                    gt_instances.bboxes = bboxes
                    gt_instances.labels = torch.as_tensor([inst['bbox_label'] for inst in instances],
                                                          device=device, dtype=torch.long)
                else:
                    gt_instances.bboxes = torch.zeros((0, 4), device=device, dtype=target_dtype)
                    gt_instances.labels = torch.zeros((0,), device=device, dtype=torch.long)
                all_gt_instances.append(gt_instances)
        # ==================== [FINAL CHECK DEBUG] ====================
        # if len(all_gt_instances) > 0:
        #     # 1. 检查标签数值范围
        #     all_labels = torch.cat([gt.labels for gt in all_gt_instances])
        #     unique_l = torch.unique(all_labels).tolist()
        #     print(f"\n>>> [CRITICAL CHECK] Unique Labels in this batch: {unique_l}")
        #
        #     # 2. 检查是否有样本被成功分配 (TAL Assigner 模拟检查)
        #     # 如果 Max Score 这么低，说明模型根本没把这些框当成正样本
        #     if any(l < 0 for l in unique_l):
        #         print("!!! WARNING: Found negative labels (-1), YOLO will ignore these!")
        #
        # # 3. 检查特征图数值是否溢出
        # print(f">>> DEBUG 2D: cls_logits Mean={cls_logits[0].mean():.2f}, Var={cls_logits[0].var():.2f}")
        # # [DEBUG] 在 loss_imgs 的 bboxes 缩放逻辑之后插入
        # if len(gt_instances.bboxes) > 0:
        #     print(f"\n>>> [BOX_DEBUG] Sample Image Shape: {img_shape}")
        #     print(f"    - Scale factor used: {cur_scale}")
        #     print(f"    - GT Bboxes (First 3): \n{gt_instances.bboxes[:3]}")
        #
        #     # 检查是否溢出边界
        #     x_max = gt_instances.bboxes[:, [0, 2]].max()
        #     y_max = gt_instances.bboxes[:, [1, 3]].max()
        #     if x_max > img_shape[1] or y_max > img_shape[0]:
        #         print(f"    [!!!] WARNING: Boxes are OUT OF BOUNDS! Max X: {x_max}, Max Y: {y_max}")

        # 在 loss_by_feat 之前
        # self._debug_save_gt_images(all_img_metas, all_gt_instances, 0)
        # ==================== [DEBUG END] ====================
        # 调用 Head 计算损失
        img_losses = self.img_bbox_head.loss_by_feat(
            cls_logits, bbox_preds, bbox_dist_preds,
            batch_gt_instances=all_gt_instances,
            batch_img_metas=all_img_metas,
            **kwargs
        )
        self._cache_head_outputs = None
        return {f'{k}_2d': v / (num_views * batch_size) for k, v in img_losses.items()}
        # return {f'{k}_2d': v for k, v in img_losses.items()}

    # def _debug_save_gt_images(self, img_metas, gt_instances, batch_idx):
    #     save_dir = 'debug_gt_boxes_original'
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir, exist_ok=True)
    #
    #     # 此时 gt_instances 和 img_metas 的长度应当一致，均为 B * N
    #     for i, gt in enumerate(gt_instances):
    #         meta = img_metas[i]
    #         # --- 修复路径获取逻辑 ---
    #         # 在之前的 loss_imgs 中，我们已经确保 img_metas[i] 存储的是该视角唯一的路径
    #         img_path = meta.get('img_path', None)
    #         # 1. 读取该视角对应的原图
    #         img = cv2.imread(img_path)
    #         # 2. 获取该视角特有的缩放比例
    #         cur_scale = meta.get('scale_factor')
    #         boxes = gt.bboxes.detach().cpu().numpy()
    #         labels = gt.labels.detach().cpu().numpy()
    #         if len(boxes) > 0:
    #             for box, label in zip(boxes, labels):
    #                 # 3. 逆缩放：从模型输入尺度回到原图尺度
    #                 x1 = int(box[0] / cur_scale[0])
    #                 y1 = int(box[1] / cur_scale[1])
    #                 x2 = int(box[2] / cur_scale[0])
    #                 y2 = int(box[3] / cur_scale[1])
    #
    #                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 cv2.putText(img, f"cls:{label}", (x1, max(0, y1 - 10)),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #
    #             # 文件名加入视角信息以便区分
    #             view_name = os.path.dirname(img_path).split('/')[-1]  # 尝试提取如 CAM_FRONT
    #             fname = f"b{batch_idx}_v{i}_{view_name}_{os.path.basename(img_path)}"
    #             target_path = os.path.join(save_dir, fname)
    #             cv2.imwrite(target_path, img)
    #             print(f"  >>> [DEBUG_VIS] Saved View {i} to {target_path}")

    def predict_imgs(self, x, batch_data_samples, rescale=True, **kwargs):
        """修复推理阶段逻辑：使用 img_path 作为唯一 ID 适配 2D 评估."""
        if not hasattr(self, '_cache_head_outputs') or self._cache_head_outputs is None:
            self._cache_head_outputs = self.img_bbox_head(x)

        # 在推理逻辑中，获取 cls_logit 后
        # 假设 cls_logit 的形状是 (bs, num_classes, h, w)
        # cls_logits = self._cache_head_outputs[0]
        # 在 predict_imgs 中获取 cls_logits 后的位置
        # 假设 cls_logits 是从 head.predict 或类似函数返回的 list

        # for i, logit in enumerate(cls_logits):
        #     # logit 形状通常是 (bs, num_classes, h, w) 或 (bs, h*w, num_classes)
        #     m_score = logit.sigmoid().max().item()
        #     a_score = logit.sigmoid().mean().item()
        #     print(f">>> [LAYER {i}] Max: {m_score:.6f}, Avg: {a_score:.6f}, Shape: {logit.shape}")
        #
        # # 如果你只想看全图最高分
        # total_max = max([l.sigmoid().max().item() for l in cls_logits])
        # print(f">>> [INFERENCE DEBUG] Global Max Sigmoid Score: {total_max:.6f}")

        all_img_metas = []
        for sample_3d in batch_data_samples:
            base_meta = sample_3d.metainfo
            # 调试用打印可以保留，确认 img_path 是长度为 6 的列表
            # print(f"Processing sample_idx: {base_meta['sample_idx']}")

            num_views = base_meta.get('num_views', 6)

            for i in range(num_views):
                img_meta = {
                    'img_shape': base_meta['img_shape'][i] if isinstance(base_meta['img_shape'], list) else base_meta[
                        'img_shape'],
                    'ori_shape': base_meta.get('ori_shape')[i],
                    'scale_factor': base_meta['scale_factor'][i] if isinstance(base_meta['scale_factor'], list) else
                    base_meta['scale_factor'],
                }
                if 'pad_param' in base_meta:
                    pp = base_meta['pad_param']
                    img_meta['pad_param'] = pp[i] if isinstance(pp, list) else pp

                all_img_metas.append(img_meta)

        # 1. 获得 2D 检测结果 (InstanceData 列表，长度为 Batch * Num_Views)
        results_list = self.img_bbox_head.predict_by_feat(
            *self._cache_head_outputs[:2],
            batch_img_metas=all_img_metas,
            rescale=rescale,
            **kwargs
        )
        self._cache_head_outputs = None
        for batch_id, sample_3d in enumerate(batch_data_samples):
            img_paths = sample_3d.metainfo['img_path']
            # 必须以 img_paths 的顺序来循环，因为 results_list 是按这个顺序排的
            for view_idx, img_path in enumerate(img_paths):
                target_idx = batch_id * 6 + view_idx
                results_list[target_idx].set_metainfo(dict(img_id=os.path.basename(img_path)))

        # results_list = []
        # 2. 【核心修改】注入 img_id。由于没有 'images' 字典，我们直接使用 'img_path'
        # 注意：results_list 是平铺的列表，我们需要按顺序对应到每个 sample 的每个 view
        # idx = 0
        # for batch_id, sample_3d in enumerate(batch_data_samples):
        #     cam_instances_dict = sample_3d.get('gt_instances').get('cam_instances')
        #     img_paths = sample_3d.metainfo['img_path']
        #     # 必须以 img_paths 的顺序来循环，因为 results_list 是按这个顺序排的
        #     for view_idx, (cam_k, cam_v) in enumerate(cam_instances_dict.items()):
        #         target_idx = batch_id * 6 + view_idx
        #         # --- 重新构造 InstanceData ---
        #         new_instances = InstanceData()
        #         if len(cam_v) > 0:
        #             # 这里拿出的 bbox 必须手动缩放且 clamp，否则 AP 依然不是 1
        #             bboxes = torch.as_tensor([inst['bbox'] for inst in cam_v], device="cuda", dtype=torch.float32)
        #
        #             # 这里的 bbox_label 必须检查是否与 Dataset 的类目 ID 一致
        #             labels = torch.as_tensor([inst['bbox_label'] for inst in cam_v], device="cuda", dtype=torch.long)
        #             scores = torch.ones(len(cam_v), device="cuda")
        #
        #             new_instances.bboxes = bboxes
        #             new_instances.labels = labels
        #             new_instances.scores = scores
        #         else:
        #             new_instances.bboxes = torch.zeros((0, 4), device="cuda")
        #             new_instances.labels = torch.zeros((0,), device="cuda", dtype=torch.long)
        #             new_instances.scores = torch.zeros((0,), device="cuda")
        #
        #         # 注入元信息，img_id 必须与评估器加载的数据集 JSON 里的 file_name 完全一致
        #         new_instances.set_metainfo(dict(img_id=os.path.basename(img_paths[view_idx])))
        #         results_list.append(new_instances)
        return results_list

    def extract_pts_feat(self, voxel_dict, points=None, img_feats=None, batch_input_metas=None):
        if not self.with_pts_bbox: return None
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors'],
                                                img_feats, batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck: x = self.pts_neck(x)
        return x

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        if pts_feats and self.with_pts_bbox:
            results_list_3d = self.pts_bbox_head.predict(
                pts_feats, batch_data_samples, **kwargs)
        else:
            results_list_3d = None

        if img_feats and self.with_img_bbox:
            # TODO check this for camera modality
            results_list_2d = self.predict_imgs(img_feats, batch_data_samples,
                                                **kwargs)
        else:
            results_list_2d = None

        # 2. 手动组装返回列表
        for i in range(len(batch_data_samples)):
            # 挂载 3D 结果（NuScenesMetric 主要看这里）
            if results_list_3d is not None:
                batch_data_samples[i].pred_instances_3d = results_list_3d[i]

            # 挂载 2D 结果（CocoMetric 及其它 2D Metric 看这里）
            if results_list_2d is not None:
                batch_data_samples[i].pred_instances = None
                num_views = batch_data_samples[i].num_views
                views2d_results = results_list_2d[num_views*i:num_views*i+num_views]
                batch_data_samples[i].views2d_results = views2d_results
        return batch_data_samples

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.pts_bbox_head.loss(pts_feats, batch_data_samples,
                                                 **kwargs)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.loss_imgs(img_feats, batch_data_samples)
            losses.update(losses_img)
        return losses