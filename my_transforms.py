import torch
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SemanticAwareDistanceSampling(BaseTransform):
    def __init__(self, 
                 dist_threshold=25.0, 
                 keep_ratio_bg=0.3,
                 feat_channels=4):
        self.dist_threshold = dist_threshold
        self.keep_ratio_bg = keep_ratio_bg
        self.feat_channels = feat_channels

    def transform(self, results):
        points = results['points']
        pts_tensor = points.tensor
        
        # 计算距离
        dist = torch.norm(pts_tensor[:, :2], p=2, dim=1)
        
        semantic_scores = pts_tensor[:, -self.feat_channels:]
        foreground_scores = semantic_scores[:, 1:] # 取出 Ped, Cyc, Car 的分数
        is_foreground = torch.argmax(foreground_scores, dim=1) > 0

        can_drop_mask = (dist < self.dist_threshold) & (~is_foreground)
        keep_mask = torch.ones(len(pts_tensor), dtype=torch.bool, device=pts_tensor.device)
        
        drop_candidate_indices = torch.where(can_drop_mask)[0]
        num_drop_candidates = len(drop_candidate_indices)
        
        if num_drop_candidates > 0:
            num_keep = int(num_drop_candidates * self.keep_ratio_bg)
            rand_perm = torch.randperm(num_drop_candidates)
            drop_indices = drop_candidate_indices[rand_perm[num_keep:]]
            keep_mask[drop_indices] = False
            
        results['points'] = points[keep_mask]
        return results



@TRANSFORMS.register_module()
class ImprovedSADS(BaseTransform):
    def __init__(self, 
                 dist_threshold=30.0, 
                 keep_ratio_bg=0.5,
                 fg_threshold=0.15):
        self.dist_threshold = dist_threshold
        self.keep_ratio_bg = keep_ratio_bg
        self.fg_threshold = fg_threshold

    def transform(self, results):
        points = results['points']
        pts_tensor = points.tensor
        
        # 距离计算
        dist = torch.norm(pts_tensor[:, :2], p=2, dim=1)
        semantic_scores = pts_tensor[:, -4:]
        # 保留前景
        fg_prob_sum = torch.sum(semantic_scores[:, 1:], dim=1)
        is_protected = (fg_prob_sum > self.fg_threshold) | (dist >= self.dist_threshold)
        can_sample_mask = ~is_protected
        
        # 随机采样
        sample_indices = torch.where(can_sample_mask)[0]
        if len(sample_indices) > 0:
            num_keep = int(len(sample_indices) * self.keep_ratio_bg)
            rand_perm = torch.randperm(len(sample_indices))
            keep_sub_indices = sample_indices[rand_perm[:num_keep]]
            
            # 最终索引 = 保护点 + 选中的背景点
            final_indices = torch.cat([torch.where(is_protected)[0], keep_sub_indices])
            results['points'] = points[final_indices]
            
        return results