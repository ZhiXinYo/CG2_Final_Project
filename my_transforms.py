# # my_transforms.py
# import torch
# import numpy as np
# from mmcv.transforms import BaseTransform
# from mmdet3d.registry import TRANSFORMS

# @TRANSFORMS.register_module()
# class RandomDropPaintedFeatures(BaseTransform):
#     """
#     随机丢弃 Painted 的语义特征。
#     """
#     def __init__(self, drop_prob=0.2, painted_dims=4):
#         self.drop_prob = drop_prob
#         self.painted_dims = painted_dims

#     def transform(self, results):
#         if np.random.rand() < self.drop_prob:
#             results['points'].tensor[:, -self.painted_dims:] = 0.0
#         return results

# import torch
# import numpy as np
# from mmcv.transforms import BaseTransform
# from mmdet3d.registry import TRANSFORMS

# @TRANSFORMS.register_module()
# class MultiBinDistanceSampling(BaseTransform):
#     """
#     更精细的距离感知采样：
#     将 0-70m 分为多个 Bin，每个 Bin 设置不同的保留率。
#     目标：极度压缩近处冗余，完全保护远处稀疏目标。
#     """
#     def __init__(self, 
#                  bin_edges=[0, 15, 30, 45, 70], 
#                  keep_rates=[0.1, 0.3, 0.7, 1.0]):
#         """
#         bin_edges: 距离区间的边界
#         keep_rates: 每个区间对应的点云保留比例
#         """
#         self.bin_edges = bin_edges
#         self.keep_rates = keep_rates
#         assert len(bin_edges) == len(keep_rates) + 1

#     def transform(self, results):
#         points = results['points']
#         # 计算水平距离
#         dist = torch.norm(points.tensor[:, :2], p=2, dim=1)
        
#         keep_indices = []
        
#         for i in range(len(self.keep_rates)):
#             # 找到落在当前 Bin 里的点
#             mask = (dist >= self.bin_edges[i]) & (dist < self.bin_edges[i+1])
#             indices = torch.where(mask)[0]
            
#             if len(indices) == 0:
#                 continue
                
#             rate = self.keep_rates[i]
#             if rate >= 1.0:
#                 # 远处点，全保
#                 keep_indices.append(indices)
#             else:
#                 # 近处点，随机抽样
#                 num_keep = int(len(indices) * rate)
#                 if num_keep > 0:
#                     rand_idx = torch.randperm(len(indices))[:num_keep]
#                     keep_indices.append(indices[rand_idx])
        
#         if len(keep_indices) > 0:
#             final_indices = torch.cat(keep_indices)
#             results['points'] = points[final_indices]
            
#         return results


import torch
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SemanticAwareDistanceSampling(BaseTransform):
    """
    改进版 StVD：
    1. 保护语义：如果图像分割分数显示该点是 Pedestrian/Cyclist/Car，则 100% 保留。
    2. 距离感知：只对近处的、被识别为“背景”的点进行随机采样。
    3. 保护远处：远处的所有点（无论语义）全部保留。
    """
    def __init__(self, 
                 dist_threshold=25.0, 
                 keep_ratio_bg=0.3, # 稍微提高保留率，不要丢太狠
                 feat_channels=4): # 语义分数的通道数
        self.dist_threshold = dist_threshold
        self.keep_ratio_bg = keep_ratio_bg
        self.feat_channels = feat_channels

    def transform(self, results):
        points = results['points']
        pts_tensor = points.tensor
        
        # 1. 计算距离
        dist = torch.norm(pts_tensor[:, :2], p=2, dim=1)
        
        # 2. 识别“前景”点
        # 假设点云最后 4 维是 [Background, Pedestrian, Cyclist, Car]
        # 我们看后 3 维中是否有任何一个分数超过了背景分数，或者设置一个阈值（如 > 0.1）
        semantic_scores = pts_tensor[:, -self.feat_channels:]
        foreground_scores = semantic_scores[:, 1:] # 取出 Ped, Cyc, Car 的分数
        
        # 如果任何一个前景分数的最大值 > 背景分数，我们认为它是“潜在物体”
        # 或者更简单的：如果背景分数不是最高的，它就是前景
        is_foreground = torch.argmax(semantic_scores, dim=1) > 0
        
        # 3. 确定哪些点可以被丢弃
        # 条件：(在近处) AND (不是前景)
        can_drop_mask = (dist < self.dist_threshold) & (~is_foreground)
        
        # 4. 执行随机采样
        keep_mask = torch.ones(len(pts_tensor), dtype=torch.bool, device=pts_tensor.device)
        
        drop_candidate_indices = torch.where(can_drop_mask)[0]
        num_drop_candidates = len(drop_candidate_indices)
        
        if num_drop_candidates > 0:
            # 随机选择要丢弃的索引
            num_keep = int(num_drop_candidates * self.keep_ratio_bg)
            rand_perm = torch.randperm(num_drop_candidates)
            
            # 标记那些“没被选中保留”的背景点为 False
            drop_indices = drop_candidate_indices[rand_perm[num_keep:]]
            keep_mask[drop_indices] = False
            
        # 5. 更新结果
        results['points'] = points[keep_mask]
            
        return results