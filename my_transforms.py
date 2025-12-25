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

import torch
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class MultiBinDistanceSampling(BaseTransform):
    """
    更精细的距离感知采样：
    将 0-70m 分为多个 Bin，每个 Bin 设置不同的保留率。
    目标：极度压缩近处冗余，完全保护远处稀疏目标。
    """
    def __init__(self, 
                 bin_edges=[0, 15, 30, 45, 70], 
                 keep_rates=[0.1, 0.3, 0.7, 1.0]):
        """
        bin_edges: 距离区间的边界
        keep_rates: 每个区间对应的点云保留比例
        """
        self.bin_edges = bin_edges
        self.keep_rates = keep_rates
        assert len(bin_edges) == len(keep_rates) + 1

    def transform(self, results):
        points = results['points']
        # 计算水平距离
        dist = torch.norm(points.tensor[:, :2], p=2, dim=1)
        
        keep_indices = []
        
        for i in range(len(self.keep_rates)):
            # 找到落在当前 Bin 里的点
            mask = (dist >= self.bin_edges[i]) & (dist < self.bin_edges[i+1])
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
                
            rate = self.keep_rates[i]
            if rate >= 1.0:
                # 远处点，全保
                keep_indices.append(indices)
            else:
                # 近处点，随机抽样
                num_keep = int(len(indices) * rate)
                if num_keep > 0:
                    rand_idx = torch.randperm(len(indices))[:num_keep]
                    keep_indices.append(indices[rand_idx])
        
        if len(keep_indices) > 0:
            final_indices = torch.cat(keep_indices)
            results['points'] = points[final_indices]
            
        return results