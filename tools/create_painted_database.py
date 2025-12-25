# tools/create_painted_gt_db.py
import pickle
import os
from os import path as osp
import mmengine
import numpy as np
from mmengine import track_iter_progress
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops
from mmdet3d.utils import register_all_modules

def create_painted_groundtruth_database(data_path, info_prefix):
    """
    专门为 PointPainting 生成 8维 的 GT Database
    """
    print(f'Create Painted GT Database for KittiDataset')
    
    # 1. 定义数据集配置
    # 关键修改：指向 velodyne_painted，且 load_dim=8
    dataset_cfg = dict(
        type='KittiDataset',
        data_root=data_path,
        ann_file=f'{info_prefix}_infos_train.pkl', 
        data_prefix=dict(
            pts='training/velodyne_painted_1225', # <--- 指向你生成的数据
            img='training/image_2'
        ),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=8,  # <--- 读取 8 维
                use_dim=8,   # <--- 使用 8 维
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                backend_args=None)
        ]
    )

    # 2. 构建数据集
    dataset = DATASETS.build(dataset_cfg)

    # 3. 定义保存路径
    # 我们保存到一个新的文件夹，避免污染原始的 gt_database
    database_save_path = osp.join(data_path, f'{info_prefix}_gt_database_painted')
    db_info_save_path = osp.join(data_path, f'{info_prefix}_dbinfos_train_painted.pkl')
    
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    group_counter = 0
    
    # 4. 遍历数据集并裁剪
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy() # 这里应该是 (N, 8)
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
            
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        # 获取盒子内的点索引
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            # 文件名示例: 000003_Car_1.bin
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            # 相对路径，供 info pkl 使用
            rel_filepath = osp.join(f'{info_prefix}_gt_database_painted', filename)

            # 提取该物体的点云 (N, 8)
            gt_points = points[point_indices[:, i]]
            
            # 将点云移动到物体局部坐标系 (减去中心坐标)
            # 注意：只操作前3维 (xyz)，后面的维度 (r, c1, c2, c3, c4) 保持不变
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            # 保存为 bin 文件
            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            # 记录 info
            if True: # 保存所有类别
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    # 5. 打印统计信息并保存 pkl
    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

if __name__ == '__main__':
    register_all_modules()

    # 这里的参数要和你的 data 目录一致
    create_painted_groundtruth_database(
        data_path='./data/kitti_painted',
        info_prefix='kitti_1225'
    )