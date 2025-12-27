_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', 
    '../_base_/default_runtime.py'
]

default_scope = 'mmdet3d'

data_root = 'data/kitti_painted/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
metainfo = dict(classes=class_names)
backend_args = None


db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train_painted.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=8,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8, 
        use_dim=8,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=8,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_painted'), # 指向 Painted 文件夹
            pipeline=train_pipeline,
            modality=dict(use_lidar=True, use_camera=False),
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

# 验证和测试 loader
# val_dataloader = dict(
#     dataset=dict(
#         type='KittiDataset',
#         data_root=data_root,
#         data_prefix=dict(pts='training/velodyne_painted'),
#         ann_file='kitti_infos_val.pkl',
#         pipeline=test_pipeline,
#         modality=dict(use_lidar=True, use_camera=False),
#         test_mode=True,
#         metainfo=metainfo,
#         box_type_3d='LiDAR',
#         backend_args=backend_args))
val_dataloader = None
test_dataloader = val_dataloader


model = dict(
    # PointPillars 的 VFE 层需要改为接受 8 通道
    voxel_encoder=dict(in_channels=8)
)

lr = 0.001
epoch_num = 80

optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=999)

val_dataloader = None
val_evaluator = None
val_cfg = None

test_dataloader = None
test_evaluator = None
test_cfg = None

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,        # 每2轮保存
        max_keep_ckpts=10, # 最多保留10个
        save_best=None
    )
)