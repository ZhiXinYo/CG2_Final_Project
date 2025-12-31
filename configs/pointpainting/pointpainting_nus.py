_base_ = [
    '../_base_/models/pointpillars_hv_fpn_nus.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py'
]

VERSION = 'v1.0-trainval'

model = dict(
    pts_voxel_encoder=dict(
        in_channels=15,
    )
)

point_cloud_range = [-50, -50, -5, 50, 50, 3]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-50, -50.8, -5, 50, 49.2, 3]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
metainfo = dict(classes=class_names, version=VERSION)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='samples/LIDAR_TOP_PAINTED', img='', sweeps='sweeps/LIDAR_TOP_PAINTED')
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=16,
        use_dim=16,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        load_dim=16,
        use_dim=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        sweeps_num=10,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=16,
        use_dim=16,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        load_dim=16,
        use_dim=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        sweeps_num=10,
        test_mode=True,
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
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=16,
        use_dim=16,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        load_dim=16,
        use_dim=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        sweeps_num=10,
        test_mode=True,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]


train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
    )
)
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_dataloader = test_dataloader

train_cfg = dict(
    max_epochs=24,
    val_interval=8,
)

# optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=4096.)

# load_from = 'checkpoints/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth'