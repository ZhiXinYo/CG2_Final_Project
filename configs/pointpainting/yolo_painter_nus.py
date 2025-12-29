# 引用 KITTI 和 PointPillars 的基础配置
_base_ = [
    'pointpainting_nus.py'
]

# ========================Frequently modified parameters======================

num_classes = 10  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 2
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 4
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.001
max_epochs = 24  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (512, 288)  # width, height  (512, 288)  (1024, 576)
# Dataset type, this will be used to define the dataset
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# Strides of multi-scale prior box
strides = [8, 16, 32]
# The output channel of the last stage
last_stage_out_channels = 1024
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
tal_topk = 15  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 4  # 0.5, 1, 4
loss_bbox_weight = 7.5  # 7.5, 1.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4  # 1.5 / 4, 1.5 / 20
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals in stage 1
save_epoch_intervals = 10
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 2
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# 允许跨库调用
custom_imports = dict(imports=['mmyolo.models'])
# 在 custom_imports 之后添加
default_scope = 'mmdet3d'

model = dict(
    type='YoloPointPainter',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        # 同步 YOLO 的设置：将 0-255 缩放到 0-1
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True,  # 保持与原本 YOLO 配置一致
        pad_size_divisor=32),  # 确保图像宽高是 32 的倍数，适配 YOLO 的下采样
    img_backbone=dict(
        type='mmyolo.YOLOv8CSPDarknet',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='checkpoints/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth',
        #     prefix='backbone.'  # <--- 关键：告诉它只取权重里以 backbone. 开头的部分
        # ),
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    img_neck=dict(
        type='mmyolo.YOLOv8PAFPN',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='checkpoints/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth',
        #     prefix='neck.'  # <--- 关键：告诉它只取权重里以 backbone. 开头的部分
        # ),
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    img_bbox_head=dict(
        type='mmyolo.YOLOv8Head',
        head_module=dict(
            type='mmyolo.YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=32,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='mmyolo.DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.FocalLoss',  # mmdet.CrossEntropyLoss, mmdet.FocalLoss
            use_sigmoid=True,
            reduction='none',  # none, sum, mean
            loss_weight=loss_cls_weight),  # loss_cls_weight
        loss_bbox=dict(
            type='mmyolo.IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',  # sum, mean
            loss_weight=loss_bbox_weight,  # loss_bbox_weight 7.5
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',  # mean
            loss_weight=loss_dfl_weight),  # loss_dfl_weight 1.5 / 4
        train_cfg=dict(
            assigner=dict(
                type='mmyolo.BatchTaskAlignedAssigner',
                num_classes=num_classes,
                use_ciou=True,  # True
                topk=tal_topk,
                alpha=tal_alpha,
                beta=tal_beta,
                eps=1e-9)),
        test_cfg=dict(
            # The config of multi-label for multi-class prediction.
            multi_label=True,
            # The number of boxes before NMS
            nms_pre=30000,
            score_thr=0.001,  # Threshold to filter out boxes. 0.001
            nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
            max_per_img=300),  # Max number of detections of each image
    ),
)

VERSION = 'v1.0-mini'  # v1.0-mini, v1.0-trainval
# 1. 修改输入模态
input_modality = dict(use_lidar=True, use_camera=True)  # 开启相机

# 2. 修改数据前缀（指向原始数据，而不是预涂色的）
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')
backend_args = None
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
ori_img_shape = (1600, 900)
metainfo = dict(classes=class_names, version=VERSION, ori_img_shape=ori_img_shape)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

train_pipeline = [
    # 加载原始点云 (nuScenes 原始是 5 维: x, y, z, intensity, timestamp)
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    # 加载多帧点云
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=5,
        remove_close=True,
        backend_args=backend_args),

    # --- 【新增】在线涂色必须加载图像 ---
    dict(type='LoadMultiViewImageFromFiles', backend_args=backend_args, num_views=6),
    # 2. 【核心修复】使用 MultiViewWrapper 包装 Resize
    dict(
        type='MultiViewWrapper',
        process_fields=['img', 'cam2img'],  # 确保包含相机内参
        transforms=[
            dict(type='Resize3D', scale=img_scale, keep_ratio=True)
            # 注意：普通的 mmdet.Resize 不会改 cam2img，
            # 除非你自定义一个支持矩阵运算的 Resize 类
        ]
    ),
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

    # --- 【修改】必须把 img 传给模型 ---
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'cam_instances'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,  # 改回原始 nuScenes 维度
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,  # 建议与训练集保持一致 (10帧=1当前+9堆叠)
        use_dim=5,
        test_mode=True,
        backend_args=backend_args),
    # --- 【新增】必须加载图像 ---
    dict(type='LoadMultiViewImageFromFiles', backend_args=backend_args, num_views=6),
    # 2. 【核心修复】使用 MultiViewWrapper 包装 Resize
    dict(
        type='MultiViewWrapper',
        process_fields=['img', 'cam2img'],  # 确保包含相机内参
        transforms=[
            dict(type='Resize3D', scale=img_scale, keep_ratio=True)
            # 注意：普通的 mmdet.Resize 不会改 cam2img，
            # 除非你自定义一个支持矩阵运算的 Resize 类
        ]
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,  # 适配 nuScenes 常用分辨率
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
    # --- 【修改】打包时必须包含 img ---
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=5,
        test_mode=True,
        backend_args=backend_args),
    dict(type='LoadMultiViewImageFromFiles', backend_args=backend_args, num_views=6),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# ================= 核心修改 3: 修改数据加载和评估 =================

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,  # 显式声明类型
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        # 关键修改：通过 data_prefix 指向你涂色后的文件夹
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

val_evaluator = [
    dict(
        type='NuScenesMetric',  # 评估你的 3D 结果
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        metric='bbox',
        backend_args=backend_args),
    dict(
        type='NuscenesCocoMetric',  # 专门评估你的 2D 结果
        ann_file='data/nuscenes/nuscenes_2d_val.json',  # 需准备 COCO 格式的 2D 标注
        metric='bbox',
        classwise=True,
        prefix='pred_instances_2d'),  # 对应你 data_sample 中的 2D 字段名
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=35, norm_type=2)
)

# training schedule for 2x
# ================= 验证频率设置 =================
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=2,    # 设置为 1，表示每个 Epoch 结束后都进行验证
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]

# optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=4096.)

# 分别指定预训练权重
# YOLO 的权重 (建议先加载预训练好的)
# load_from = 'checkpoints/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_101206-b975b1cd.pth'