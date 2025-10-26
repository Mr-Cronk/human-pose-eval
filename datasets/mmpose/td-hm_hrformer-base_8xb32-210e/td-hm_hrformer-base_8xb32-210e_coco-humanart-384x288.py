_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={'relative_position_bias_table': dict(decay_mult=0.)}))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRFormer',
        in_channels=3,
        norm_cfg=norm_cfg,
        extra=dict(
            drop_path_rate=0.2,
            with_rpe=True,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, ),
                num_heads=[2],
                mlp_ratios=[4]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2),
                num_channels=(78, 156),
                num_heads=[2, 4],
                mlp_ratios=[4, 4],
                window_sizes=[7, 7]),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2, 2),
                num_channels=(78, 156, 312),
                num_heads=[2, 4, 8],
                mlp_ratios=[4, 4, 4],
                window_sizes=[7, 7, 7]),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2, 2, 2),
                num_channels=(78, 156, 312, 624),
                num_heads=[2, 4, 8, 16],
                mlp_ratios=[4, 4, 4, 4],
                window_sizes=[7, 7, 7, 7])),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrformer_base-32815020_20220226.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=78,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

exlpose_to_coco_converter = dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
                (5, 10),
                (6, 11),
                (7, 12),
                (8, 13),
                (9, 14),
                (10, 15),
                (11, 16),
            ])

exlpose_val = dict(
    type='ExlposeDataset',
    data_root='data/ExLPose/',
    data_mode='topdown',
    ann_file='annotations/ExLPose/ExLPose_test_LL-A.json',
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=[],
)

coco_val = dict(
    type='CocoDataset',
    data_root='data/coco/',
    data_mode='topdown',
    ann_file='annotations/person_keypoints_val2017.json',
    bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=[],
)

val_dataset = dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[coco_val, exlpose_val],
        pipeline=val_pipeline,
        test_mode=True,
)

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    metrics=[
        dict(type='CocoMetric',
             ann_file='data/coco/annotations/person_keypoints_val2017.json'),
        dict(type='CocoMetric',
            ann_file='data/ExLPose/annotations/ExLPose/ExLPose_test_LL-A.json',
            use_area=False,
            gt_converter=exlpose_to_coco_converter,
            prefix='exlpose')
    ],
    datasets=[coco_val, exlpose_val],
    )
test_evaluator = val_evaluator

# fp16 settings
fp16 = dict(loss_scale='dynamic')
