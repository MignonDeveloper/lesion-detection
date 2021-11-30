# ! config
work_dir = './work_dirs/baseline_deformable-detr-aug'
seed = 777
gpus = 4
gpu_ids = range(4)

# ! Dataset
dataset_type = 'CocoDataset'
data_root = '/home/minyong/bb-detection/data/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ! classes
classes = ('01_ulcer', '02_mass', '04_lymph', '05_bleeding')




# * ===============================================



# ! model setting
model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=2,
        num_classes=4,
        in_channels=2048,
        sync_cls_avg_factor=True,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))),
    test_cfg=dict(max_per_img=2))




# ! Pipeline
albu_train_transforms = [
    # dict(
    #     type="OneOf",
    #     transforms=[
    #         dict(type="HorizontalFlip", p=0.5),
    #         dict(type="VerticalFlip", p=0.5),
    #         dict(type="Flip", p=0.5),
    #     ],
    #     p=0.5),

    dict(
        type='OneOf',
        transforms=[
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.9),
            dict(
                type="HueSaturationValue",
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.9),
        ],
        p=0.5),

    dict(
        type="Cutout",
        num_holes=8,
        max_h_size=16,
        max_w_size=16,
        fill_value=0,
        p=0.5),
]




train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(576, 576), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(576,576),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(576,576),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]




data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'train_fold4.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'valid_fold4.json',
        img_prefix=data_root + 'train/',
        pipeline=val_pipeline,
        samples_per_gpu=8,),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'test_annotations.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))


# ! training schedules (optimizer, scheduler, learning rate)
# * optimizer
# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4 / (8 // gpus),
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# * learning rate with scheduler
lr_config = dict(policy='step', step=[20])

# * epochs 
runner = dict(type='EpochBasedRunner', max_epochs=40)
evaluation = dict(interval=1, metric="bbox")

# * model save and evaluation
checkpoint_config = dict(interval=1)

# * The logger used to record the training process.
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth"
workflow = [('train', 1)]
