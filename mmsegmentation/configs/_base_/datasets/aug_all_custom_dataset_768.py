# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/input/data/mmseg/'

# class settings
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
palette = [
    [0, 0, 0],
    [192, 0, 128], [0, 128, 192], [0, 128, 64],
    [128, 0, 0], [64, 0, 128], [64, 0, 192],
    [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]
    ]

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=15,
        interpolation=1,
        border_mode=0,
        p=0.4),
        
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                # brightness_limit=[0.1, 0.3],
                # contrast_limit=[0.1, 0.3],
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.2)
        ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='Blur',
                blur_limit=3,
                p=1.0),
            dict(
                type='CLAHE',
                clip_limit=0.4,
                p=1.0)
        ],
        p=0.2),        
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomGamma',
                p=1.0),
            dict(
                type='GaussNoise',
                p=1.0)
        ],
        p=0.3),                    
]

# img_scale = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiCopyPaste', categories=[3,4,5,9,10], p=[0.1, 0.1, 0.1, 0.2, 0.2],transform=1),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False),
    dict(type='GridMask', cutout_ratio=[(0.4, 0.4)], fill_in=(255,255,255)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Resize', img_scale=(768, 768), keep_ratio=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        # flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True), # 이거 변경해서 돌려봐야겠다.
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False, 
        img_dir=data_root + "all/images/train",
        ann_dir=data_root + "all/annotations/train",
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/valid",
        ann_dir=data_root + "annotations/valid",
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "test",
        pipeline=test_pipeline))

