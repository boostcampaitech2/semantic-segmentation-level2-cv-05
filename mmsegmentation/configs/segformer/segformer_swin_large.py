_base_ = [
    '_base_/schedule/schedule_cosine.py',
    '_base_/datasets/all_custom_dataset.py',
    '_base_/default_runtime.py',
]

# wandb name
wandb_name = "_[Final]_segformer_swin_large"
seed = 1026


norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/opt/ml/segmentation/mmsegmentation/swin_large_patch4_window12_384_22k_seg.pth'
        ),
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        channels=768,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, )),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1536,
        in_index=3,
        channels=256, 
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


data = dict(
    samples_per_gpu = 6
)
optimizer = dict(
    _delete_ = True,
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))

lr_config = dict(
    warmup_iters=625*4)
