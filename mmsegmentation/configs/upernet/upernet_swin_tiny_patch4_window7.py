_base_ = [
    '../_base_/models/upernet_swin.py', 
    '../_base_/datasets/custom_dataset.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_20k.py'
]

model = dict(
    pretrained='/opt/ml/mmsegmentation/custom_configs/pretrain/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=11),
    auxiliary_head=dict(in_channels=384, num_classes=11))
