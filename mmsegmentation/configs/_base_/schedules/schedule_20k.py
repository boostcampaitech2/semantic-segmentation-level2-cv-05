# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=329*4,
    warmup_ratio=0.001,
    step=[12, 18],
    gamma=0.1
    )
# runtime settings

runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, classwise=True, save_best='mIoU')
