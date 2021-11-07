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
    # _delete_ =True,
    policy= 'CosineRestart',
    periods=[14,10],
    restart_weights=[1,0.01],
    min_lr_ratio=0.01,
    by_epoch=True, 
    warmup='linear',
    warmup_iters=434*4,
    warmup_ratio=1e-3,
    )
# runtime settings

runner = dict(type='EpochBasedRunner', max_epochs=18)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, classwise=True, save_best='mIoU')
