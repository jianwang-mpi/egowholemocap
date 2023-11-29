#  Copyright Jian Wang @ MPI-INF (c) 2023.
_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=2e-4,
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[40, 55])

evaluation = dict(
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/diffusion_full_body_train_uncond_global',
    metric='mpjpe',
    save_best='mpjpe',
    rule='less'
)
checkpoint_config = dict(interval=2)

total_epochs = 20

seq_len = 196
mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/mean_std.pkl'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='diffusion_full_body_train_uncond_global')),
    ])

# model settings
model = dict(
    type='DiffusionEDGEFullBodyGlobal',
    load_model_path=None,
    representation_dim=3 + (15 + 21 + 21) * 3,
    cond_feature_dim=15 * 3,
    seq_len=seq_len,
    guidance_weight=0,
    cond_drop_prob=1,
    return_diffusion=False,
    human_body_joint_loss_weight=3.0
)

train_pipeline = [
    dict(type='EgoFeaturesNormalize', mean_std_path=mean_std_path,
         to_tensor=True,
         normalize_name_list=['left_hand_features',
                              'right_hand_features',
                              'mo2cap2_body_features',
                              'root_features']),
    dict(type='Collect', keys=['left_hand_features',
                               'right_hand_features',
                               'mo2cap2_body_features',
                               'root_features'],
         meta_keys=['global_smplx_joints'])
]

val_pipeline = [
    dict(type='EgoFeaturesNormalize', mean_std_path=mean_std_path,
         to_tensor=True,
         normalize_name_list=['left_hand_features',
                              'right_hand_features',
                              'mo2cap2_body_features',
                              'root_features']),
    dict(type='Collect', keys=['left_hand_features',
                               'right_hand_features',
                               'mo2cap2_body_features',
                               'root_features'],
         meta_keys=['global_smplx_joints'])
]

test_pipeline = val_pipeline
skip_frames = 1


data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='EgoSMPLXFeaturesDataset',
        data_path_list='/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/features_seqs_196.pkl',
        seq_len=seq_len,
        pipeline=train_pipeline,
        split_sequence=True,
        skip_frame=3,
        test_mode=False,
    ),
    val=dict(
        type='EgoSMPLXFeaturesDataset',
        data_path_list='/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/features_seqs_196.pkl',
        seq_len=seq_len,
        pipeline=val_pipeline,
        split_sequence=True,
        skip_frame=seq_len,
        test_mode=True,
    ),
    test=dict(
        type='EgoSMPLXFeaturesDataset',
        data_path_list='/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/features_seqs_196.pkl',
        seq_len=seq_len,
        pipeline=test_pipeline,
        split_sequence=True,
        skip_frame=seq_len,
        test_mode=True,
    ),
)
