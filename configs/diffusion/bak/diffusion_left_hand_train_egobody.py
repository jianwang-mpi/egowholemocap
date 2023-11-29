#  Copyright Jian Wang @ MPI-INF (c) 2023.
_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=1e-4,
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[40, 55])

evaluation = dict(
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/diffusion_left_hand_train_egobody',
    metric='mpjpe',
    save_best='mpjpe',
    rule='less'
)
checkpoint_config = dict(interval=1)

total_epochs = 10

seq_len = 196
mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/global_aligned_mean_std.pkl'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='diffusion_left_hand_train_egobody', name='left_hand_train')),
    ])

# model settings
model = dict(
    type='DiffusionHands',
    right_hand=False,
    load_model_path=None,
    representation_dim=21 * 3,
    cond_feature_dim=15 * 3,
    seq_len=seq_len
)

train_pipeline = [
    dict(type='AlignGlobalSMPLXJoints', align_every_joint=True),
    dict(type='SplitGlobalSMPLXJoints'),
    dict(type='PreProcessHandMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='PreProcessMo2Cap2BodyMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='Collect', keys=['left_hand_features', 'right_hand_features', 'mo2cap2_body_features'],
         meta_keys=['json_path_calib', 'aligned_smplx_joints'])
]

val_pipeline = [
    dict(type='AlignGlobalSMPLXJoints', align_every_joint=True),
    dict(type='SplitGlobalSMPLXJoints'),
    dict(type='PreProcessHandMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='PreProcessMo2Cap2BodyMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='Collect', keys=['left_hand_features', 'right_hand_features', 'mo2cap2_body_features'],
         meta_keys=['json_path_calib', 'aligned_smplx_joints'])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=12,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='EgoBodyDataset',
        data_path='/CT/datasets04/static00/EgoBody',
        seq_len=seq_len,
        skip_frames=50,
        smplx_model_dir='/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new',
        pipeline=train_pipeline,
        split_sequence=True,
        data_dirs=['smplx_interactee_test', 'smplx_interactee_train', 'smplx_interactee_val',
                   'smplx_camera_wearer_test', 'smplx_camera_wearer_train'],
        # data_dirs=['smplx_interactee_val'],
        test_mode=False
    ),
    val=dict(
        type='EgoBodyDataset',
        data_path='/CT/datasets04/static00/EgoBody',
        seq_len=seq_len,
        skip_frames=seq_len,
        smplx_model_dir='/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new',
        pipeline=val_pipeline,
        split_sequence=True,
        data_dirs=['smplx_camera_wearer_val', ],
        test_mode=True
    ),
    test=dict(
        type='EgoBodyDataset',
        data_path='/CT/datasets04/static00/EgoBody',
        seq_len=seq_len,
        skip_frames=seq_len,
        smplx_model_dir='/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new',
        pipeline=test_pipeline,
        split_sequence=True,
        data_dirs=['smplx_camera_wearer_val',],
        test_mode=True
    ),
)
