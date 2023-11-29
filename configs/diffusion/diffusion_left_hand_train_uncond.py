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
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/diffusion_full_body_train_uncond_eb_stu_rp',
    metric='mpjpe',
    save_best='mpjpe',
    rule='less'
)
checkpoint_config = dict(interval=2)

total_epochs = 20

seq_len = 196
mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/global_aligned_mean_std.pkl'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='diffusion_full_body_train_uncond_eb_stu_rp')),
    ])

# model settings
model = dict(
    type='DiffusionEDGEHand',
    load_model_path=None,
    representation_dim=20 * 3,
    cond_feature_dim=15 * 3,
    is_left_hand=True,
    seq_len=seq_len,
    guidance_weight=0,
    cond_drop_prob=1,
    return_diffusion=False,
    human_body_joint_loss_weight=3.0
)

egobody_train_pipeline = [
    dict(type='SplitGlobalSMPLXJoints'),
    dict(type='PreProcessHandMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='PreProcessMo2Cap2BodyMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='Collect', keys=['left_hand_features', 'right_hand_features', 'mo2cap2_body_features'],
         meta_keys=['aligned_smplx_joints'])
]

renderpeople_pipeline = [
    dict(type='AlignGlobalSMPLXJoints', align_every_joint=True),
    dict(type='SplitGlobalSMPLXJoints', smplx_joint_name='aligned_smplx_joints'),
    dict(type='PreProcessHandMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='PreProcessMo2Cap2BodyMotion', normalize=True, mean_std_path=mean_std_path),
    dict(type='Collect', keys=['mo2cap2_body_features', 'left_hand_features', 'right_hand_features'],
         meta_keys=['aligned_smplx_joints'])
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
skip_frames = 1
egobody_dataset_config = dict(
    type='EgoSMPLXDataset',
    data_path_list=[{
        'path': '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/global_aligned_smplx_inputs.pkl',
        'frame_rate': 30
    }],
    seq_len=seq_len,
    pipeline=egobody_train_pipeline,
    split_sequence=True,
    target_frame_rate=25,
    skip_frame=skip_frames,
    test_mode=False
)
renderpeople_dataset_config = dict(
    type='RenderpeopleMotionDataset',
    data_path=r'/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
    seq_len=seq_len,
    skip_frames=skip_frames,
    pipeline=renderpeople_pipeline,
    split_sequence=True,
    human_names=['render_people_adanna_joints_all'],
    test_mode=True
)
studio_motion_dataset_config = dict(
    type='MocapStudioMotionDataset',
    seq_len=seq_len,
    skip_frames=skip_frames,
    pipeline=renderpeople_pipeline,
    split_sequence=True,
    test_mode=False,
    local=False,
    data_cfg={
    }
)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='ConcatDataset',
        datasets=[egobody_dataset_config, renderpeople_dataset_config, studio_motion_dataset_config],
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
        data_dirs=['smplx_camera_wearer_val', ],
        test_mode=True
    ),
)
