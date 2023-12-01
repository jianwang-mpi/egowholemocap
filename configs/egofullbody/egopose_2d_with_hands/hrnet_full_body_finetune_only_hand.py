_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=2e-5,
)

optimizer_config = None

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[40, 55])

evaluation = dict(
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_full_body_finetune_only_hand',
    metric='pck',
    save_best='pck',
    rule='greater'
)
checkpoint_config = dict(interval=1)

total_epochs = 10
img_res = 256
joint_num = 42
fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
load_from = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_full_body_train_only_hand/best_pck_epoch_3.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='hrnet_full_body_finetune_only_hand')),
    ]
)

channel_cfg = dict(
    num_output_channels=joint_num,
    dataset_joints=joint_num,
    dataset_channel=[
        list(range(joint_num)),
    ],
    inference_channel=list(range(joint_num))
)

# model settings
model = dict(
    type='Egocentric2DPoseEstimator',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict())

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='Generate2DPoseConfidence'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=1.5),
    dict(
        type='Collect',
        keys=[
            'img', 'target', 'target_weight'
        ],
        meta_keys=['image_file', 'keypoints_3d', 'joints_3d', 'joints_3d_visible', 'joints_2d']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='Generate2DPoseConfidence'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=1.5),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=['image_file', 'target', 'joints_2d', 'joints_3d', 'joints_3d_visible', 'keypoints_2d_visible']),
]

test_pipeline = val_pipeline

data_cfg_train = dict(
    num_joints=joint_num,
    camera_param_path=fisheye_camera_path,
    joint_type='renderpeople',
    only_hand=True,
    image_size=[img_res, img_res],
    heatmap_size=(64, 64),
    joint_weights=[1.] * joint_num,
    use_different_joint_weights=False,
)

data_cfg_test = dict(
    num_joints=joint_num,
    camera_param_path=fisheye_camera_path,
    joint_type='renderpeople',
    only_hand=True,
    image_size=[img_res, img_res],
    heatmap_size=(64, 64),
    joint_weights=[1.] * joint_num,
    use_different_joint_weights=False,
)

finetune_path_dict = {
    'jian1': {
        'path': r'/HPS/ScanNet/work/egocentric_view/05082022/jian1',
    },
    'jian2': {
        'path': r'/HPS/ScanNet/work/egocentric_view/05082022/jian2',
    },
    'diogo1': {
        'path': r'/HPS/ScanNet/work/egocentric_view/05082022/diogo1',
    },
    'diogo2': {
        'path': r'/HPS/ScanNet/work/egocentric_view/05082022/diogo2',
    },
    'pranay2': {
        'path': r'/HPS/ScanNet/work/egocentric_view/05082022/pranay2',
    },
}

path_dict = {
    'new_jian1': {
        'path': r'/HPS/ScanNet/work/egocentric_view/25082022/jian1',
    },
    'new_jian2': {
        'path': r'/HPS/ScanNet/work/egocentric_view/25082022/jian2',
    },
    'new_diogo1': {
        'path': r'/HPS/ScanNet/work/egocentric_view/25082022/diogo1',
    },
    'new_diogo2': {
        'path': r'/HPS/ScanNet/work/egocentric_view/25082022/diogo2',
    },
}

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='MocapStudioHandDataset',
        path_dict=finetune_path_dict,
        data_cfg=data_cfg_train,
        pipeline=train_pipeline,
        test_mode=True,
    ),
    val=dict(
        type='MocapStudioHandDataset',
        path_dict=path_dict,
        data_cfg=data_cfg_test,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type='MocapStudioHandDataset',
        path_dict=path_dict,
        data_cfg=data_cfg_test,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
