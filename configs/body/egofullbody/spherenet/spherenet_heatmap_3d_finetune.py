_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=1e-4,
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    step=1000)

evaluation = dict(
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/spherenet_heatmap_3d_finetune',
    metric='mpjpe',
    save_best='mpjpe',
    rule='less'
)
checkpoint_config = dict(interval=3)

total_epochs = 30
img_res = 256
fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
load_from = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/spherenet_heatmap_3d/epoch_3.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='spherenet_heatmap_3d_finetune', name='spherenet')),
    ])

channel_cfg = dict(
    num_output_channels=15,
    dataset_joints=15,
    dataset_channel=[
        list(range(15)),
    ],
    inference_channel=list(range(15)))

# model settings
model = dict(
    type='Egocentric3DPoseEstimator',
    backbone=dict(
        type='SphereNet',
    ),
    keypoint_head=dict(
        type='Heatmap3DNet',
        in_channels=2048,
        num_deconv_layers=3,
        num_deconv_filters=(1024, 1024, 15 * 64),
        num_deconv_kernels=(4, 4, 4),
        out_channels=15 * 64,
        heatmap_shape=(64, 64, 64),
        fisheye_model_path=fisheye_camera_path, joint_num=15,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
        input_transform='resize_concat',
        in_index=[0, 1, 2, 3],
        ),
    train_cfg=dict(),
    test_cfg=dict()
)

data_cfg = dict(
    num_joints=15,
    camera_param_path=fisheye_camera_path,
    joint_type='mo2cap2',
    image_size=[img_res, img_res],
    heatmap_size=(64, 64),
    joint_weights=[1.] * 15,
    use_different_joint_weights=False,
)

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
    dict(
        type='Collect',
        keys=[
            'img', 'keypoints_3d', 'keypoints_3d_visible',
        ],
        meta_keys=['image_file', 'joints_2d']),
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
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=['image_file', 'keypoints_3d', 'keypoints_3d_visible']),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='MocapStudioFinetuneDataset',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='MocapStudioDataset',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type='MocapStudioDataset',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
