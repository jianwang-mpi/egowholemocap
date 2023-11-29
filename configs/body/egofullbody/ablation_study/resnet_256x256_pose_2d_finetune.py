_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=2e-4,
)

optimizer_config = None

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[40, 55])

evaluation = dict(
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/resnet_256x256_pose_2d_finetune',
    metric='pck',
    save_best='pck'
)
checkpoint_config = dict(interval=2)

total_epochs = 10
img_res = 256
fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
load_from = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/resnet_256x256_pose_2d_train/best_pck_epoch_4.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='resnet_256x256_pose_2d_finetune')),
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
    type='Egocentric2DPoseEstimator',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
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
    dict(type='TopDownGenerateTarget', sigma=1.5),
    dict(
        type='Collect',
        keys=[
            'img', 'target', 'target_weight',
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
        meta_keys=['image_file', 'target', 'joints_2d']),
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
