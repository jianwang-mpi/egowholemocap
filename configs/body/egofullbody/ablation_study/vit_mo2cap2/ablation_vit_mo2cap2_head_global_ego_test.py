_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=1e-4,
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
    metric='pa-mpjpe',
)
checkpoint_config = dict(interval=1)

total_epochs = 25
img_res = 256
fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
load_from = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/ablation_vit_mo2cap2_head/best_mpjpe_epoch_1.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='ablation_vit_mo2cap2_head_finetune')),
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
    type='Mo2Cap2',
    pose_2d_module_cfg=dict(
        type='Egocentric2DPoseEstimator',
        backbone=dict(
            type='ViT',
            img_size=(img_res, img_res),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        train_cfg=dict(),
        test_cfg=dict()
    ),
    pose_2d_module_load_path='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/vit_256x256_pose_2d_finetune/best_pck_epoch_6.pth',
    joint_num=15,
    input_feature_dim=768,
    freeze_pose_2d_module=True,
    distance_loss_cfg=dict(type='MSELoss', loss_weight=1.0, use_target_weight=False),
    fisheye_camera_path=fisheye_camera_path,
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
        type='GlobalEgoTestDataset',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type='GlobalEgoTestDataset',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
