_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=1e-3,
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
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/refine_net_fc',
    metric='mpjpe',
    save_best='mpjpe',
    rule='less'
)
checkpoint_config = dict(interval=1)

total_epochs = 10
img_res = 256
fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='refine_net_fc'))
    ])

channel_cfg = dict(
    num_output_channels=15,
    dataset_joints=15,
    dataset_channel=[
        list(range(15)),
    ],
    inference_channel=list(range(15))
)

# model settings
model = dict(
    type='RefineNet',
    pose_backbone=None,
    refinenet=dict(
        type='RefineNetMLP',
        body_feature_network_layers=(15 * (3 + 256 + 2), 512, 15 * 3),
        fisheye_calibration_file=fisheye_camera_path,
        bbox_size=256,
        image_feature_size=256,
        joint_num=15,
        add_joint_noise=True,
        noise_std=0.002
    ),
    posenet_pretrained=None,
    refinenet_pretrained=None,
    freeze_backbone=False,
    loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
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
    dict(type='CopyImage', source='img', target='original_image'),
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
        type='ToTensorWithName',
        img_name='original_image'
    ),
    dict(
        type='NormalizeTensorWithName',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        img_name='original_image'
    ),
    dict(
        type='Collect',
        keys=[
            'img', 'original_image', 'keypoints_3d', 'keypoints_3d_visible',
        ],
        meta_keys=['image_file', 'joints_2d']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='original_image'),
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
        type='ToTensorWithName',
        img_name='original_image'
    ),
    dict(
        type='NormalizeTensorWithName',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        img_name='original_image'
    ),
    dict(
        type='Collect',
        keys=[
            'img', 'original_image', 'keypoints_3d', 'keypoints_3d_visible',
        ],
        meta_keys=['image_file', 'joints_2d']),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='RenderpeopleMixamoDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels_old.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
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
