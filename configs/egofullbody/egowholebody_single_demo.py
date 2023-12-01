#  Copyright Jian Wang @ MPI-INF (c) 2023.

_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py',
          'modules.py']

optimizer = dict(
    type='Adam',
    lr=1e-4,
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[40, 55])

evaluation = dict(
    # res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_256x256_renderpeople_mixamo_full_body_2d',
    metric='hand-mpjpe',
    # save_best='pck',
    # rule='greater'
)
checkpoint_config = dict(interval=1)

total_epochs = 10
img_res = 256

fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='egowholebody_single_demo')),
    ])

model = dict(
    type='EgocentricFullBodyPose',
    body_pose_dict={{_base_.body_pose_model}},
    hand_detection_dict={{_base_.hand_detection_model}},
    hand_pose_estimation_dict={{_base_.hand_pose_model}},
    hand_process_pipeline=[dict(type='CropHandImageFisheye', fisheye_camera_path=fisheye_camera_path,
                                input_img_h=1024, input_img_w=1280,
                                crop_img_size=256, enlarge_scale=1.3),
                           dict(type='RGB2BGRHand'),
                           dict(type='ToTensorHand'), ],
    train_cfg={},
    test_cfg={
        'fisheye_camera_path': fisheye_camera_path,
    },
    pretrained_body_pose='../../demo/resources/fisheyevit.pth',
    pretrained_hand_detection='../../demo/resources/hand_detection.pth',
    pretrained_hand_pose_estimation='../../demo/resources/hand_pose_estimation.pth',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='img_original'),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect',
         keys=['img', 'img_original'],
         meta_keys=['image_file',
                    ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='img_original'),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect',
         keys=['img', 'img_original'],
         meta_keys=['image_file',]),
]

test_pipeline = val_pipeline

data_cfg = dict(
    num_joints=15,
    img_dir='../../demo/resources/imgs',
    camera_param_path=fisheye_camera_path,
    joint_type='mo2cap2',
    image_size=[img_res, img_res],
    heatmap_size=(64, 64),
    joint_weights=[1.] * 15,
    use_different_joint_weights=False,
)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=256),
    test_dataloader=dict(samples_per_gpu=256),
    train=dict(
        type='EvalDataset',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='EvalDataset',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    test=dict(
        type='EvalDataset',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
)
