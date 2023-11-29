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
        dict(type='WandbLoggerHook', init_kwargs=dict(project='egofullbody_test_b_256_egopw_different_hand_detection')),
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
    pretrained_body_pose='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/vit_256x256_heatmap_3d_finetune/best_mpjpe_epoch_9.pth',
    pretrained_hand_detection='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_full_body_finetune_only_hand/epoch_1.pth',
    pretrained_hand_pose_estimation='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hands4whole_train/best_hand_mpjpe_epoch_9.pth',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='img_original'),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect',
         keys=['img', 'img_original'],
         meta_keys=['image_file', 'keypoints_3d',
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
         meta_keys=['image_file', 'keypoints_3d', 'ext_pose_gt',
                    'ego_camera_pose', 'ext_id', 'seq_name'
                    ]),
]

test_pipeline = val_pipeline

data_cfg = dict(
    num_joints=15,
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
        type='EgoPWFinetuneDataset',
        root_path='/HPS/Mo2Cap2Plus1/static00/ExternalEgo/External_camera_all',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='EgoPWFinetuneDataset',
        root_path='/HPS/Mo2Cap2Plus1/static00/ExternalEgo/External_camera_all',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=True,
    ),
    test=dict(
        type='EgoPWFinetuneDataset',
        root_path='/HPS/Mo2Cap2Plus1/static00/ExternalEgo/External_camera_all',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=True,
    ),
)
