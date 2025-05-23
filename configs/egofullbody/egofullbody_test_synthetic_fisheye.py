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
    metric=['hand-mpjpe'],
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
        dict(type='WandbLoggerHook', init_kwargs=dict(project='egofullbody_test_synthetic_fisheye_2')),
    ])

model = dict(
    type='EgocentricFullBodyPose',
    body_pose_dict={{_base_.fisheye_body_pose_model_wo_uncert}},
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
    pretrained_body_pose='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/undistort_vit_heatmap_3d/epoch_7.pth',
    pretrained_hand_detection='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_full_body_finetune_only_hand/best_pck_epoch_10.pth',
    pretrained_hand_pose_estimation='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hands4whole_train_synthetic/epoch_1.pth',
    # pretrained_hand_pose_estimation=None,

)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='img_original'),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='Generate2DHandPose', fisheye_model_path=fisheye_camera_path),
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
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='Generate2DHandPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropHandImageFisheye', fisheye_camera_path=fisheye_camera_path,
         input_img_h=1024, input_img_w=1280,
         crop_img_size=256, enlarge_scale=1.3),
    dict(type='RGB2BGRHand'),
    dict(type='ToTensorHand'),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect',
         keys=['img', 'img_original', 'left_hand_img', 'right_hand_img', 'left_hand_transform', 'right_hand_transform'],
         meta_keys=['image_file', 'keypoints_3d', 'left_hand_keypoints_3d', 'right_hand_keypoints_3d', 'ext_pose_gt',
                    'ego_camera_pose', 'ext_id', 'seq_name'
                    ]),
]

test_pipeline = val_pipeline

data_cfg = dict(
    num_joints=73,
    camera_param_path=fisheye_camera_path,
    joint_type='renderpeople',
    image_size=[img_res, img_res],
    hand_image_size=[img_res, img_res],
    heatmap_size=(64, 64),
    joint_weights=None,
    use_different_joint_weights=False,
)


data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='RenderpeopleMixamoDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='RenderpeopleMixamoHandTestDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo_test/renderpeople_mixamo_labels_test.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo_test',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        part_dataset=None,
    ),
    test=dict(
        type='RenderpeopleMixamoHandTestDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo_test_seq/renderpeople_mixamo_labels_test_seq.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo_test_seq',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        part_dataset=None,
    ),
)
