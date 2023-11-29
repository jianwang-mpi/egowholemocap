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
        dict(type='WandbLoggerHook', init_kwargs=dict(project='hands4whole_test')),
    ])

model = dict(
    type='EgoHandPose',
    pretrained='/HPS/EgoSyn/work/EgocentricHand/Hand4Whole_RELEASE/demo/hand/snapshot_12.pth.tar',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='Generate2DHandPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropHandImageFisheye', fisheye_camera_path=fisheye_camera_path,
         input_img_h=1024, input_img_w=1280,
         crop_img_size=256, enlarge_scale=1.4),
    # dict(type='ResizeImageWithName', img_h=img_res, img_w=img_res, img_name='left_hand_img',
    #      keypoints_name_list=['left_hand_keypoints_2d']),
    # dict(type='ResizeImageWithName', img_h=img_res, img_w=img_res, img_name='right_hand_img',
    #      keypoints_name_list=['right_hand_keypoints_2d']),
    dict(type='RGB2BGRHand'),
    dict(type='ToTensorHand'),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    # dict(type='Generate2DPoseConfidence'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # dict(type='TopDownGenerateTarget', sigma=1.5),
    dict(type='Collect',
         keys=['left_hand_img', 'right_hand_img', 'left_hand_keypoints_3d', 'right_hand_keypoints_3d', ],
         meta_keys=['image_file', 'keypoints_3d', 'left_hand_transform', 'right_hand_transform',
                    ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='Generate2DHandPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropHandImageFisheye', fisheye_camera_path=fisheye_camera_path,
         input_img_h=1024, input_img_w=1280,
         crop_img_size=256, enlarge_scale=1.3),
    # dict(type='ResizeImageWithName', img_h=img_res, img_w=img_res, img_name='left_hand_img',
    #      keypoints_name_list=['left_hand_keypoints_2d']),
    # dict(type='ResizeImageWithName', img_h=img_res, img_w=img_res, img_name='right_hand_img',
    #      keypoints_name_list=['right_hand_keypoints_2d']),
    dict(type='RGB2BGRHand'),
    dict(type='ToTensorHand'),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    # dict(type='Generate2DPoseConfidence'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # dict(type='TopDownGenerateTarget', sigma=1.5),
    dict(type='Collect', keys=['left_hand_img', 'right_hand_img', 'left_hand_transform', 'right_hand_transform', ],
         meta_keys=['image_file', 'keypoints_3d',
                    'left_hand_keypoints_3d', 'right_hand_keypoints_3d',
                    'left_hand_transform', 'right_hand_transform',
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

path_dict = {
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

test_path_dict = {
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
        type='RenderpeopleMixamoDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='MocapStudioHandDataset',
        path_dict=test_path_dict,
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type='MocapStudioHandDataset',
        path_dict=test_path_dict,
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
