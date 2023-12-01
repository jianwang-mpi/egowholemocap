_base_ = ['/CT/EgoMocap/work/EgocentricFullBody/configs/_base_/default_runtime.py']

optimizer = dict(
    type='Adam',
    lr=5e-4,
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
    res_folder='/CT/EgoMocap/work/EgocentricFullBody/work_dirs/expose_256x256_transl_with_smpl',
    metric='mpjpe',
    save_best='mpjpe',
    rule='less'
)
checkpoint_config = dict(interval=1)

total_epochs = 10
img_res = 256
fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
# load_from = '/CT/EgoMocap/work/EgocentricFullBody/resources/pretrained_models/vitpose_base_coco_aic_mpii.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='expose_256x256_transl_with_smpl')),
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
    type='SMPLXNetWrapper',
    exp_cfg_path='/CT/EgoMocap/work/EgocentricFullBody/configs/body/egofullbody/renderpeople_mixamo/expose/expose_conf_only_body.yaml',
    smplx_loss=dict(
        type='FisheyeMeshLoss',
        joints_2d_loss_weight=0,
        joints_3d_loss_weight=5,
        vertex_loss_weight=0,
        smpl_pose_loss_weight=1,
        smpl_beta_loss_weight=0.001,
        smpl_beta_regularize_loss_weight=0.01,
        camera_param_path=fisheye_camera_path
    ),
    pretrained='/CT/EgoMocap/work/EgocentricFullBody/resources/pretrained_models/expose/checkpoints'
)



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img', 'keypoints_2d', 'keypoints_2d_visible', 'keypoints_3d', 'keypoints_3d_visible',
            'pose', 'beta', 'has_smpl'
        ],
        meta_keys=[]),
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

data_cfg = dict(
    camera_param_path=fisheye_camera_path,
    joint_type='smplx',
    image_size=[img_res, img_res],
)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='EgoMixDataset',
        configs=[
            dict(
                type='RenderpeopleMixamoDataset',
                ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels.pkl',
                img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                test_mode=False,
            ),
            dict(
                type='SyntheticSMPLDataset',
                ann_file='/HPS/EgoSyn/work/synthetic/depth_matterport_seg_seq/smplh_amass_labels.pkl',
                img_prefix='/HPS/EgoSyn/work/synthetic/depth_matterport_seg_seq',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                test_mode=False,
            )
        ],
        partition=[0.5, 0.5],
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
