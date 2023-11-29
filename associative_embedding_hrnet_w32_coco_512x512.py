checkpoint_config = dict(interval=50)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author=
        'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/'),
    keypoint_info=dict({
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    }),
    skeleton_info=dict({
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    }),
    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2,
        1.2, 1.5, 1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
evaluation = dict(interval=50, metric='mAP', save_best='AP')
optimizer = dict(type='Adam', lr=0.0015)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 260])
total_epochs = 300
channel_cfg = dict(
    dataset_joints=17,
    dataset_channel=[[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128],
    num_joints=17,
    dataset_channel=[[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ],
    num_scales=1,
    scale_aware_sigma=False)
model = dict(
    type='AssociativeEmbedding',
    pretrained=
    'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
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
                num_channels=(32, 64, 128, 256)))),
    keypoint_head=dict(
        type='AESimpleHead',
        in_channels=32,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=True,
        with_ae_loss=[True],
        extra=dict(final_conv_kernel=1),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=17,
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True],
        with_ae=[True],
        project2image=True,
        align_corners=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='BottomUpGenerateTarget', sigma=2, max_num_people=30),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ])
]
data_root = 'data/coco'
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=24),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_train2017.json',
        img_prefix='data/coco/train2017/',
        data_cfg=dict(
            image_size=512,
            base_size=256,
            base_sigma=2,
            heatmap_size=[128],
            num_joints=17,
            dataset_channel=[[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ]],
            inference_channel=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ],
            num_scales=1,
            scale_aware_sigma=False),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='BottomUpRandomAffine',
                rot_factor=30,
                scale_factor=[0.75, 1.5],
                scale_type='short',
                trans_factor=40),
            dict(type='BottomUpRandomFlip', flip_prob=0.5),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='BottomUpGenerateTarget', sigma=2, max_num_people=30),
            dict(
                type='Collect',
                keys=['img', 'joints', 'targets', 'masks'],
                meta_keys=[])
        ],
        dataset_info=dict(
            dataset_name='coco',
            paper_info=dict(
                author=
                'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(
                    name='nose',
                    id=0,
                    color=[51, 153, 255],
                    type='upper',
                    swap=''),
                1:
                dict(
                    name='left_eye',
                    id=1,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_eye'),
                2:
                dict(
                    name='right_eye',
                    id=2,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_eye'),
                3:
                dict(
                    name='left_ear',
                    id=3,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_ear'),
                4:
                dict(
                    name='right_ear',
                    id=4,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_ear'),
                5:
                dict(
                    name='left_shoulder',
                    id=5,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_shoulder'),
                6:
                dict(
                    name='right_shoulder',
                    id=6,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_shoulder'),
                7:
                dict(
                    name='left_elbow',
                    id=7,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_elbow'),
                8:
                dict(
                    name='right_elbow',
                    id=8,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_elbow'),
                9:
                dict(
                    name='left_wrist',
                    id=9,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_wrist'),
                10:
                dict(
                    name='right_wrist',
                    id=10,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_wrist'),
                11:
                dict(
                    name='left_hip',
                    id=11,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_hip'),
                12:
                dict(
                    name='right_hip',
                    id=12,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_hip'),
                13:
                dict(
                    name='left_knee',
                    id=13,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_knee'),
                14:
                dict(
                    name='right_knee',
                    id=14,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_knee'),
                15:
                dict(
                    name='left_ankle',
                    id=15,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_ankle'),
                16:
                dict(
                    name='right_ankle',
                    id=16,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_ankle')
            }),
            skeleton_info=dict({
                0:
                dict(
                    link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
                2:
                dict(
                    link=('right_ankle', 'right_knee'),
                    id=2,
                    color=[255, 128, 0]),
                3:
                dict(
                    link=('right_knee', 'right_hip'),
                    id=3,
                    color=[255, 128, 0]),
                4:
                dict(
                    link=('left_hip', 'right_hip'), id=4, color=[51, 153,
                                                                 255]),
                5:
                dict(
                    link=('left_shoulder', 'left_hip'),
                    id=5,
                    color=[51, 153, 255]),
                6:
                dict(
                    link=('right_shoulder', 'right_hip'),
                    id=6,
                    color=[51, 153, 255]),
                7:
                dict(
                    link=('left_shoulder', 'right_shoulder'),
                    id=7,
                    color=[51, 153, 255]),
                8:
                dict(
                    link=('left_shoulder', 'left_elbow'),
                    id=8,
                    color=[0, 255, 0]),
                9:
                dict(
                    link=('right_shoulder', 'right_elbow'),
                    id=9,
                    color=[255, 128, 0]),
                10:
                dict(
                    link=('left_elbow', 'left_wrist'),
                    id=10,
                    color=[0, 255, 0]),
                11:
                dict(
                    link=('right_elbow', 'right_wrist'),
                    id=11,
                    color=[255, 128, 0]),
                12:
                dict(
                    link=('left_eye', 'right_eye'),
                    id=12,
                    color=[51, 153, 255]),
                13:
                dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
                14:
                dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
                15:
                dict(
                    link=('left_eye', 'left_ear'), id=15, color=[51, 153,
                                                                 255]),
                16:
                dict(
                    link=('right_eye', 'right_ear'),
                    id=16,
                    color=[51, 153, 255]),
                17:
                dict(
                    link=('left_ear', 'left_shoulder'),
                    id=17,
                    color=[51, 153, 255]),
                18:
                dict(
                    link=('right_ear', 'right_shoulder'),
                    id=18,
                    color=[51, 153, 255])
            }),
            joint_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0,
                1.0, 1.2, 1.2, 1.5, 1.5
            ],
            sigmas=[
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
            ])),
    val=dict(
        type='BottomUpCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=dict(
            image_size=512,
            base_size=256,
            base_sigma=2,
            heatmap_size=[128],
            num_joints=17,
            dataset_channel=[[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ]],
            inference_channel=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ],
            num_scales=1,
            scale_aware_sigma=False),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
            dict(
                type='BottomUpResizeAlign',
                transforms=[
                    dict(type='ToTensor'),
                    dict(
                        type='NormalizeTensor',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                ]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'image_file', 'aug_data', 'test_scale_factor', 'base_size',
                    'center', 'scale', 'flip_index'
                ])
        ],
        dataset_info=dict(
            dataset_name='coco',
            paper_info=dict(
                author=
                'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(
                    name='nose',
                    id=0,
                    color=[51, 153, 255],
                    type='upper',
                    swap=''),
                1:
                dict(
                    name='left_eye',
                    id=1,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_eye'),
                2:
                dict(
                    name='right_eye',
                    id=2,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_eye'),
                3:
                dict(
                    name='left_ear',
                    id=3,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_ear'),
                4:
                dict(
                    name='right_ear',
                    id=4,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_ear'),
                5:
                dict(
                    name='left_shoulder',
                    id=5,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_shoulder'),
                6:
                dict(
                    name='right_shoulder',
                    id=6,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_shoulder'),
                7:
                dict(
                    name='left_elbow',
                    id=7,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_elbow'),
                8:
                dict(
                    name='right_elbow',
                    id=8,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_elbow'),
                9:
                dict(
                    name='left_wrist',
                    id=9,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_wrist'),
                10:
                dict(
                    name='right_wrist',
                    id=10,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_wrist'),
                11:
                dict(
                    name='left_hip',
                    id=11,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_hip'),
                12:
                dict(
                    name='right_hip',
                    id=12,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_hip'),
                13:
                dict(
                    name='left_knee',
                    id=13,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_knee'),
                14:
                dict(
                    name='right_knee',
                    id=14,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_knee'),
                15:
                dict(
                    name='left_ankle',
                    id=15,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_ankle'),
                16:
                dict(
                    name='right_ankle',
                    id=16,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_ankle')
            }),
            skeleton_info=dict({
                0:
                dict(
                    link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
                2:
                dict(
                    link=('right_ankle', 'right_knee'),
                    id=2,
                    color=[255, 128, 0]),
                3:
                dict(
                    link=('right_knee', 'right_hip'),
                    id=3,
                    color=[255, 128, 0]),
                4:
                dict(
                    link=('left_hip', 'right_hip'), id=4, color=[51, 153,
                                                                 255]),
                5:
                dict(
                    link=('left_shoulder', 'left_hip'),
                    id=5,
                    color=[51, 153, 255]),
                6:
                dict(
                    link=('right_shoulder', 'right_hip'),
                    id=6,
                    color=[51, 153, 255]),
                7:
                dict(
                    link=('left_shoulder', 'right_shoulder'),
                    id=7,
                    color=[51, 153, 255]),
                8:
                dict(
                    link=('left_shoulder', 'left_elbow'),
                    id=8,
                    color=[0, 255, 0]),
                9:
                dict(
                    link=('right_shoulder', 'right_elbow'),
                    id=9,
                    color=[255, 128, 0]),
                10:
                dict(
                    link=('left_elbow', 'left_wrist'),
                    id=10,
                    color=[0, 255, 0]),
                11:
                dict(
                    link=('right_elbow', 'right_wrist'),
                    id=11,
                    color=[255, 128, 0]),
                12:
                dict(
                    link=('left_eye', 'right_eye'),
                    id=12,
                    color=[51, 153, 255]),
                13:
                dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
                14:
                dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
                15:
                dict(
                    link=('left_eye', 'left_ear'), id=15, color=[51, 153,
                                                                 255]),
                16:
                dict(
                    link=('right_eye', 'right_ear'),
                    id=16,
                    color=[51, 153, 255]),
                17:
                dict(
                    link=('left_ear', 'left_shoulder'),
                    id=17,
                    color=[51, 153, 255]),
                18:
                dict(
                    link=('right_ear', 'right_shoulder'),
                    id=18,
                    color=[51, 153, 255])
            }),
            joint_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0,
                1.0, 1.2, 1.2, 1.5, 1.5
            ],
            sigmas=[
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
            ])),
    test=dict(
        type='BottomUpCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=dict(
            image_size=512,
            base_size=256,
            base_sigma=2,
            heatmap_size=[128],
            num_joints=17,
            dataset_channel=[[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ]],
            inference_channel=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ],
            num_scales=1,
            scale_aware_sigma=False),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
            dict(
                type='BottomUpResizeAlign',
                transforms=[
                    dict(type='ToTensor'),
                    dict(
                        type='NormalizeTensor',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                ]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'image_file', 'aug_data', 'test_scale_factor', 'base_size',
                    'center', 'scale', 'flip_index'
                ])
        ],
        dataset_info=dict(
            dataset_name='coco',
            paper_info=dict(
                author=
                'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(
                    name='nose',
                    id=0,
                    color=[51, 153, 255],
                    type='upper',
                    swap=''),
                1:
                dict(
                    name='left_eye',
                    id=1,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_eye'),
                2:
                dict(
                    name='right_eye',
                    id=2,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_eye'),
                3:
                dict(
                    name='left_ear',
                    id=3,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_ear'),
                4:
                dict(
                    name='right_ear',
                    id=4,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_ear'),
                5:
                dict(
                    name='left_shoulder',
                    id=5,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_shoulder'),
                6:
                dict(
                    name='right_shoulder',
                    id=6,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_shoulder'),
                7:
                dict(
                    name='left_elbow',
                    id=7,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_elbow'),
                8:
                dict(
                    name='right_elbow',
                    id=8,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_elbow'),
                9:
                dict(
                    name='left_wrist',
                    id=9,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_wrist'),
                10:
                dict(
                    name='right_wrist',
                    id=10,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_wrist'),
                11:
                dict(
                    name='left_hip',
                    id=11,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_hip'),
                12:
                dict(
                    name='right_hip',
                    id=12,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_hip'),
                13:
                dict(
                    name='left_knee',
                    id=13,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_knee'),
                14:
                dict(
                    name='right_knee',
                    id=14,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_knee'),
                15:
                dict(
                    name='left_ankle',
                    id=15,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_ankle'),
                16:
                dict(
                    name='right_ankle',
                    id=16,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_ankle')
            }),
            skeleton_info=dict({
                0:
                dict(
                    link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
                2:
                dict(
                    link=('right_ankle', 'right_knee'),
                    id=2,
                    color=[255, 128, 0]),
                3:
                dict(
                    link=('right_knee', 'right_hip'),
                    id=3,
                    color=[255, 128, 0]),
                4:
                dict(
                    link=('left_hip', 'right_hip'), id=4, color=[51, 153,
                                                                 255]),
                5:
                dict(
                    link=('left_shoulder', 'left_hip'),
                    id=5,
                    color=[51, 153, 255]),
                6:
                dict(
                    link=('right_shoulder', 'right_hip'),
                    id=6,
                    color=[51, 153, 255]),
                7:
                dict(
                    link=('left_shoulder', 'right_shoulder'),
                    id=7,
                    color=[51, 153, 255]),
                8:
                dict(
                    link=('left_shoulder', 'left_elbow'),
                    id=8,
                    color=[0, 255, 0]),
                9:
                dict(
                    link=('right_shoulder', 'right_elbow'),
                    id=9,
                    color=[255, 128, 0]),
                10:
                dict(
                    link=('left_elbow', 'left_wrist'),
                    id=10,
                    color=[0, 255, 0]),
                11:
                dict(
                    link=('right_elbow', 'right_wrist'),
                    id=11,
                    color=[255, 128, 0]),
                12:
                dict(
                    link=('left_eye', 'right_eye'),
                    id=12,
                    color=[51, 153, 255]),
                13:
                dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
                14:
                dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
                15:
                dict(
                    link=('left_eye', 'left_ear'), id=15, color=[51, 153,
                                                                 255]),
                16:
                dict(
                    link=('right_eye', 'right_ear'),
                    id=16,
                    color=[51, 153, 255]),
                17:
                dict(
                    link=('left_ear', 'left_shoulder'),
                    id=17,
                    color=[51, 153, 255]),
                18:
                dict(
                    link=('right_ear', 'right_shoulder'),
                    id=18,
                    color=[51, 153, 255])
            }),
            joint_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0,
                1.0, 1.2, 1.2, 1.5, 1.5
            ],
            sigmas=[
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
            ])))
