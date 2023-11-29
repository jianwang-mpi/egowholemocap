#  Copyright Jian Wang @ MPI-INF (c) 2023.
fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
hand_pose_model = dict(
    type='EgoHandPose',
    pretrained='/HPS/EgoSyn/work/EgocentricHand/Hand4Whole_RELEASE/demo/hand/snapshot_12.pth.tar',
)

hand_detection_model = dict(
    type='Egocentric2DPoseEstimator',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
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
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=42,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict())

body_pose_model = dict(
    type='Egocentric3DPoseEstimator',
    # pretrained='/CT/EgoMocap/work/EgocentricFullBody/resources/pretrained_models/vitpose_base_coco_aic_mpii.pth',
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
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
        type='Heatmap3DNet',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(1024, 15 * 64),
        num_deconv_kernels=(4, 4),
        out_channels=15 * 64,
        heatmap_shape=(64, 64, 64),
        fisheye_model_path=fisheye_camera_path, joint_num=15,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True)
        ),
    train_cfg=dict(),
    test_cfg=dict(
        return_heatmap=False,
        return_confidence=True,
        sigma=3
    )
)


body_pose_model_with_heatmap = dict(
    type='Egocentric3DPoseEstimator',
    # pretrained='/CT/EgoMocap/work/EgocentricFullBody/resources/pretrained_models/vitpose_base_coco_aic_mpii.pth',
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
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
        type='Heatmap3DNet',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(1024, 15 * 64),
        num_deconv_kernels=(4, 4),
        out_channels=15 * 64,
        heatmap_shape=(64, 64, 64),
        fisheye_model_path=fisheye_camera_path, joint_num=15,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
        ),
    train_cfg=dict(),
    test_cfg=dict(return_2d_heatmap=True, sigma=3.0),
)