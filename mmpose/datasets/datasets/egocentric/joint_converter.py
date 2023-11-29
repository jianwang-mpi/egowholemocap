from mmpose.data.keypoints_mapping.renderpeople import render_people_joint_names
from mmpose.data.keypoints_mapping.renderpeople_old import render_people_joint_names as render_people_old_joint_names
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_joint_names
# from mmpose.data.keypoints_mapping.mo2cap2_with_hands import mo2cap2_with_hands_joint_names
from mmpose.data.keypoints_mapping.smplx import smplx_joint_names
from mmpose.data.keypoints_mapping.mano import mano_joint_names, mano_left_hand_joint_names, mano_right_hand_joint_names
from mmpose.data.keypoints_mapping.smplh import smplh_joint_names
from mmpose.data.keypoints_mapping.smpl import smpl_joint_names
from mmpose.data.keypoints_mapping.studio import studio_joint_names
from mmpose.data.keypoints_mapping.mo2cap2_with_head import mo2cap2_with_head_joint_names
from mmpose.data.keypoints_mapping.beat import beat_joint_names
import numpy as np


def dset_to_body_model(model_type='smplx', dset='coco', joints_to_ign=None,
                       use_face_contour=False, **kwargs):
    if joints_to_ign is None:
        joints_to_ign = []

    mapping = {}

    if model_type == 'smplx':
        keypoint_names = smplx_joint_names
    elif model_type == 'beat':
        keypoint_names = beat_joint_names
    elif model_type == 'mano_left':
        keypoint_names = mano_left_hand_joint_names
    elif model_type == 'mano_right':
        keypoint_names = mano_right_hand_joint_names
    elif model_type == 'studio':
        keypoint_names = studio_joint_names
    elif model_type == 'renderpeople':
        keypoint_names = render_people_joint_names
    elif model_type == 'renderpeople_old':
        keypoint_names = render_people_old_joint_names
    elif model_type == 'mo2cap2':
        keypoint_names = mo2cap2_joint_names
    elif model_type == 'smpl':
        keypoint_names = smpl_joint_names
    elif model_type == 'smplh':
        keypoint_names = smplh_joint_names
    else:
        raise ValueError('Unknown model dataset: {}'.format(model_type))

    if dset == 'mo2cap2':
        dset_keyp_names = mo2cap2_joint_names
    elif dset == 'beat':
        dset_keyp_names = beat_joint_names
    elif dset == 'mo2cap2_with_head':
        dset_keyp_names = mo2cap2_with_head_joint_names
    elif dset == 'mano_left':
        dset_keyp_names = mano_left_hand_joint_names
    elif dset == 'mano_right':
        dset_keyp_names = mano_right_hand_joint_names
    elif dset == 'studio':
        dset_keyp_names = studio_joint_names
    elif dset == 'renderpeople':
        dset_keyp_names = render_people_joint_names
    elif dset == 'renderpeople_old':
        dset_keyp_names = render_people_old_joint_names
    elif dset == 'smpl':
        dset_keyp_names = smpl_joint_names
    elif dset == 'smplh':
        dset_keyp_names = smplh_joint_names
    else:
        raise ValueError('Unknown dset dataset: {}'.format(dset))

    for idx, name in enumerate(keypoint_names):
        if 'contour' in name and not use_face_contour:
            continue
        if name in dset_keyp_names:
            mapping[idx] = dset_keyp_names.index(name)

    model_keyps_idxs = np.array(list(mapping.keys()), dtype=np.int32)
    dset_keyps_idxs = np.array(list(mapping.values()), dtype=np.int32)

    return dset_keyps_idxs, model_keyps_idxs
